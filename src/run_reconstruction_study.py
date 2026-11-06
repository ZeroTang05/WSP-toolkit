"""Real-checkpoint local reconstruction study, not a task-accuracy benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--observation',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use fresh output')
    a.output.mkdir(parents=True); code=a.output/'code'
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,code/Path(__file__).name);sys.path.insert(0,str(code.resolve()))
    import numpy as np
    import torch
    from weight_programming import WeightStore, input_hessian, sparsegpt_reconstruct
    torch.set_num_threads(4);start=time.perf_counter()
    # The earlier frozen observation contains source hashes and actual variant ID.
    protocol=json.loads((a.observation/'protocol.json').read_text())
    meta=a.observation/'service/request_0001/meta.json';metadata=json.loads(meta.read_text())
    if metadata['variant_id']!=metadata['original_variant_id']:raise ValueError('Original observations required')
    store=WeightStore(a.model,a.output/'edits')
    base_id=store.snapshot(with_edits=False)['base_id']
    if base_id!=metadata['binding']['base_id']:raise ValueError('Observation/base mismatch')
    layer=protocol['request']['return']['mlp_act'][0]
    name=f'model.layers.{layer}.mlp.down_proj.weight'; w=store.load_original(name).float()
    folder=meta.parent; act=np.load(folder/f'mlp_{layer}.npy',mmap_mode='r');lengths=np.load(folder/'lengths.npy')
    if len(lengths)!=128:raise ValueError('Predeclared 64/64 calibration split requires 128 texts')
    tensors=[]
    for subset in [range(64),range(64,128)]:
        tensors.append(torch.from_numpy(np.concatenate([act[i,:int(lengths[i])] for i in subset])).double())
    fit,hold=tensors; h=input_hessian(fit);hh=input_hessian(hold)
    source={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')}
    (a.output/'protocol.json').write_text(json.dumps(dict(
        observation_metadata_sha256=sha(meta),observation_protocol_sha256=sha(a.observation/'protocol.json'),
        base_id=base_id,weight=name,shape=list(w.shape),source_sha256=source,
        fitting_ids=protocol['calibration_ids'][:64],heldout_ids=protocol['calibration_ids'][64:],
        fit_tokens=len(fit),heldout_tokens=len(hold),sparsities=[.2,.5],blocksize=128,relative_damping=.01,
        scope='Single matrix on reference-text activations; held-out within calibration, not public task test. Layer inherited from observation tutorial, not selected as most important.'),indent=2))
    print(json.dumps(dict(event='hessians_complete',fit_tokens=len(fit),heldout_tokens=len(hold))),flush=True)
    def energy(delta, moment):return float(((delta.double()@moment)*delta.double()).sum()/2)
    denominator_fit=energy(w,h);denominator_hold=energy(w,hh)
    records=[]
    def record(method,sparsity,candidate,mask,diagnostics=None):
        delta=candidate.double()-w.double()
        item=dict(method=method,sparsity=sparsity,removed_entries=int(mask.sum()),
                  fit_relative_output_squared_error=energy(delta,h)/denominator_fit,
                  heldout_relative_output_squared_error=energy(delta,hh)/denominator_hold,
                  relative_weight_squared_error=float(delta.square().sum()/w.double().square().sum()),
                  diagnostics=diagnostics)
        records.append(item);print(json.dumps(item),flush=True)
    for s in [.2,.5]:
        scores={'magnitude':w.double().abs(),'wanda':w.double().abs()*torch.sqrt(h.diagonal()/2)[None,:]}
        # Match the existing rowwise Wanda/magnitude selection convention.
        for method,score in scores.items():
            mask=torch.zeros_like(w,dtype=torch.bool)
            mask.scatter_(1,torch.argsort(score,dim=1,stable=True)[:,:int(w.shape[1]*s)],True)
            record(method+'_zero',s,w.masked_fill(mask,0),mask)
            restored,_,info=sparsegpt_reconstruct(w,h,prune_mask=mask,blocksize=128)
            record(method+'_same_mask_compensated',s,restored,mask,info)
        restored,mask,info=sparsegpt_reconstruct(w,h,sparsity=s,blocksize=128)
        record('sparsegpt_adaptive_mask_zero',s,w.masked_fill(mask,0),mask)
        record('sparsegpt_adaptive_compensated',s,restored,mask,info)
    (a.output/'results.json').write_text(json.dumps(records,indent=2))
    (a.output/'complete.json').write_text(json.dumps(dict(status='local_reconstruction_study_complete',
        elapsed_seconds=time.perf_counter()-start,conditions=len(records),
        observation_array_sha256=sha(folder/f'mlp_{layer}.npy'),
        results_sha256=sha(a.output/'results.json')),indent=2))


if __name__=='__main__':main()
