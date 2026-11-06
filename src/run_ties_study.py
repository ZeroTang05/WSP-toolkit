"""Compile real trained updates with TIES; no task-effect claim from geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh study output required')
    import torch
    from safetensors.torch import save_file
    from delta_programming import load_effective
    from weight_programming.merging import ties_merge
    torch.set_num_threads(4);started=time.perf_counter()
    loaded=[load_effective(path) for path in a.sources];base_ids=[meta['base_id'] for meta,_ in loaded]
    if len(set(base_ids))!=1:raise ValueError('Sources do not share a base')
    if any(set(pairs)!=set(loaded[0][1]) for _,pairs in loaded):raise ValueError('Different module scopes')
    a.output.mkdir(parents=True);code=a.output/'code';code.mkdir()
    for name in ['run_ties_study.py','delta_programming.py','delta_bank.py','analyze_delta.py']:
        shutil.copy2(Path(__file__).parent/name,code/name)
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    origin=[dict(path=str(path.resolve()),manifest_sha256=sha(path/'manifest.json'),
        tensor_sha256=meta['tensor_sha256'],training_protocol_sha256=sha(path.parent/'protocol.json'))
        for path,(meta,_) in zip(a.sources,loaded)]
    protocol=dict(sources=origin,base_id=base_ids[0],density=.2,zero_sign='reference_majority',
        scope='TIES over the complete effective q/v update scope, not raw LoRA factors or entire model',
        controls=['ordinary arithmetic mean','norm-matched ordinary mean'],
        source_sha256={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')},
        interpretation='Geometric/parameter study only. No claim of task improvement, compression or original-WSP discovery.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    deltas=[{name:(B@A).float() for name,(A,B) in pairs.items()} for _,pairs in loaded]
    del loaded
    merged,evidence=ties_merge(deltas,base_ids,density=.2)
    names=sorted(merged);dot=soup_sq=ties_sq=0.;distance_sq=0.
    modules={}
    for name in names:
        ordinary=torch.stack([d[name] for d in deltas]).mean(0)
        m=merged[name];a64=ordinary.double();m64=m.double()
        aa=float(a64.square().sum());bb=float(m64.square().sum());ab=float((a64*m64).sum())
        soup_sq+=aa;ties_sq+=bb;dot+=ab;distance_sq+=float((a64-m64).square().sum())
        modules[name]=dict(mean_norm=aa**.5,ties_norm=bb**.5,
            cosine=ab/(aa*bb)**.5 if aa*bb else None,**evidence['modules'][name])
    metadata=dict(kind='effective_dense_deltas_not_replacement_weights',base_id=base_ids[0],method='TIES',density='0.2')
    save_file({name:m.contiguous() for name,m in merged.items()},str(a.output/'dense_updates.safetensors'),metadata=metadata)
    analysis=dict(**evidence,ordinary_mean_norm=soup_sq**.5,ties_to_mean_norm=(ties_sq/soup_sq)**.5 if soup_sq else None,
        cosine_to_ordinary_mean=dot/(soup_sq*ties_sq)**.5 if soup_sq*ties_sq else None,
        distance_to_ordinary_mean=distance_sq**.5,modules_comparison=modules,
        dense_update_bytes=(a.output/'dense_updates.safetensors').stat().st_size,
        dense_update_sha256=sha(a.output/'dense_updates.safetensors'),elapsed_seconds=time.perf_counter()-started,
        effect_status='No task forwards performed; geometry is not capability evidence')
    (a.output/'analysis.json').write_text(json.dumps(analysis,indent=2))
    (a.output/'complete.json').write_text(json.dumps(dict(status='real_effective_delta_merge_complete',
        elapsed_seconds=analysis['elapsed_seconds'],dense_update_sha256=analysis['dense_update_sha256']),indent=2))
    print(json.dumps({k:v for k,v in analysis.items() if k not in ['modules','modules_comparison']},indent=2),flush=True)


if __name__=='__main__':main()
