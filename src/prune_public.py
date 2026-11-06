"""Public-calibration Wanda adaptation exported only as source-bound bitmasks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from weightlab import Lab,write_json
from paper_methods import InputEnergy,wanda_mask,WANDA_COMMIT
from contract_registry import ContractRegistry,ToolRequest
from prepare_training_views import read_jsonl
from mask_bundle import save_masks


NAMES=['prune_public.py','paper_methods.py','contract_registry.py','weightlab.py','mask_bundle.py','delta_bank.py','prepare_training_views.py','finqa_views.py','verify_artifacts.py']
SOURCES={name:(Path(__file__).parent/name).read_bytes() for name in NAMES}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--view',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fraction',type=float,required=True);p.add_argument('--calibration-n',type=int,default=128)
    a=p.parse_args()
    if not 0<a.fraction<1 or a.calibration_n<1:raise ValueError('Invalid pruning budget')
    raw=(a.view/'calibration.jsonl').read_bytes();view=json.loads((a.view/'manifest.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=view['files']['calibration.jsonl']:raise ValueError('Changed calibration view')
    rows=read_jsonl(a.view/'calibration.jsonl')
    if a.calibration_n>len(rows) or any(r['source_split']!='train' or r['view_role']!='calibration' for r in rows):
        raise ValueError('Not an allowed calibration view')
    rows=rows[:a.calibration_n]
    if a.output.exists():raise FileExistsError('Use a fresh output directory')
    a.output.mkdir(parents=True)
    (a.output/'code').mkdir()
    for name,data in SOURCES.items():(a.output/'code'/name).write_bytes(data)
    registry=ContractRegistry();spec=registry.methods['wanda.row_unstructured.v1']
    if spec['reference_commit']!=WANDA_COMMIT:raise ValueError('Wrong author reference')
    request=ToolRequest(spec['id'],spec['observation'],spec['selection'],spec['mutation'],'none','calibration')
    protocol=dict(model=str(a.model.resolve()),fraction=a.fraction,calibration_n=a.calibration_n,
        calibration_sha256=view['files']['calibration.jsonl'],calibration_ids=[r['id'] for r in rows],
        method=spec['id'],contract_spec_sha256=registry.spec_sha256,
        code_sha256={name:hashlib.sha256(data).hexdigest() for name,data in SOURCES.items()},
        adaptation='Official GSM8K training-derived calibration, variable-length prompt and reference rationale; sequential decoder layers; all linear projections except lm_head; no backward or recovery training',
        selection='fixed fraction and first 128 calibration records; no test/development selection',
        scope='Wanda primitives with architecture/data adaptation, not reproduction of paper C4 benchmark')
    write_json(a.output/'protocol.json',protocol)
    lab=Lab(a.model);base=lab.manifest();write_json(a.output/'base_manifest.json',base)
    started=time.perf_counter();masks={};certificates=[]
    with torch.inference_mode():
        for i,layer in enumerate(lab.layers):
            modules={n:m for n,m in layer.named_modules() if isinstance(m,torch.nn.Linear)}
            stats={n:InputEnergy(m.in_features,m.weight.device) for n,m in modules.items()}
            handles=[m.register_forward_pre_hook(lambda m,x,n=n:stats[n].add(x[0])) for n,m in modules.items()]
            try:
                for row in rows:
                    prefix=lab.tokenizer.encode(lab.prefix(row),add_special_tokens=False)
                    target=lab.tokenizer.encode(row['answer'],add_special_tokens=False)
                    lab.model(torch.tensor([prefix+target],device='cuda'),use_cache=False)
            finally:
                for handle in handles:handle.remove()
            for name,module in modules.items():
                energy=stats[name].energy()
                mask=wanda_mask(module.weight,energy,a.fraction)
                certificate=registry.apply_wanda_linear(request,module,energy,a.fraction)
                masks[f'model.layers.{i}.{name}']=mask.cpu()
                certificates.append(dict(layer=i,module=name,sequences=stats[name].sequences,**certificate))
            print(json.dumps(dict(layer=i,elapsed_seconds=time.perf_counter()-started)),flush=True)
    meta=save_masks(masks,base,a.output/'mask',dict(paper=spec['paper'],reference_commit=WANDA_COMMIT,protocol=protocol))
    write_json(a.output/'certificates.json',certificates)
    write_json(a.output/'complete.json',dict(status='calibrated_mask_exported',
        elapsed_seconds=time.perf_counter()-started,masked=sum(v['masked'] for v in meta['targets'].values()),
        prunable_parameters=sum(v.numel() for v in masks.values()),
        full_checkpoint_saved=False,representation='base plus packed bitmask; dense runtime'))


if __name__=='__main__':main()
