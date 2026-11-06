"""Validate two functional deltas on a real base without saving full variants.

The two roles use scaled versions of an existing synthetic pilot adapter. This
is runtime correctness evidence, not an application or specialization result.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import torch
from delta_bank import DeltaBank,export_legacy_adapter
from weightlab import Lab,write_json


def parameter_hash(model):
    h=hashlib.sha256()
    for name,value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter',required=True,type=Path)
    parser.add_argument('--base-manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'complete.json').exists():raise FileExistsError('Smoke evidence already exists')
    for role,coefficient in [('half',.5),('full',1.)]:
        if not (args.output/role).exists():
            export_legacy_adapter(args.adapter,args.base_manifest,args.output/role,coefficient=coefficient)
    source=json.loads(args.base_manifest.read_text())['source']
    lab=Lab(source)
    bank=DeltaBank.from_verified_base(lab.model,args.base_manifest)
    for role in ['half','full']:bank.register(role,args.output/role)
    lab.model.float().requires_grad_(False)
    before=parameter_hash(lab.model)
    ids=lab.tokenizer('A short arithmetic question: what is twelve plus nine?',return_tensors='pt').input_ids.to('cuda')
    results={}
    for role in ['half','full']:
        with torch.inference_mode(),bank.activate(role):
            actual=lab.model(ids,use_cache=False).logits.float().cpu()
        reference=deepcopy(lab.model)
        meta,factors=bank.roles[role]
        with torch.no_grad():
            for name,spec in meta['targets'].items():
                target=reference.get_submodule(name)
                A=factors[name+'.A'].to(target.weight.device)
                B=factors[name+'.B'].to(target.weight.device)
                target.weight.add_(meta['coefficient']*spec['alpha']/spec['rank']*(B@A))
        with torch.inference_mode():expected=reference(ids,use_cache=False).logits.float().cpu()
        error=(actual-expected).abs()
        if error.max().item()>.001:raise AssertionError(f'Functional/materialized mismatch: {error.max()}')
        results[role]=dict(max_logit_error=error.max().item(),mean_logit_error=error.mean().item(),
            matching_argmax_fraction=(actual.argmax(-1)==expected.argmax(-1)).float().mean().item())
        del reference
        gc.collect()
        torch.cuda.empty_cache()
    after=parameter_hash(lab.model)
    if before!=after:raise AssertionError('Base parameters changed')
    base_bytes=sum(p.stat().st_size for p in Path(source).iterdir() if p.is_file())
    delta_bytes=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
    write_json(args.output/'complete.json',dict(status='runtime_smoke_only',roles=results,
        base_parameters_immutable=True,base_parameter_hash=before,shared_base_disk_bytes=base_bytes,
        delta_bank_disk_bytes=delta_bytes,estimated_two_full_checkpoint_bytes=2*base_bytes,
        bank=bank.storage(),note='Two coefficients of one synthetic pilot adapter; not two independently specialized agents. FP32 equivalence checked; BF16 inference quality still needs benchmark evaluation.'))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
