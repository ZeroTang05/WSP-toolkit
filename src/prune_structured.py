"""Declared LLM-Pruner-inspired MLP Taylor adaptation on public calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import random
import torch
from weightlab import Lab,write_json,cast_floating_parameters
from paper_methods import run_taylor
from contract_registry import ContractRegistry,ToolRequest
from prepare_training_views import read_jsonl
from delta_bank import fingerprint,sha


NAMES=['prune_structured.py','paper_methods.py','weightlab.py','contract_registry.py',
    'prepare_training_views.py','finqa_views.py','verify_artifacts.py','delta_bank.py','structure_bundle.py']
SOURCES={name:(Path(__file__).parent/name).read_bytes() for name in NAMES}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--view',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fraction',type=float,required=True);p.add_argument('--calibration-n',type=int,default=128)
    p.add_argument('--calibration-seed',type=int)
    p.add_argument('--statistics-dtype',choices=['bfloat16','float32'],default='bfloat16')
    a=p.parse_args();meta=json.loads((a.view/'manifest.json').read_text())
    if sha(a.view/'calibration.jsonl')!=meta['files']['calibration.jsonl']:raise ValueError('Changed calibration source')
    rows=read_jsonl(a.view/'calibration.jsonl')
    if not 1<=a.calibration_n<=len(rows) or not 0<a.fraction<1:raise ValueError('Invalid compression budget')
    if any(row['view_role']!='calibration' or row['source_split']!='train' for row in rows):raise ValueError('Wrong data permissions')
    if a.calibration_seed is None:rows=rows[:a.calibration_n]
    else:rows=[rows[i] for i in sorted(random.Random(a.calibration_seed).sample(range(len(rows)),a.calibration_n))]
    registry=ContractRegistry();spec=registry.methods['taylor.gated_mlp.first_order.v1']
    registry.validate(ToolRequest(spec['id'],spec['observation'],spec['selection'],spec['mutation'],'observe','calibration'))
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);(a.output/'code').mkdir();(a.output/'contracts').mkdir()
    for name,data in SOURCES.items():(a.output/'code'/name).write_bytes(data)
    (a.output/'contracts/methods.json').write_bytes(registry.path.read_bytes())
    config=dict(model=str(a.model.resolve()),fraction=a.fraction,calibration_n=len(rows),
        calibration_seed=a.calibration_seed,statistics_dtype=a.statistics_dtype,buffer_policy='preserve native buffer dtype and values',
        calibration_sha256=meta['files']['calibration.jsonl'],calibration_ids=[r['id'] for r in rows],
        method_id=spec['id'],contract_spec_sha256=registry.spec_sha256,
        objective='mean example answer-token CE including EOS; gradients observed, no optimizer updates',
        adaptation='gated MLP only, uniform width aligned to 128; first-order scores; no recovery; not full LLM-Pruner',
        code_sha256={name:hashlib.sha256(data).hexdigest() for name,data in SOURCES.items()})
    write_json(a.output/'protocol.json',config)
    lab=Lab(a.model);base=lab.manifest();write_json(a.output/'base_manifest.json',base)
    if a.statistics_dtype=='float32':cast_floating_parameters(lab.model,torch.float32)
    before=sum(p.numel() for p in lab.model.parameters());start=time.perf_counter()
    evidence=run_taylor(lab,rows,a.fraction,include_eos=True)
    after=sum(p.numel() for p in lab.model.parameters())
    selector=dict(version=1,kind='gated_mlp_channel_selector',base_id=fingerprint(base['files']),
        base_files=base['files'],kept_indices=evidence['kept_indices'],method_id=spec['id'],
        implementation_grade=spec['implementation_grade'],protocol_sha256=sha(a.output/'protocol.json'))
    write_json(a.output/'selector.json',selector);write_json(a.output/'method_evidence.json',evidence)
    write_json(a.output/'complete.json',dict(status='structural_selector_exported',base_parameters=before,
        compressed_parameters=after,total_parameter_reduction=1-after/before,elapsed_seconds=time.perf_counter()-start,
        fp32_mask_slice_max_error=evidence['fp32_equivalence_max_error'],full_checkpoint_saved=False,
        note='Legal dense architecture and local equivalence only; quality preservation and acceleration require separate evaluation.'))


if __name__=='__main__':main()
