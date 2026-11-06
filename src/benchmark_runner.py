"""Versioned public-benchmark evaluation using the pinned upstream harness.

Task formatting, extraction and scoring remain in lm-evaluation-harness. Local
official data aliases only avoid network-dependent dataset loading. A smoke run
is explicitly distinguished from a full evaluation.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import time
import yaml
from delta_bank import DeltaBank,fingerprint
from mask_bundle import apply_masks
from task_plan import validate_plan
from structure_bundle import apply_structure

SOURCE_BYTES=Path(__file__).read_bytes()
SOURCE_SHA256=hashlib.sha256(SOURCE_BYTES).hexdigest()
HELPER_SOURCES={name:(Path(__file__).parent/name).read_bytes()
    for name in ['delta_bank.py','mask_bundle.py','task_plan.py','structure_bundle.py','weightlab.py']}


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str))
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--tasks-dir',required=True,type=Path)
    parser.add_argument('--tasks',default='idea4_gsm8k_cot_zeroshot_v1')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-new-tokens',type=int,default=512)
    parser.add_argument('--smoke',type=int,default=None)
    parser.add_argument('--delta',type=Path,default=None)
    parser.add_argument('--coefficient',type=float,default=None)
    parser.add_argument('--mask',type=Path,default=None)
    parser.add_argument('--plan',type=Path,default=None)
    parser.add_argument('--structure',type=Path,default=None)
    args=parser.parse_args()
    if args.mask is not None and args.delta is not None:raise ValueError('Combined interventions require a separate protocol')
    if args.structure is not None and (args.mask is not None or args.delta is not None):
        raise ValueError('Structural composition requires a separate verified protocol')
    if args.plan is not None and (args.delta is None or args.coefficient is not None):
        raise ValueError('Task plan requires a delta and excludes global coefficient override')
    args.output.mkdir(parents=True,exist_ok=True)
    config=dict(model=str(args.model.resolve()),tasks=args.tasks.split(','),
        task_files={p.name:digest(p) for p in args.tasks_dir.glob('*.yaml')
            if yaml.safe_load(p.read_text()).get('task') in args.tasks.split(',')},
        batch_size=args.batch_size,max_new_tokens=args.max_new_tokens,smoke_limit=args.smoke,
        chat_template=True,enable_thinking=False,num_fewshot=0,evaluation_seed=20261106,
        harness_version=importlib.metadata.version('lm_eval'))
    config['implementation_files']={name:hashlib.sha256(data).hexdigest() for name,data in HELPER_SOURCES.items()}
    if args.delta is not None:
        config['delta']=dict(directory=str(args.delta.resolve()),manifest_sha256=digest(args.delta/'manifest.json'),
            factors_sha256=digest(args.delta/'adapter.safetensors'),coefficient_override=args.coefficient)
    if args.mask is not None:
        config['mask']=dict(directory=str(args.mask.resolve()),manifest_sha256=digest(args.mask/'manifest.json'),
            payload_sha256=digest(args.mask/'masks.npz'))
    if args.plan is not None:config['plan']=dict(file=str(args.plan.resolve()),sha256=digest(args.plan))
    if args.structure is not None:config['structure']=dict(file=str(args.structure.resolve()),sha256=digest(args.structure))
    if (args.output/'protocol.json').exists():
        if json.loads((args.output/'protocol.json').read_text())!=config:raise ValueError('Run config changed')
        if (args.output/'complete.json').exists():
            print('Already complete; preserving results',flush=True)
            return
    else:save(args.output/'protocol.json',config)
    (args.output/('runner_source_'+SOURCE_SHA256+'.py')).write_bytes(SOURCE_BYTES)
    helper_dir=args.output/('helpers_'+SOURCE_SHA256);helper_dir.mkdir(exist_ok=True)
    for name,data in HELPER_SOURCES.items():(helper_dir/name).write_bytes(data)
    lock=open(f'/tmp/idea4-{os.getuid()}-gpu-{os.environ.get("CUDA_VISIBLE_DEVICES","default")}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    import torch
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    torch.set_num_threads(4)
    # v0.4.9.2 pretty_print_task assumes YAMLs live under lm_eval/tasks.
    # Install only our data-loading aliases in this private benchmark venv;
    # do not patch any formatter, metric, filter, or model implementation.
    task_dir=Path(lm_eval.__file__).resolve().parent/'tasks'/'idea4_local'
    task_dir.mkdir(exist_ok=True)
    for path in args.tasks_dir.glob('*.yaml'):
        shutil.copy2(path,task_dir/path.name)
    files={p.name:digest(p) for p in args.model.glob('*') if p.suffix in ('.json','.safetensors','.jinja')}
    identity=dict(source=config['model'],files=files)
    identity_path=args.output/'model_fingerprint.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('Model weights/configuration changed; cannot reuse this response cache')
    save(identity_path,identity)
    started=time.perf_counter()
    model=HFLM(pretrained=str(args.model.resolve()),dtype='bfloat16',device='cuda:0',
        batch_size=args.batch_size,max_length=2048,trust_remote_code=False,
        attn_implementation='sdpa',enable_thinking=False,chat_template_args={'enable_thinking':False})
    scope=contextlib.nullcontext()
    if args.mask is not None:
        apply_masks(model.model,args.mask,files)
    if args.structure is not None:apply_structure(model.model,args.structure,files)
    if args.delta is not None:
        delta_meta=json.loads((args.delta/'manifest.json').read_text())
        actual={name:files.get(name) for name in delta_meta['base_files']}
        if fingerprint(actual)!=delta_meta['base_id']:raise ValueError('Delta does not match benchmark base')
        bank=DeltaBank(model.model,fingerprint(actual))
        bank.register('evaluation',args.delta,args.coefficient)
        coefficients=None
        if args.plan is not None:
            coefficients=validate_plan(json.loads(args.plan.read_text()),delta_meta,model.model.config.num_hidden_layers)
        scope=bank.activate('evaluation',module_coefficients=coefficients)
    with scope:
        results=lm_eval.simple_evaluate(model=model,tasks=config['tasks'],num_fewshot=0,
            task_manager=TaskManager(include_path=str(task_dir)),
            batch_size=args.batch_size,limit=args.smoke,log_samples=True,
            use_cache=str(args.output/'responses'),bootstrap_iters=1000,
            apply_chat_template=True,gen_kwargs=dict(max_gen_toks=args.max_new_tokens,do_sample=False),
            random_seed=20261106,numpy_random_seed=20261106,torch_random_seed=20261106,fewshot_random_seed=20261106)
    if not results:raise RuntimeError('Harness returned no results')
    samples=results.pop('samples')
    for name,rows in samples.items():
        with open(args.output/(name+'.samples.jsonl'),'w') as stream:
            for row in rows:stream.write(json.dumps(row,default=str)+'\n')
    save(args.output/'results.json',results)
    # lm-eval emits a sample record per document AND per filter. GSM8K has
    # strict-match and flexible-extract, so 2638 logs still mean 1319 questions.
    raw_counts={name:len(rows) for name,rows in samples.items()}
    counts={name:len({row['doc_id'] for row in rows}) for name,rows in samples.items()}
    if args.smoke is None and 'idea4_gsm8k_cot_zeroshot_v1' in counts:
        if counts['idea4_gsm8k_cot_zeroshot_v1']!=1319:raise AssertionError('Not a complete GSM8K test')
    if args.smoke is None:
        for name,n in [('idea4_arc_challenge_v1',1172),('idea4_hellaswag_v1',10042)]:
            if name in counts and counts[name]!=n:raise AssertionError('Incomplete benchmark: '+name)
    save(args.output/'complete.json',dict(status='smoke_complete' if args.smoke else 'full_complete',
        samples=counts,raw_sample_records=raw_counts,elapsed_seconds=time.perf_counter()-started,
        torch=torch.__version__,transformers=importlib.metadata.version('transformers'),
        gpu=torch.cuda.get_device_name(),code_sha256=SOURCE_SHA256))
    print(json.dumps(results['results'],indent=2),flush=True)


if __name__=='__main__':main()
