"""Full original-model MATH-500 evaluation or evaluation of frozen replacements.

GPU 0 is locked for one problem (four samples) at a time, then the model is
offloaded before releasing it. No task subset or shorter-generation shortcut.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replacements',type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh final evaluation required; do not restart live inference')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    if importlib.metadata.version('transformers')!='5.3.0':raise ValueError('Separate pinned Qwen3.5 environment required')
    base=json.loads((a.model/'idea4_wsp_base_manifest.json').read_text())
    if base['revision']!='2fc06364715b967f1860aea9cf38778875588b17':raise ValueError('Original target revision required')
    for name,record in base['files'].items():
        if sha(a.model/name)!=record['sha256']:raise ValueError('Changed original model')
    data=json.loads((a.dataset/'manifest.json').read_text())
    if data['role']!='final_evaluation' or data['revision']!='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be':raise ValueError('Pinned final dataset required')
    if sha(a.dataset/'test.jsonl')!=data['files']['test.jsonl']['sha256']:raise ValueError('Changed final dataset')
    original=[json.loads(x) for x in (a.dataset/'test.jsonl').read_text().split('\n') if x.strip()]
    if len(original)!=500 or len({r['unique_id'] for r in original})!=500:raise ValueError('Full official 500 required')
    rows=[dict(id=r['unique_id'],problem=r['problem'],answer=r['answer'],source='MATH-500',
        subject=r['subject'],level=r['level'],data_role='evaluation',source_split='test') for r in original]
    a.output.mkdir(parents=True);code=a.output/'code';code.mkdir()
    shutil.copy2(__file__,code/Path(__file__).name)
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(code.resolve()))
    from weight_programming import WeightStore,ForwardPolicy
    from weight_programming.qwen35_backend import Qwen35Backend
    from weight_programming.evaluation import evaluate_math
    from weight_programming.grading import MathVerifyGrader
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
    torch.set_num_threads(4);store=WeightStore(a.model,a.output/'frozen-edits')
    replacement_source=None
    if a.replacements is not None:
        root=a.replacements.resolve();manifest=json.loads((root/'manifest.json').read_text())
        if manifest['base_id']!=store.snapshot(False)['base_id'] or manifest['kind']!='replacement_tensors_not_a_complete_model':raise ValueError('Wrong replacement base/type')
        for name,entry in manifest['tensors'].items():
            path=(root/entry['file']).resolve()
            if path.parent!=root or sha(path)!=entry['sha256']:raise ValueError('Changed replacement tensor')
            values=load_file(str(path))
            if set(values)!={name}:raise ValueError('Wrong replacement payload')
            store.save_edit(name,values[name].float())
        replacement_source=dict(path=str(root),manifest_sha256=sha(root/'manifest.json'))
    setup=dict(dataset=data,base_manifest_sha256=sha(a.model/'idea4_wsp_base_manifest.json'),replacements=replacement_source,
        source_sha256={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')},
        samples=4,max_new_tokens=32768,thinking=True,backend='HF 5.3.0 / BF16 / SDPA',
        scheduling='Exclusive GPU0 per problem; offload model before releasing lock; waiting and transfer costs recorded',
        scope='Independent full evaluation; no vLLM or original-harness equivalence claim')
    (a.output/'setup.json').write_text(json.dumps(setup,indent=2))
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True);backend=Qwen35Backend(model,tokenizer,store)
    lock=open(f'/tmp/idea4-{os.getuid()}-gpu-0.lock','a')
    @contextmanager
    def gpu_case(position):
        waiting=time.perf_counter();print(json.dumps(dict(event='waiting_gpu_0',position=position+1,items=500)),flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX);wait=time.perf_counter()-waiting;started=time.perf_counter()
        try:
            model.to('cuda:0');backend.device=torch.device('cuda:0')
            yield
        finally:
            try:
                model.to('cpu');backend.device=torch.device('cpu');torch.cuda.empty_cache()
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
            with (a.output/'scheduling.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(position=position+1,gpu_wait_seconds=wait,resident_seconds=time.perf_counter()-started))+'\n')
    result=evaluate_math(backend,store,rows,a.output/'evaluation',ForwardPolicy(),MathVerifyGrader(),
        samples=4,case_context=gpu_case,edited=a.replacements is not None)
    (a.output/'complete.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
