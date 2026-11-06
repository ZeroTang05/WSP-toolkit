"""Original Qwen3.5 base practice traces at the author's sampling/token settings.

Training-derived GSM examples only. This is failure-mechanism evidence, not
MATH500 replication, a public accuracy estimate, or an independent edit discovery.
"""
import argparse
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=8)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh output required; do not restart live generation')
    if not 1<=a.n<=128:raise ValueError('1..128 practice examples required')
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    manifest=json.loads((a.model/'idea4_wsp_base_manifest.json').read_text())
    if manifest['revision']!='2fc06364715b967f1860aea9cf38778875588b17':raise ValueError('Pinned original base required')
    for name,entry in manifest['files'].items():
        if sha(a.model/name)!=entry['sha256']:raise ValueError('Changed original source')
    view=json.loads((a.view/'manifest.json').read_text());data=a.view/'calibration.jsonl'
    if sha(data)!=view['files']['calibration.jsonl']:raise ValueError('Changed calibration data')
    source=[json.loads(line) for line in data.read_text().splitlines()][:a.n]
    if len(source)!=a.n or any(r['source_split']!='train' or r['view_role']!='calibration' for r in source):
        raise ValueError('Training-derived calibration only')
    rows=[]
    for r in source:
        match=re.search(r'\nThe answer is (.+)\.\s*\Z',r['answer'])
        if match is None:raise ValueError('Canonical training-view final-answer suffix missing')
        rows.append(dict(id=r['id'],problem=r['prompt'],answer=match.group(1).strip(),source='gsm8k_train_calibration'))
    a.output.mkdir(parents=True);code=a.output/'code'
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,code/Path(__file__).name);sys.path.insert(0,str(code.resolve()))
    protocol=dict(model_revision=manifest['revision'],base_manifest_sha256=sha(a.model/'idea4_wsp_base_manifest.json'),
        calibration_sha256=sha(data),ids=[r['id'] for r in rows],seed=20261106,thinking=True,
        answer_transform='Canonical suffix from existing hashed view: The answer is N. -> N',
        temperature=.6,top_p=.95,top_k=20,presence_penalty=1.5,max_new_tokens=32768,n=1,
        source_sha256={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')},
        scope='Training-practice failure traces; not held-out benchmark accuracy or independent edit discovery. HF engine, no vLLM equivalence claim.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    lock=open(f'/tmp/idea4-{os.getuid()}-gpu-0.lock','a');waiting=time.perf_counter()
    print(json.dumps(dict(event='waiting_gpu_0',items=a.n)),flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
    wait_seconds=time.perf_counter()-waiting;started=time.perf_counter()
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
    from weight_programming import WeightStore,ForwardService,ForwardPolicy,PracticeSet
    from weight_programming.qwen35_backend import Qwen35Backend
    from weight_programming.grading import MathVerifyGrader
    torch.set_num_threads(4)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    store=WeightStore(a.model,a.output/'edits');backend=Qwen35Backend(model,tok,store)
    service=ForwardService(backend,store,PracticeSet(rows,role='calibration',name='gsm-training-practice'),
        a.output/'service',ForwardPolicy(),MathVerifyGrader())
    print(json.dumps(dict(event='practice_generation_start',items=a.n,gpu_lock_wait_seconds=wait_seconds)),flush=True)
    result=service.request(dict(type='generate',model='original',problems=[r['id'] for r in rows],n=1,max_tokens=32768))
    (a.output/'complete.json').write_text(json.dumps(dict(status='original_practice_generation_complete',
        summaries=result['summaries'],gpu_lock_wait_seconds=wait_seconds,active_seconds=time.perf_counter()-started,
        artifact_dir=result['artifact_dir'],scope=protocol['scope']),indent=2))
    print(json.dumps(dict(event='practice_generation_complete',summaries=result['summaries'])),flush=True)


if __name__=='__main__':main()
