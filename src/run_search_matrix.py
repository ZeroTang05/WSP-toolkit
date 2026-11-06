"""Frozen-source, fail-fast research runner; waits for existing training jobs.

GPU execution remains serialized by Lab. This runner never trains a direction,
reads test outcomes, or silently retries a partial search.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


FILES=['optimize_public_v2.py','search_v2.py','search_metrics.py','task_plan.py',
    'delta_bank.py','weightlab.py','prepare_training_views.py','finqa_views.py','contract_registry.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--models-root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();result_root=root/'benchmarks/search-v2';result_root.mkdir(exist_ok=True)
    lock=open(result_root/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files={name:(root/'src'/name).read_bytes() for name in FILES}
    spec=(root/'contracts/methods.json').read_bytes()
    signature=hashlib.sha256(b''.join(files.values())+spec).hexdigest()
    frozen=root/'research/frozen'/('search-v2-'+signature)
    (frozen/'code').mkdir(parents=True,exist_ok=True);(frozen/'contracts').mkdir(exist_ok=True)
    for name,data in files.items():(frozen/'code'/name).write_bytes(data)
    (frozen/'contracts/methods.json').write_bytes(spec)
    protocol=dict(source_bundle_sha256=signature,models=['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct'],
        seeds=[11,22,33],policies=['fixed','random','greedy','bo','llm'],queries=21,domain_n=64,guard_n=32,
        guard_mode='norm_ce',guard_margin=.02,source=str(frozen),scope='C1 rank8; equal-query study; complete cost matching still required')
    if (result_root/'protocol.json').exists() and json.loads((result_root/'protocol.json').read_text())!=protocol:
        raise ValueError('Matrix source or protocol changed; use a new version')
    (result_root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

    def run(model,seed,policy,smoke=False):
        delta=root/'benchmarks/optimization/gsm8k'/model/'lora'/f'seed{seed}'/'delta'
        output=(root/'benchmarks/search-v2-smoke'/policy) if smoke else result_root/model/f'seed{seed}'/policy
        if (output/'complete.json').exists():return
        if output.exists():raise FileExistsError('Preserve and investigate incomplete search: '+str(output))
        command=[str(root/'.venv/bin/python'),'-u',str(frozen/'code/optimize_public_v2.py'),
            '--model',str(a.models_root/model),'--delta',str(delta),'--view',str(root/'benchmarks/views/gsm8k-v1'),
            '--guard',str(root/'benchmarks/raw/idea4_arc_challenge_v1/validation.jsonl'),
            '--policy',policy,'--seed',str(seed),'--queries','3' if smoke else '21',
            '--domain-n','4' if smoke else '64','--guard-n','4' if smoke else '32','--output',str(output)]
        if smoke:command+=['--smoke']
        if policy=='llm':command+=['--controller',str(a.models_root/'Qwen3-4B-Instruct-2507')]
        log=result_root/(('smoke_' if smoke else '')+model+f'_seed{seed}_'+policy+'.log')
        print(json.dumps(dict(event='start',output=str(output))),flush=True)
        with open(log,'w') as stream:subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        complete=json.loads((output/'complete.json').read_text())
        if complete['queries']!=(3 if smoke else 21):raise ValueError('Wrong query count')
        print(json.dumps(dict(event='complete',output=str(output),coordinates=complete['coordinates'])),flush=True)

    for policy in ['fixed','bo','llm']:run('Qwen3-0.6B',11,policy,True)
    for model in protocol['models']:
        for seed in protocol['seeds']:
            dependency=root/'benchmarks/optimization/gsm8k'/model/'lora'/f'seed{seed}'/'complete.json'
            deadline=time.monotonic()+86400
            if not dependency.exists():print(json.dumps(dict(event='waiting_for_training',file=str(dependency))),flush=True)
            while not dependency.exists():
                if time.monotonic()>deadline:raise TimeoutError(str(dependency))
                time.sleep(30)
            for policy in protocol['policies']:run(model,seed,policy)
    (result_root/'matrix_complete.json').write_text(json.dumps(dict(status='development_search_matrix_complete',protocol=protocol),indent=2))


if __name__=='__main__':main()
