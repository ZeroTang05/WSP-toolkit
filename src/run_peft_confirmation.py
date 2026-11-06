"""Standard PEFT backend confirmation with the same three trained experts."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


FILES=['finqa_workflow_v2.py','finqa_roles.py','finqa_executor.py','finqa_views.py',
    'delta_bank.py','weightlab.py','prepare_training_views.py','peft_bank.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();dest=root/'benchmarks/finqa-peft-v1';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={name:(root/'src'/name).read_bytes() for name in FILES}
    vendor={p.name:p.read_bytes() for p in (root/'vendor/finqa').iterdir() if p.is_file()}
    identity=hashlib.sha256(b''.join(sources.values())+b''.join(vendor[k] for k in sorted(vendor))).hexdigest()
    frozen=root/'research/frozen'/('peft-confirm-'+identity)
    (frozen/'code').mkdir(parents=True,exist_ok=True);(frozen/'vendor/finqa').mkdir(parents=True,exist_ok=True)
    for name,data in sources.items():(frozen/'code'/name).write_bytes(data)
    for name,data in vendor.items():(frozen/'vendor/finqa'/name).write_bytes(data)
    spec=dict(source=identity,seeds=[11,22,33],partitions=['dev','test'],backend='peft',mode='roles',
        scope='Same factors/data/role prompts; independent BF16 quality due to cast-order difference; not serving-system novelty')
    path=dest/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('PEFT study changed')
    path.write_text(json.dumps(spec,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for seed in spec['seeds']:
        args=[]
        for role in ['retriever','planner','verifier']:
            train=root/'benchmarks/optimization/finqa/Qwen3-1.7B'/role/f'seed{seed}'
            deadline=time.monotonic()+172800
            while not (train/'complete.json').exists():
                if time.monotonic()>deadline:raise TimeoutError(str(train))
                time.sleep(30)
            args+=['--'+role,str(train/'delta')]
        for partition,smoke in [('dev',8),('dev',None),('test',None)]:
            output=dest/f'seed{seed}'/('smoke' if smoke else 'full')/partition
            if (output/'complete.json').exists():continue
            command=[str(root/'.venv-bench/bin/python'),'-u',str(frozen/'code/finqa_workflow_v2.py'),
                '--model',str(a.model),'--inputs',str(root/'benchmarks/views/finqa-roles-v3'/(partition+'_inputs.jsonl')),
                '--labels',str(root/'benchmarks/raw/finqa'/(partition+'.json')),'--partition',partition,
                '--mode','roles','--backend','peft','--output',str(output)]+args
            if smoke:command+=['--smoke',str(smoke)]
            print(json.dumps(dict(event='start',output=str(output))),flush=True)
            with open(dest/(f'seed{seed}_{partition}'+('_smoke' if smoke else '')+'.log'),'w') as stream:
                subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            result=json.loads((output/'complete.json').read_text())
            if result['n']!=(smoke if smoke else 883 if partition=='dev' else 1147):raise ValueError('Incomplete PEFT partition')
            print(json.dumps(dict(event='complete',output=str(output))),flush=True)
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='peft_confirmation_complete'),indent=2))


if __name__=='__main__':main()
