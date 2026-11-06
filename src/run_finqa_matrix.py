"""Frozen FinQA main comparisons and explicitly labeled diagnostic ablations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


FILES=['finqa_workflow_v2.py','finqa_roles.py','finqa_executor.py','finqa_views.py',
    'delta_bank.py','weightlab.py','prepare_training_views.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path);a=p.parse_args();root=a.root.resolve()
    dest=root/'benchmarks/finqa-v2';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data={name:(root/'src'/name).read_bytes() for name in FILES}
    vendor={path.name:path.read_bytes() for path in (root/'vendor/finqa').iterdir() if path.is_file()}
    signature=hashlib.sha256(b''.join(data.values())+b''.join(vendor[k] for k in sorted(vendor))).hexdigest()
    frozen=root/'research/frozen'/('finqa-v2-'+signature)
    (frozen/'code').mkdir(parents=True,exist_ok=True);(frozen/'vendor/finqa').mkdir(parents=True,exist_ok=True)
    for name,value in data.items():(frozen/'code'/name).write_bytes(value)
    for name,value in vendor.items():(frozen/'vendor/finqa'/name).write_bytes(value)
    spec=dict(version='finqa-v2',source_bundle=signature,model=str(a.model.resolve()),
        baselines=['single','prompt','base_refine'],main_modes=['roles','multitask'],main_seeds=[11,22,33],
        ablations=['replicas','random_roles','no_verifier','no_retriever','oracle_retrieval'],ablation_seeds=[11],
        partitions=['dev','test'],oracle='gold support diagnostic only; excluded from deployable comparisons',
        capacity='rank8 shared multitask versus rank8 per-role; rank24 total-capacity control remains separate',
        budget='roles max4 calls/776 generated tokens; base_refine exactly4 calls, cap776; report realized cost',
        scope='Main optimized modes three seeds; seed11-only ablations do not establish seed-robust ablation effects')
    protocol=dest/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=spec:raise ValueError('FinQA matrix protocol changed')
    protocol.write_text(json.dumps(spec,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    train=root/'benchmarks/optimization/finqa/Qwen3-1.7B'
    index=json.loads((dest/'evaluation-index.json').read_text()) if (dest/'evaluation-index.json').exists() else []

    def wait(path):
        deadline=time.monotonic()+172800
        if not path.exists():print(json.dumps(dict(event='waiting_for_training',file=str(path))),flush=True)
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(30)

    def run(mode,partition,seed=11,smoke=None):
        if mode not in ['single','prompt','base_refine']:
            names=['multitask'] if mode=='multitask' else ['retriever','planner','verifier']
            for name in names:wait(train/name/f'seed{seed}'/'complete.json')
        output=dest/('smoke' if smoke else 'full')/f'seed{seed}'/mode/partition
        if (output/'complete.json').exists():
            if not smoke and not any(row['path']==str(output) for row in index):
                index.append(dict(mode=mode,partition=partition,seed=seed,path=str(output),diagnostic_oracle=mode=='oracle_retrieval'))
                (dest/'evaluation-index.json').write_text(json.dumps(index,indent=2))
            return output
        command=[str(root/'.venv/bin/python'),'-u',str(frozen/'code/finqa_workflow_v2.py'),
            '--model',str(a.model),'--inputs',str(root/'benchmarks/views/finqa-roles-v3'/(partition+'_inputs.jsonl')),
            '--labels',str(root/'benchmarks/raw/finqa'/(partition+'.json')),'--partition',partition,'--mode',mode,'--output',str(output)]
        if smoke:command+=['--smoke',str(smoke)]
        if mode=='multitask':command+=['--multitask',str(train/'multitask'/f'seed{seed}'/'delta')]
        elif mode not in ['single','prompt','base_refine']:
            for name in ['retriever','planner','verifier']:command+=['--'+name,str(train/name/f'seed{seed}'/'delta')]
        print(json.dumps(dict(event='start',output=str(output))),flush=True)
        log=dest/(f'{mode}_{partition}_seed{seed}'+('_smoke' if smoke else '')+'.log')
        with open(log,'w') as stream:subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        completed=json.loads((output/'complete.json').read_text())
        if completed['n']!=(smoke if smoke else 883 if partition=='dev' else 1147):raise ValueError('Incomplete application partition')
        print(json.dumps(dict(event='complete',output=str(output))),flush=True)
        if not smoke:
            index.append(dict(mode=mode,partition=partition,seed=seed,path=str(output),diagnostic_oracle=mode=='oracle_retrieval'))
            (dest/'evaluation-index.json').write_text(json.dumps(index,indent=2))
        return output

    for mode in ['single','prompt','base_refine']:run(mode,'dev',smoke=2)
    for mode in ['roles','multitask','replicas','no_verifier','no_retriever','oracle_retrieval','random_roles']:run(mode,'dev',smoke=8)
    shared=dest/'smoke/seed11/roles/dev/predictions.jsonl'
    replicas=dest/'smoke/seed11/replicas/dev/predictions.jsonl'
    def semantic_rows(path):
        rows=[json.loads(line) for line in path.read_text().split('\n') if line]
        return {row['id']:dict(program=row['program'],calls=[(call['prompt_hash'],call['text']) for call in row['calls']]) for row in rows}
    if semantic_rows(shared)!=semantic_rows(replicas):raise ValueError('Shared and replicated experts differ; investigate before main application claims')
    (dest/'representation-smoke.json').write_text(json.dumps(dict(status='exact_prompt_and_output_match',n=8),indent=2))
    for mode in spec['baselines']:
        for partition in spec['partitions']:run(mode,partition)
    for seed in spec['main_seeds']:
        for mode in spec['main_modes']:
            for partition in spec['partitions']:run(mode,partition,seed)
    for mode in spec['ablations']:
        for partition in spec['partitions']:run(mode,partition,11)
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='finqa_matrix_complete',spec=spec),indent=2))


if __name__=='__main__':main()
