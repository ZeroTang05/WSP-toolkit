"""Full public confirmation with content-addressed, semantically equal states.

All main GSM8K confirmations use batch32, MC tasks batch8. Existing batch8
generation results remain an earlier protocol and are never mixed into these
paired comparisons. Zero/all-one plans reuse their exact executable state.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import yaml


FILES=['benchmark_runner.py','delta_bank.py','mask_bundle.py','task_plan.py','structure_bundle.py','weightlab.py',
    'audit_search.py','prepare_training_views.py','finqa_views.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--models-root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();dest=root/'benchmarks/confirm-v2';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source={name:(root/'src'/name).read_bytes() for name in FILES}
    signature=hashlib.sha256(b''.join(source.values())).hexdigest()
    frozen=root/'research/frozen'/('confirmation-'+signature);(frozen/'code').mkdir(parents=True,exist_ok=True)
    for name,data in source.items():(frozen/'code'/name).write_bytes(data)
    tasks=root/'benchmarks/tasks'
    # Only this explicitly training-partition alias is used for batch memory checks.
    check=yaml.safe_load((tasks/'gsm8k_cot_zeroshot.yaml').read_text())
    check.update(task='idea4_gsm8k_train_batchcheck_v2',test_split='train',tag=['idea4_batchcheck'])
    (tasks/'idea4_gsm8k_train_batchcheck_v2.yaml').write_text(yaml.safe_dump(check,sort_keys=False))
    spec=dict(version='confirmation-v2',source_bundle=signature,gsm8k_batch_size=32,mc_batch_size=8,
        max_new_tokens=512,quality_source='full public partitions only',
        caching='identical base/delta semantics, coordinates, runtime and task protocol; no fabricated timing for reused results')
    path=dest/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Confirmation protocol changed')
    path.write_text(json.dumps(spec,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    index=[];verified=set()

    def wait_for(path):
        deadline=time.monotonic()+172800
        if not path.exists():print(json.dumps(dict(event='waiting',dependency=str(path))),flush=True)
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(30)

    def evaluate(model,state,extra,group,tasks_name,batch,smoke=None):
        output=dest/model/state/group
        if (output/'complete.json').exists():
            complete=json.loads((output/'complete.json').read_text())
            if complete['status']!=('smoke_complete' if smoke else 'full_complete'):raise ValueError('Wrong completion scope')
            return str(output)
        command=[str(root/'.venv-bench/bin/python'),'-u',str(frozen/'code/benchmark_runner.py'),
            '--model',str(a.models_root/model),'--tasks-dir',str(tasks),'--tasks',tasks_name,
            '--batch-size',str(batch),'--output',str(output)]+extra
        if smoke:command+=['--smoke',str(smoke)]
        print(json.dumps(dict(event='start',output=str(output))),flush=True)
        with open(dest/(model+'_'+state+'_'+group+'.log'),'w') as stream:
            subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='complete',output=str(output))),flush=True)
        return str(output)

    def confirm(model,base,delta=None,plan=None):
        if model not in verified:
            for name,value in base['files'].items():
                if sha(a.models_root/model/name)!=value:raise ValueError('Current base changed: '+model+'/'+name)
            evaluate(model,'batchcheck',[],'train32','idea4_gsm8k_train_batchcheck_v2',32,32)
            verified.add(model)
        coordinates=[1.]*4;extra=[];semantics=dict(base=base['files'])
        if plan is not None:coordinates=plan['coordinates']
        if delta is not None and any(value!=0 for value in coordinates):
            meta=json.loads((delta/'manifest.json').read_text())
            semantics.update(delta_manifest=sha(delta/'manifest.json'),coordinates=coordinates)
            extra=['--delta',str(delta)]
            if coordinates!=[1.]*4:
                state_id=hashlib.sha256(json.dumps(semantics,sort_keys=True).encode()).hexdigest()[:20]
                plan_file=dest/(model+'_'+state_id+'_plan.json')
                if not plan_file.exists():plan_file.write_text(json.dumps(plan,indent=2))
                extra+=['--plan',str(plan_file)]
        state_id=hashlib.sha256(json.dumps(semantics,sort_keys=True).encode()).hexdigest()[:20]
        gsm=evaluate(model,state_id,extra,'gsm8k','idea4_gsm8k_cot_zeroshot_v1',32)
        guard=evaluate(model,state_id,extra,'guards','idea4_arc_challenge_v1,idea4_hellaswag_v1',8)
        return dict(state_id=state_id,gsm8k=gsm,guards=guard,identity_selected=delta is not None and all(value==0 for value in coordinates))

    def record(model,seed,method,evaluation):
        index.append(dict(model=model,seed=seed,method=method,**evaluation))
        (dest/'evaluation-index.json').write_text(json.dumps(index,indent=2))

    models=['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct']
    for model in models:
        training=root/'benchmarks/optimization/gsm8k'/model/'lora'/'seed11'
        wait_for(training/'complete.json');base=json.loads((training/'base_manifest.json').read_text())
        record(model,None,'base',confirm(model,base))
        for seed in [11,22,33]:
            training=root/'benchmarks/optimization/gsm8k'/model/'lora'/f'seed{seed}'
            wait_for(training/'complete.json');delta=training/'delta'
            record(model,seed,'standard_lora',confirm(model,base,delta))
            for policy in ['fixed','random','greedy','bo','llm']:
                search=root/'benchmarks/search-v2'/model/f'seed{seed}'/policy
                wait_for(search/'complete.json')
                subprocess.run([str(root/'.venv/bin/python'),str(frozen/'code/audit_search.py'),'--run',str(search)],env=env,check=True)
                plan=json.loads((search/'selected_plan.json').read_text())
                record(model,seed,policy,confirm(model,base,delta,plan))
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='full_public_confirmation_complete',entries=len(index)),indent=2))


if __name__=='__main__':main()
