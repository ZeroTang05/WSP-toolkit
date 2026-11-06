"""Independent resident-budget study; source snapshots never alter V2 runs."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

FILES=['optimize_public_budget.py','search_v2.py','search_metrics.py','task_plan.py','delta_bank.py',
    'weightlab.py','prepare_training_views.py','finqa_views.py','contract_registry.py','audit_search.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--models-root',required=True,type=Path);a=p.parse_args();root=a.root.resolve()
    dest=root/'benchmarks/search-budget-v1';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files={Path('code')/name:(root/'src'/name).read_bytes() for name in FILES}
    files[Path('contracts/methods.json')]=(root/'contracts/methods.json').read_bytes()
    signature=hashlib.sha256(b''.join(str(k).encode()+v for k,v in sorted(files.items()))).hexdigest()
    frozen=root/'research/frozen'/('resident-budget-'+signature)
    for path,value in files.items():
        output=frozen/path;output.parent.mkdir(parents=True,exist_ok=True)
        if output.exists() and output.read_bytes()!=value:raise ValueError('Changed frozen source')
        output.write_bytes(value)
    spec=dict(version='resident-budget-v1',source=signature,
        models=['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct'],seeds=[11,22,33],
        policies=['fixed','random','greedy','bo','llm'],domain_n=64,guard_n=32,query_cap=256,
        budget='active time of a fresh full21 global-grid calibration for the same model and seed',
        cost='main-entry minus GPU lock wait: target/controller setup, source fingerprints, decisions and evaluations; shared prior LoRA training reported separately',
        deadline='soft deadline; late operations logged and excluded from selected candidates; actual overrun separately reported',
        scope='New study, not replacement of equal21query V2. Calibration fixed grid reused as reference, not an independent second fixed run. Complete heldout confirmation still required.')
    config=dest/'protocol.json'
    if config.exists() and json.loads(config.read_text())!=spec:raise ValueError('Resident study protocol changed')
    config.write_text(json.dumps(spec,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    python=str(root/'.venv/bin/python')

    def wait(path):
        deadline=time.monotonic()+259200
        if not path.exists():print(json.dumps(dict(event='waiting',file=str(path))),flush=True)
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(30)

    def run(model,seed,policy,smoke=False,budget=None):
        parent=dest/('smoke' if smoke else 'full')/model/f'seed{seed}'
        output=parent/policy
        if not (output/'complete.json').exists():
            if output.exists():raise FileExistsError('Preserve partial cost search '+str(output))
            delta=root/'benchmarks/optimization/gsm8k'/model/'lora'/f'seed{seed}'/'delta'
            wait(delta.parent/'complete.json')
            command=[python,'-u',str(frozen/'code/optimize_public_budget.py'),'--model',str(a.models_root/model),
                '--delta',str(delta),'--view',str(root/'benchmarks/views/gsm8k-v1'),
                '--guard',str(root/'benchmarks/raw/idea4_arc_challenge_v1/validation.jsonl'),
                '--output',str(output),'--policy',policy,'--seed',str(seed),'--queries','256',
                '--domain-n','4' if smoke else '64','--guard-n','4' if smoke else '32']
            if budget is not None:command+=['--budget-seconds',str(budget)]
            if smoke:command+=['--smoke']
            if policy=='llm':command+=['--controller',str(a.models_root/'Qwen3-4B-Instruct-2507')]
            print(json.dumps(dict(event='start',path=str(output),budget_seconds=budget)),flush=True)
            log=dest/(model+f'_{seed}_{policy}'+('_smoke' if smoke else '')+'.log')
            with open(log,'w') as stream:subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        subprocess.run([python,str(frozen/'code/audit_search.py'),'--run',str(output)],env=env,check=True)
        complete=json.loads((output/'complete.json').read_text())
        if complete['budget_seconds']!=budget:raise ValueError('Different budget from calibration')
        if complete['query_cap_censored']:raise ValueError('Query cap reached; not a full time-budget comparison')
        print(json.dumps(dict(event='complete',path=str(output),queries=complete['queries'],eligible=complete['eligible_queries'])),flush=True)
        return output,complete

    # Small public-derived smoke establishes execution/clock/audit correctness.
    # It does not choose the formal budget or hyperparameters.
    _,calibration=run('Qwen3-0.6B',11,'fixed',smoke=True)
    smoke_budget=max(30.,calibration['active_seconds'])
    for policy in ['random','bo','llm']:run('Qwen3-0.6B',11,policy,smoke=True,budget=smoke_budget)
    records=[]
    for model in spec['models']:
        for seed in spec['seeds']:
            pilot,value=run(model,seed,'fixed')
            budget=value['active_seconds'];pilot_sha=hashlib.sha256((pilot/'complete.json').read_bytes()).hexdigest()
            records.append(dict(model=model,seed=seed,policy='fixed',path=str(pilot),budget_seconds=budget,calibration_sha256=pilot_sha))
            for policy in ['random','greedy','bo','llm']:
                output,_=run(model,seed,policy,budget=budget)
                records.append(dict(model=model,seed=seed,policy=policy,path=str(output),budget_seconds=budget,calibration_sha256=pilot_sha))
                (dest/'evaluation-index.json').write_text(json.dumps(records,indent=2))
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='resident_budget_matrix_complete',spec=spec),indent=2))


if __name__=='__main__':main()
