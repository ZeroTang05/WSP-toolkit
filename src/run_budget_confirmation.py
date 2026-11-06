"""Full heldout confirmation of resident-budget selections on exact V2 runtime.

Wait for both matrices to finish. Reuse verified completed parameter states,
not policy names, filesystem names or approximate tensor similarity. Reuse is
explicit in the index; no new inference/search timing is fabricated.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from confirmation_state import recorded_key,checked_plan_key,sha
from analyze_model_trends import checked_scores,analyze
from audit_search import audit

SOURCES={name:(Path(__file__).parent/name).read_bytes() for name in [
    'run_budget_confirmation.py','confirmation_state.py','task_plan.py','audit_search.py','delta_bank.py',
    'analyze_model_trends.py','analyze_public.py','prepare_training_views.py','finqa_views.py']}
BENCHFILES=['benchmark_runner.py','delta_bank.py','mask_bundle.py','task_plan.py','structure_bundle.py','weightlab.py',
    'audit_search.py','prepare_training_views.py','finqa_views.py']
GROUPS={'gsm8k':('idea4_gsm8k_cot_zeroshot_v1',32,['gsm8k']),
    'guards':('idea4_arc_challenge_v1,idea4_hellaswag_v1',8,['arc','hellaswag'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--models-root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    dest=root/'benchmarks/budget-confirm-v1';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signature=hashlib.sha256(b''.join(name.encode()+data for name,data in sorted(SOURCES.items()))).hexdigest()
    code=dest/('code_'+signature);code.mkdir(exist_ok=True)
    for name,data in SOURCES.items():(code/name).write_bytes(data)
    def wait(path):
        deadline=time.monotonic()+259200
        if not path.exists():print(json.dumps(dict(event='waiting',dependency=str(path))),flush=True)
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(30)
    parent=root/'benchmarks/confirm-v2';searches=root/'benchmarks/search-budget-v1'
    wait(parent/'protocol.json');wait(searches/'protocol.json')
    parent_spec=json.loads((parent/'protocol.json').read_text());search_spec=json.loads((searches/'protocol.json').read_text())
    runtime=root/'research/frozen'/('confirmation-'+parent_spec['source_bundle'])/'code'
    def verify_runtime():
        if hashlib.sha256(b''.join((runtime/name).read_bytes() for name in BENCHFILES)).hexdigest()!=parent_spec['source_bundle']:
            raise ValueError('Changed parent evaluation runtime')
    verify_runtime()
    spec=dict(version='budget-confirm-v1',source_bundle=signature,parent=parent_spec,search=search_spec,
        gsm_batch=32,mc_batch=8,policy='full official partitions; reuse only same native-base/functional-delta state on exact parent backend',
        scope='Quality confirmation of budget-selected states; reused inference is not a new replicate or fresh timing measurement')
    config=dest/'protocol.json'
    if config.exists() and json.loads(config.read_text())!=spec:raise ValueError('Budget confirmation protocol changed')
    config.write_text(json.dumps(spec,indent=2))
    wait(parent/'matrix_complete.json');wait(searches/'matrix_complete.json');verify_runtime()
    prior=json.loads((parent/'evaluation-index.json').read_text())
    selected=json.loads((searches/'evaluation-index.json').read_text())
    expected={(m,s,p) for m in search_spec['models'] for s in search_spec['seeds'] for p in search_spec['policies']}
    keys=[(r['model'],r['seed'],r['policy']) for r in selected]
    if set(keys)!=expected or len(keys)!=len(expected):raise ValueError('Incomplete/duplicate cost selection index')
    expected_parent={(m,None,'base') for m in search_spec['models']}|{
        (m,s,p) for m in search_spec['models'] for s in search_spec['seeds'] for p in ['standard_lora']+search_spec['policies']}
    parent_keys=[(r['model'],r['seed'],r['method']) for r in prior]
    if set(parent_keys)!=expected_parent or len(parent_keys)!=len(expected_parent):raise ValueError('Incomplete/duplicate parent confirmation index')
    cache={};checked_paths={};reference_settings={};reference_items={}
    def validate(run,model,group):
        path=Path(run);marker=(str(path),model,group)
        if marker not in checked_paths:
            values={task:checked_scores(path,task,model) for task in GROUPS[group][2]}
            # Timing/helper/data protocol checked by checked_scores; require
            # exact frozen runner identity rather than merely equal metrics.
            complete=json.loads((path/'complete.json').read_text())
            if complete['code_sha256']!=sha(runtime/'benchmark_runner.py'):raise ValueError('Different benchmark backend')
            for task,value in values.items():
                key=(model,task)
                if key in reference_settings and (value[2]!=reference_settings[key] or value[1]!=reference_items[key]):
                    raise ValueError('Changed evaluation protocol or document contents')
                reference_settings[key]=value[2];reference_items[key]=value[1]
            checked_paths[marker]=values
        return checked_paths[marker]
    for row in prior:
        for group in GROUPS:
            run=Path(row[group]);state=recorded_key(run);values=validate(run,row['model'],group)
            key=(row['model'],state,group)
            if key in cache and cache[key]!=str(run):
                other=validate(cache[key],row['model'],group)
                for task in values:
                    if values[task][1:]!=other[task][1:] or not (values[task][0]==other[task][0]).all():
                        raise ValueError('Duplicate executable states disagree on public predictions/protocol')
            else:cache[key]=str(run)
    results=[dict(**{k:v for k,v in r.items() if k!='state_id'},state_id=recorded_key(r['gsm8k']),
        source_study='confirm-v2',reused_existing_state={group:True for group in GROUPS})
        for r in prior if r['method'] in ['base','standard_lora']]
    verified_models=set();env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for row in selected:
        model=row['model'];seed=row['seed'];policy=row['policy'];search=Path(row['path'])
        audit_result=audit(search)
        protocol=json.loads((search/'protocol.json').read_text());plan=json.loads((search/'selected_plan.json').read_text())
        pilot=search.parent/'fixed'/'complete.json'
        if sha(pilot)!=row['calibration_sha256']:raise ValueError('Budget calibration changed')
        pilot_complete=json.loads(pilot.read_text());pilot_protocol=json.loads((pilot.parent/'protocol.json').read_text())
        if pilot_complete['active_seconds']!=row['budget_seconds']:raise ValueError('Budget differs from calibration time')
        if protocol['policy']!=policy or protocol['seed']!=seed or protocol['smoke']:
            raise ValueError('Selection index does not identify this formal search')
        for field in ['model','seed','delta_manifest_sha256','shared_training_protocol_sha256',
            'domain_sha256','guard_sha256','domain_ids','guard_ids','guard_mode','guard_margin',
            'domain_n','guard_n','without_guard','without_internal_observations']:
            if protocol[field]!=pilot_protocol[field]:raise ValueError('Unequal search conditions: '+field)
        if json.loads((search/'complete.json').read_text())['query_cap_censored']:raise ValueError('Query cap censored the cost comparison')
        if policy!='fixed' and protocol['budget_seconds']!=row['budget_seconds']:raise ValueError('Budget index mismatch')
        base=json.loads((search/'base_manifest.json').read_text())
        model_path=a.models_root/model
        if str(model_path.resolve())!=protocol['model']:raise ValueError('Different model path from search')
        if model not in verified_models:
            for name,digest in base['files'].items():
                if sha(model_path/name)!=digest:raise ValueError('Source model changed')
            verified_models.add(model)
        delta=Path(protocol['delta']);state=checked_plan_key(base['files'],delta,plan,model_path)
        outputs={};reused={}
        extra=[] if not any(plan['coordinates']) else ['--delta',str(delta)]
        if extra and plan['coordinates']!=[1.]*4:extra+=['--plan',str(search/'selected_plan.json')]
        for group,(tasks,batch,_) in GROUPS.items():
            key=(model,state,group)
            if key in cache:
                outputs[group]=cache[key];reused[group]=True;continue
            output=dest/model/state[:20]/group
            fresh=False
            if (output/'complete.json').exists():
                if recorded_key(output)!=state:raise ValueError('Existing result belongs to another state')
            else:
                if output.exists():raise FileExistsError('Inspect partial confirmation before resuming '+str(output))
                for filename,digest in reference_settings[(model,GROUPS[group][2][0])]['task_files'].items():
                    if sha(root/'benchmarks/tasks'/filename)!=digest:raise ValueError('Benchmark task definition changed')
                command=[str(root/'.venv-bench/bin/python'),'-u',str(runtime/'benchmark_runner.py'),
                    '--model',str(model_path),'--tasks-dir',str(root/'benchmarks/tasks'),'--tasks',tasks,
                    '--batch-size',str(batch),'--output',str(output)]+extra
                print(json.dumps(dict(event='start',path=str(output))),flush=True)
                with open(dest/(model+'_'+state[:20]+'_'+group+'.log'),'w') as stream:
                    subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                fresh=True
            validate(output,model,group)
            if recorded_key(output)!=state:raise ValueError('Confirmed a different parameter state')
            cache[key]=str(output);outputs[group]=str(output);reused[group]=not fresh
        results.append(dict(model=model,seed=seed,method=policy,state_id=state,**outputs,
            reused_existing_state=reused,search_path=str(search),selection_audit=audit_result,
            budget_seconds=row['budget_seconds'],calibration_sha256=row['calibration_sha256']))
        (dest/'evaluation-index.json').write_text(json.dumps(results,indent=2))
    trends=analyze(dest/'evaluation-index.json')
    (dest/'model-trends.json').write_text(json.dumps(trends,indent=2))
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='budget_heldout_confirmation_complete',entries=len(results),spec=spec),indent=2))


if __name__=='__main__':main()
