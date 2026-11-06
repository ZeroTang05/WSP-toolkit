"""Exploratory representation repair, retaining frozen V3 failure controls.

Retrain only planner/verifier/multitask on header-preserving training views.
Use the exact seed11 V3 training implementation/recipe, and reuse byte-identical
retriever supervision. No extra training steps, changed targets or negatives.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

RUNTIME=['finqa_workflow_v2.py','finqa_roles.py','finqa_executor.py','finqa_views.py',
    'delta_bank.py','weightlab.py','prepare_training_views.py','peft_bank.py']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path);a=p.parse_args();root=a.root.resolve()
    dest=root/'benchmarks/finqa-header-control-v1';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=root/'benchmarks/optimization/finqa/Qwen3-1.7B'
    view=root/'benchmarks/views/finqa-roles-v4-header'
    prior=root/'benchmarks/views/finqa-roles-v3'
    manifest=json.loads((view/'manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if digest(view/name)!=expected:raise ValueError('Changed repaired view: '+name)
    for name in ['retriever_train.jsonl','dev_inputs.jsonl','test_inputs.jsonl']:
        if digest(prior/name)!=digest(view/name):raise ValueError('Unexpected change: '+name)
    sources={Path('code')/name:(root/'src'/name).read_bytes() for name in RUNTIME}
    for path in (root/'vendor/finqa').iterdir():
        if path.is_file():sources[Path('vendor/finqa')/path.name]=path.read_bytes()
    recipes={}
    for role in ['planner','verifier','multitask']:
        reference=old/role/'seed11';recipe=json.loads((reference/'protocol.json').read_text())
        if Path(recipe['model']).resolve()!=a.model.resolve():raise ValueError('Different model from V3 control')
        if digest(reference/'code/specialize.py')!=recipe['code_sha256']:raise ValueError('Changed training entry')
        for name,expected in recipe['implementation_files'].items():
            if digest(reference/'code'/name)!=expected:raise ValueError('Changed training helper')
        if digest(reference/'contracts/methods.json')!=recipe['contract_spec_sha256']:raise ValueError('Changed contract')
        recipes[role]=recipe
        for folder in ['code','contracts']:
            for path in (reference/folder).iterdir():
                if path.is_file():sources[Path('train')/role/folder/path.name]=path.read_bytes()
        # Imported by paper_methods, although its artifact readers are unused
        # by LoRA SFT. Freeze the dependency rather than use editable imports.
        sources[Path('train')/role/'code/verify_artifacts.py']=(root/'src/verify_artifacts.py').read_bytes()
    identity=hashlib.sha256(b''.join(str(k).encode()+v for k,v in sorted(sources.items()))).hexdigest()
    frozen=root/'research/frozen'/('finqa-header-'+identity)
    for path,data in sources.items():
        target=frozen/path;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=data:raise ValueError('Frozen source changed')
        target.write_bytes(data)
    spec=dict(version='header-control-v1',source_bundle=identity,seeds=[11,22,33],
        view_manifest_sha256=digest(view/'manifest.json'),training_reference=recipes,
        conditions=['prompt','original_roles_with_header','original_multitask_with_header',
            'retrained_roles_with_header','retrained_multitask_with_header'],partitions=['dev','test'],
        retriever='Reuse original per-seed factors; training file byte-identical',
        scope='Post-hoc exploratory representation control after V3 development smoke. Not a new untouched test confirmation; negatives and training budgets unchanged.')
    protocol=dest/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=spec:raise ValueError('Study protocol changed')
    protocol.write_text(json.dumps(spec,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    python=str(root/'.venv/bin/python');trained=dest/'training'

    def execute(command,log):
        print(json.dumps(dict(event='start',log=str(log))),flush=True)
        with open(log,'w') as stream:subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='complete',log=str(log))),flush=True)

    def wait(path):
        deadline=time.monotonic()+172800
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(30)

    def evaluate(condition,seed=11,smoke=None,partition='dev'):
        mode='prompt' if condition=='prompt' else 'multitask' if 'multitask' in condition else 'roles'
        output=dest/('smoke' if smoke else 'full')/f'seed{seed}'/condition/partition
        if (output/'complete.json').exists():return
        command=[python,'-u',str(frozen/'code/finqa_workflow_v2.py'),'--model',str(a.model),
            '--inputs',str(view/(partition+'_inputs.jsonl')),'--labels',str(root/'benchmarks/raw/finqa'/(partition+'.json')),
            '--partition',partition,'--mode',mode,'--output',str(output)]
        for role in ([] if mode=='prompt' else ['multitask'] if mode=='multitask' else ['retriever','planner','verifier']):
            parent=old if condition.startswith('original_') or role=='retriever' else trained
            run=parent/role/f'seed{seed}';wait(run/'complete.json')
            if role=='retriever':
                prior_protocol=json.loads((run/'protocol.json').read_text())
                if prior_protocol['training_sha256']!=digest(view/'retriever_train.jsonl'):raise ValueError('Retriever reuse mismatch')
            command+=['--'+role,str(run/'delta')]
        if smoke:command+=['--smoke',str(smoke)]
        execute(command,dest/(f'{condition}_{seed}_{partition}'+('_smoke' if smoke else '')+'.log'))
        result=json.loads((output/'complete.json').read_text())
        if result['n']!=(smoke or {'dev':883,'test':1147}[partition]):raise ValueError('Incomplete partition')

    # Input-only controls precede retraining; all conditions use the same rule.
    for condition in spec['conditions'][:3]:evaluate(condition,smoke=8)
    for seed in spec['seeds']:
        for role,recipe in recipes.items():
            output=trained/role/f'seed{seed}'
            if (output/'complete.json').exists():continue
            command=[python,'-u',str(frozen/'train'/role/'code/specialize.py'),'--view',str(view),
                '--output',str(output),'--seed',str(seed)]
            for name in ['model','train_file','steps','batch','rank','alpha','lr','max_length']:
                command+=['--'+name.replace('_','-'),str(recipe[name])]
            execute(command,dest/f'train_{role}_{seed}.log')
        for condition in spec['conditions'][3:]:evaluate(condition,seed,smoke=8)
        conditions=spec['conditions'] if seed==11 else spec['conditions'][1:]
        for condition in conditions:
            for partition in spec['partitions']:evaluate(condition,seed,partition=partition)
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='header_control_complete',spec=spec),indent=2))


if __name__=='__main__':main()
