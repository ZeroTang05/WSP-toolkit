"""Confirm relative-capacity LoRA controls after homogeneous C1 evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--models-root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    dest=root/'benchmarks/capacity-confirm-v2';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def wait(path):
        deadline=time.monotonic()+259200
        if not path.exists():print(json.dumps(dict(event='waiting',dependency=str(path))),flush=True)
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(30)
    parent=root/'benchmarks/confirm-v2';wait(parent/'matrix_complete.json')
    config=json.loads((parent/'protocol.json').read_text())
    entry=root/'research/frozen'/('confirmation-'+config['source_bundle'])/'code/benchmark_runner.py'
    if not entry.exists():raise FileNotFoundError('Exact parent runtime snapshot missing')
    controls=json.loads((root/'research/capacity-controls.json').read_text())
    references=json.loads((parent/'evaluation-index.json').read_text())
    records=[];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    for control in controls:
        model=control['model'];rank=control['relative_control']['rank']
        for seed in [11,22,33]:
            if rank==8:
                original=next(row for row in references if row['model']==model and row['seed']==seed and row['method']=='standard_lora')
                records.append(dict(model=model,seed=seed,reused_C1_identical_condition=True,**{key:original[key] for key in ['gsm8k','guards']}))
                continue
            train=root/'benchmarks/optimization/gsm8k'/model/'lora-ratio'/f'seed{seed}'
            wait(train/'complete.json');recipe=json.loads((train/'protocol.json').read_text())
            if recipe['rank']!=rank or recipe['alpha']!=2*rank:raise ValueError('Capacity training recipe mismatch')
            outputs={}
            for group,tasks,batch in [('gsm8k','idea4_gsm8k_cot_zeroshot_v1',32),('guards','idea4_arc_challenge_v1,idea4_hellaswag_v1',8)]:
                output=dest/model/f'seed{seed}'/group;outputs[group]=str(output)
                if (output/'complete.json').exists():continue
                command=[str(root/'.venv-bench/bin/python'),'-u',str(entry),'--model',str(a.models_root/model),
                    '--tasks-dir',str(root/'benchmarks/tasks'),'--tasks',tasks,'--batch-size',str(batch),
                    '--delta',str(train/'delta'),'--output',str(output)]
                print(json.dumps(dict(event='start',output=str(output))),flush=True)
                with open(dest/(model+f'_seed{seed}_{group}.log'),'w') as stream:
                    subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                result=json.loads((output/'complete.json').read_text())
                if result['status']!='full_complete':raise ValueError('Incomplete capacity benchmark')
            records.append(dict(model=model,seed=seed,rank=rank,actual_fraction=control['relative_control']['actual_fraction'],**outputs))
            (dest/'evaluation-index.json').write_text(json.dumps(records,indent=2))
    (dest/'evaluation-index.json').write_text(json.dumps(records,indent=2))
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='capacity_confirmation_complete',runtime=config,
        scope='Standard LoRA C2 capacity control; adaptive policies remain C1 in this study'),indent=2))


if __name__=='__main__':main()
