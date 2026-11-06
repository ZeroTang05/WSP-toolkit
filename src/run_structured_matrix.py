"""Paper-derived structured compression, three calibration draws, real profiles."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess


FILES=['prune_structured.py','paper_methods.py','weightlab.py','contract_registry.py','prepare_training_views.py',
    'finqa_views.py','verify_artifacts.py','delta_bank.py','structure_bundle.py','profile_variant.py',
    'benchmark_runner.py','mask_bundle.py','task_plan.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--models-root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    dest=root/'benchmarks/structured-v2';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data={name:(root/'src'/name).read_bytes() for name in FILES};contract=(root/'contracts/methods.json').read_bytes()
    signature=hashlib.sha256(b''.join(data.values())+contract).hexdigest()
    frozen=root/'research/frozen'/('structured-v2-'+signature)
    (frozen/'code').mkdir(parents=True,exist_ok=True);(frozen/'contracts').mkdir(exist_ok=True)
    for name,value in data.items():(frozen/'code'/name).write_bytes(value)
    (frozen/'contracts/methods.json').write_bytes(contract)
    spec=dict(source_bundle=signature,models=['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct'],
        calibration_seeds=[11,22,33],calibration_n=128,statistics_dtype='float32',requested_channel_fractions=[1/12,1/6],
        alignment=128,importance='sum elementwise abs(weight*mean calibration gradient) in coupled gated-MLP channels',
        method_grade='declared MLP-only first-order adaptation, not full LLM-Pruner reproduction',
        profiling=dict(process_repeats=3,within_process_repeats=15,warmup=5,random_order_seed=20261106),
        quality='full GSM8K batch32; full ARC/HellaSwag batch8; no dev/test selection of pruning indices or fraction')
    path=dest/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Structured study protocol changed')
    path.write_text(json.dumps(spec,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')

    def run(script,venv,output,args):
        if (output/'complete.json').exists():return
        output.parent.mkdir(parents=True,exist_ok=True)
        print(json.dumps(dict(event='start',output=str(output))),flush=True)
        log=output.parent/(output.name+'.log')
        with open(log,'w') as stream:
            subprocess.run([str(root/venv/'bin/python'),'-u',str(frozen/'code'/script),
                '--output',str(output)]+args,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        if not (output/'complete.json').exists():raise ValueError('Child returned without completion evidence')
        print(json.dumps(dict(event='complete',output=str(output))),flush=True)

    for model in spec['models']:
        model_path=str(a.models_root/model);prunes=[]
        run('prune_structured.py','.venv',dest/model/'preflight',
            ['--model',model_path,'--view',str(root/'benchmarks/views/gsm8k-v1'),'--fraction',str(1/12),
             '--calibration-n','2','--statistics-dtype','float32','--calibration-seed','11'])
        for seed in spec['calibration_seeds']:
            for label,fraction in [('fraction08',1/12),('fraction16',1/6)]:
                output=dest/model/f'seed{seed}'/label/'artifact'
                run('prune_structured.py','.venv',output,['--model',model_path,'--view',str(root/'benchmarks/views/gsm8k-v1'),
                    '--fraction',str(fraction),'--calibration-n','128','--calibration-seed',str(seed),'--statistics-dtype','float32'])
                prunes.append((f'seed{seed}/{label}',output/'selector.json'))
        profiles=[('base',None,repeat) for repeat in range(3)]+[(name,selector,repeat) for name,selector in prunes for repeat in range(3)]
        random.Random(20261106).shuffle(profiles)
        for name,selector,repeat in profiles:
            args=['--model',model_path]
            if selector:args+=['--structure',str(selector)]
            run('profile_variant.py','.venv',dest/model/'profiles'/name/f'process{repeat}',args)
        for name,selector in prunes:
            for group,task,batch in [('gsm8k','idea4_gsm8k_cot_zeroshot_v1',32),('guards','idea4_arc_challenge_v1,idea4_hellaswag_v1',8)]:
                run('benchmark_runner.py','.venv-bench',dest/model/name/group,
                    ['--model',model_path,'--tasks-dir',str(root/'benchmarks/tasks'),'--tasks',task,
                     '--batch-size',str(batch),'--structure',str(selector)])
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='structured_matrix_complete',protocol=spec),indent=2))


if __name__=='__main__':main()
