"""Content expansion: compile real-model programs, causal maps, full effects."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

FILES=['run_programming_expansion.py','programming_probe.py','programming_effects.py','delta_programming.py',
    'depth_programming.py','delta_bank.py','analyze_delta.py','analyze_public.py','weightlab.py','prepare_training_views.py',
    'finqa_views.py','search_metrics.py']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--models-root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();dest=root/'benchmarks/programming-expansion-v1';dest.mkdir(exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={n:(root/'src'/n).read_bytes() for n in FILES}
    signature=hashlib.sha256(b''.join(n.encode()+b for n,b in sorted(sources.items()))).hexdigest()
    frozen=root/'research/frozen'/('programming-expansion-'+signature);(frozen/'code').mkdir(parents=True,exist_ok=True)
    for name,data in sources.items():(frozen/'code'/name).write_bytes(data)
    models=['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct']
    spec=dict(version='programming-expansion-v1',source=signature,models=models,source_seeds=[11,22,33],
        programs=['soup','frobenius-r8','frobenius-r4','activation-r8','norm-matched-seed11','bi10','bi20','random10','tail10'],
        controls=['base','standard-seed11'],calibration=128,causal_response_development=64,guard_arc_validation=32,
        effects='full GSM8K1319,ARC1172,HellaSwag10042; fixed ranks and policies; no test-selected program',
        scope='New parameter-programming content, not smoke. Soup consumes three existing training runs, not matched training cost; compressed adapters are not compressed dense bases.')
    protocol=dest/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())!=spec:raise ValueError('Expansion protocol changed')
    protocol.write_text(json.dumps(spec,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def run(script,venv,args,log):
        print(json.dumps(dict(event='start',script=script,log=str(log))),flush=True)
        with open(log,'w') as stream:subprocess.run([str(root/venv/'bin/python'),'-u',str(frozen/'code'/script)]+args,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='complete',script=script,log=str(log))),flush=True)
    summaries=[]
    for model in models:
        paths=[root/'benchmarks/optimization/gsm8k'/model/'lora'/f'seed{seed}'/'delta' for seed in [11,22,33]]
        if not all((path.parent/'complete.json').exists() for path in paths):raise ValueError('Source training incomplete')
        programs=dest/model/'programs'
        if not (programs/'complete.json').exists():run('delta_programming.py','.venv',['--sources']+list(map(str,paths))+['--output',str(programs)],dest/(model+'_compile.log'))
        geo=json.loads((programs/'geometry.json').read_text());descriptions=geo['source_descriptions']
        ratios=[];profiles=[]
        layers=json.loads((a.models_root/model/'config.json').read_text())['num_hidden_layers']
        for description in descriptions:
            energy=sum(v['frobenius']**2 for v in description['modules'])
            ratios.append(sum(sum(s*s for s in v['singular_values'][:4]) for v in description['modules'])/energy)
            bands=[0.]*4
            for v in description['modules']:bands[min(3,4*v['layer']//layers)]+=v['frobenius']**2/energy
            profiles.append(bands)
        summaries.append(dict(model=model,mean_seed_cosine=sum(v['cosine'] for v in geo['pairwise'])/3,
            seed_cosines=[v['cosine'] for v in geo['pairwise']],mean_top4_update_energy=sum(ratios)/3,
            source_seed_depth_energy=profiles,soup_norm_to_seed11=geo['norm_matched_scale'],programs=geo['programs']))
        (dest/'cross-model-geometry.json').write_text(json.dumps(dict(status='real_trained_update_geometry',models=summaries,
            scope='All four models use their own parameter coordinates. Cross-model comparison uses normalized spectral/depth features; no raw cross-model cosine or capability inference.'),indent=2))
    for model in models:
        out=dest/model/'probe'
        if not (out/'complete.json').exists():run('programming_probe.py','.venv',['--model',str(a.models_root/model),
            '--programs',str(dest/model/'programs'),'--view',str(root/'benchmarks/views/gsm8k-v1'),
            '--guard',str(root/'benchmarks/raw/idea4_arc_challenge_v1/validation.jsonl'),'--output',str(out)],dest/(model+'_probe.log'))
    panels=[('gsm8k',['base','standard-seed11','soup','frobenius-r8','activation-r8']),
        ('gsm8k',['frobenius-r4','norm-matched-seed11','bi10','bi20','random10','tail10']),
        ('guards',spec['controls']+spec['programs'])]
    for group,variants in panels:
        for model in models:
            out=dest/model/'effects'
            if all((out/v/group/'complete.json').exists() for v in variants):continue
            run('programming_effects.py','.venv-bench',['--model',str(a.models_root/model),'--programs',str(dest/model/'programs'),
                '--probe',str(dest/model/'probe'),'--standard',str(root/'benchmarks/optimization/gsm8k'/model/'lora/seed11/delta'),
                '--tasks-dir',str(root/'benchmarks/tasks'),'--output',str(out),'--group',group,'--variants']+variants,
                dest/(model+'_'+group+'_'+variants[0]+'.log'))
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='full_programming_expansion_complete',spec=spec),indent=2))


if __name__=='__main__':main()
