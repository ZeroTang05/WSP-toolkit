"""Real 0.6B patching maps on explicit matched factual-completion pairs.

This is a diagnostic demonstration, not an accuracy benchmark, weight-improvement
experiment or original-WSP discovery run. Capital facts specify metric tokens.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

PAIRS=[('France','Italy','Paris','Rome'),('Germany','Spain','Berlin','Madrid'),
       ('Japan','China','Tokyo','Beijing'),('Greece','Egypt','Athens','Cairo')]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh study output required')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    a.output.mkdir(parents=True);code=a.output/'code'
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,code/Path(__file__).name);sys.path.insert(0,str(code.resolve()))
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from weight_programming import WeightStore,ForwardPolicy
    from weight_programming.hf_backend import HuggingFaceBackend
    from weight_programming.patching import ActivationPatchingStudy
    torch.set_num_threads(4);started=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').eval()
    store=WeightStore(a.model,a.output/'edits');backend=HuggingFaceBackend(model,tok,store)
    cases=[]
    for first,second,target,alternative in PAIRS:
        clean='The capital of '+first+' is';corrupt='The capital of '+second+' is'
        ids=backend.encode(clean)
        target_ids=backend.encode(' '+target);alternative_ids=backend.encode(' '+alternative)
        if len(ids)!=len(backend.encode(corrupt)) or len(target_ids)!=1 or len(alternative_ids)!=1:
            raise ValueError('Predeclared paired length/single-token metric mismatch')
        cases.append(dict(id=first+'_'+second,clean_text=clean,corrupt_text=corrupt,length=len(ids),target_token=target_ids[0],alternative_token=alternative_ids[0]))
    count=len(cases)*(2*backend.num_layers+2)
    policy=ForwardPolicy(passes=count,max_text_tokens=512)
    study=ActivationPatchingStudy(backend,store,a.output/'study',data_role='development',
        corruption_description='Authored matched capital-completion prompts; swap one country token at equal token length. No claim of independent random draws or natural failure distribution.',policy=policy)
    protocol=dict(cases=cases,backend=backend.identity,sites=['last-position residual at 0..L','last-position MLP input at 0..L-1','all-position embedding positive control'],
        planned_requests=count,source_sha256={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')},
        scope='Real-model method demonstration only. Temporary activations, no weights changed. Separate intervention track, not the original WSP permission/budget track.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[]
    for case in cases:
        arguments={k:case[k] for k in ['clean_text','corrupt_text','target_token','alternative_token']}
        for kind,layers in [('residual',range(backend.num_layers+1)),('mlp_input',range(backend.num_layers))]:
            for layer in layers:
                result=study.run(**arguments,kind=kind,layer=layer,positions=[case['length']-1])
                rows.append(dict(case=case['id'],kind=kind,layer=layer,positions='last',
                    clean=result['clean'],corrupt=result['corrupt'],patched=result['patched'],effects=result['effects']))
        control=study.run(**arguments,kind='residual',layer=0,positions=list(range(case['length'])))
        rows.append(dict(case=case['id'],kind='residual',layer=0,positions='all',
            clean=control['clean'],corrupt=control['corrupt'],patched=control['patched'],effects=control['effects']))
        print(json.dumps(dict(event='pair_complete',case=case['id'],successful=study.ledger.summary()['successful'])),flush=True)
    (a.output/'results.json').write_text(json.dumps(rows,indent=2))
    (a.output/'complete.json').write_text(json.dumps(dict(status='real_activation_patching_maps_complete',requests=len(rows),
        executed_model_forwards=3*len(rows),elapsed_seconds=time.perf_counter()-started,budget=study.ledger.summary(),scope=protocol['scope']),indent=2))


if __name__=='__main__':main()
