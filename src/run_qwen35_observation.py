"""Original pinned WSP base: actual hybrid observations on training-derived text.

No reference edit is loaded, and no independent-discovery claim is made. This
validates the original architecture's usable observations, not math improvement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=32)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh output required')
    if not 2<=a.n<=128:raise ValueError('Use 2..128 calibration examples')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    manifest=json.loads((a.model/'idea4_wsp_base_manifest.json').read_text())
    if manifest['repo']!='Qwen/Qwen3.5-0.8B' or manifest['revision']!='2fc06364715b967f1860aea9cf38778875588b17':
        raise ValueError('Original pinned base required')
    for name,entry in manifest['files'].items():
        if sha(a.model/name)!=entry['sha256']:raise ValueError('Source file changed: '+name)
    view=json.loads((a.view/'manifest.json').read_text());data=a.view/'calibration.jsonl'
    if sha(data)!=view['files']['calibration.jsonl']:raise ValueError('Calibration source changed')
    rows=[json.loads(line) for line in data.read_text().splitlines()][:a.n]
    if len(rows)!=a.n or any(r['source_split']!='train' or r['view_role']!='calibration' for r in rows):
        raise ValueError('Training-derived calibration only')
    a.output.mkdir(parents=True);code=a.output/'code'
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,code/Path(__file__).name);sys.path.insert(0,str(code.resolve()))
    import numpy as np
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from weight_programming import WeightStore,ForwardPolicy,PracticeSet,ForwardService
    from weight_programming.qwen35_backend import Qwen35Backend
    torch.set_num_threads(4);started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    store=WeightStore(a.model,a.output/'edits');backend=Qwen35Backend(model,tokenizer,store)
    practice=PracticeSet([dict(id=r['id'],problem=r['prompt'],answer=r['answer']) for r in rows],role='calibration',name='gsm8k-training-derived')
    service=ForwardService(backend,store,practice,a.output/'service',ForwardPolicy(thinking=False))
    texts=[backend.chat([dict(role='user',content=r['prompt'])],False,generate=True)+r['answer']+tokenizer.eos_token for r in rows]
    protocol=dict(model_revision=manifest['revision'],base_manifest_sha256=sha(a.model/'idea4_wsp_base_manifest.json'),
        calibration_ids=[r['id'] for r in rows],calibration_sha256=sha(data),device='cpu',dtype='bfloat16',
        source_sha256={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')},
        scope='Original architecture observation study, not a generation benchmark or independent edit discovery.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(dict(event='source_verified',backend=backend.identity)),flush=True)
    result=service.request({'type':'teacher_forced','model':'original','texts':texts,
        'return':{'token_logprobs':True,'topk':3,'hidden_states':[0,12,24],'mlp_act':[0,12,23]}})
    folder=Path(result['artifact_dir']);lengths=np.load(folder/'lengths.npy');summary=[]
    for i,length in enumerate(lengths):
        item=dict(id=rows[i]['id'],tokens=int(length),rms={})
        for name in ['hidden_0','hidden_12','hidden_24','mlp_0','mlp_12','mlp_23']:
            values=np.load(folder/(name+'.npy'),mmap_mode='r')[i,:int(length)]
            item['rms'][name]=float(np.sqrt(np.square(values,dtype=np.float64).mean()))
        summary.append(item)
    (a.output/'observations.json').write_text(json.dumps(summary,indent=2))
    attention=service.request({'type':'teacher_forced','model':'original','texts':texts[:2],
        'return':{'attn_layers':[3,23]}})
    (a.output/'complete.json').write_text(json.dumps(dict(status='original_hybrid_observation_complete',n=a.n,
        total_input_tokens=int(lengths.sum()),backend=backend.identity,budget=attention['budget'],
        attention_metadata=str(Path(attention['artifact_dir'])/'meta.json'),elapsed_seconds=time.perf_counter()-started,
        scope=protocol['scope']),indent=2))
    print(json.dumps(dict(event='hybrid_observations_complete',tokens=int(lengths.sum()),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
