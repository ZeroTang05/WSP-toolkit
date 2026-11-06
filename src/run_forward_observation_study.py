"""Real-checkpoint tutorial: one budgeted dense observation on GSM calibration.

No candidate is selected and no task-improvement claim is made. This produces
reusable public-tool examples on the actual 0.6B checkpoint, not a toy score.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--n',type=int,default=128);p.add_argument('--hidden-layer',type=int,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh tutorial output')
    if a.n<=0:raise ValueError('Positive calibration size required')
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    view=json.loads((a.view/'manifest.json').read_text());data=a.view/'calibration.jsonl'
    if sha(data)!=view['files']['calibration.jsonl']:raise ValueError('Changed calibration view')
    rows=[json.loads(line) for line in data.read_text().splitlines()][:a.n]
    if len(rows)!=a.n or any(r['source_split']!='train' or r['view_role']!='calibration' for r in rows):raise ValueError('Only full declared training-derived calibration allowed')
    a.output.mkdir(parents=True);code=a.output/'code';code.mkdir()
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,code/Path(__file__).name)
    # Import the captured code, never a later mutable source tree.
    import sys
    sys.path.insert(0,str(code.resolve()))
    from weight_programming import WeightStore,ForwardService,ForwardPolicy,PracticeSet
    from weight_programming.hf_backend import HuggingFaceBackend
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.manual_seed(401);started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').eval()
    store=WeightStore(a.model,a.output/'edits');backend=HuggingFaceBackend(model,tokenizer,store)
    practice=PracticeSet([dict(id=r['id'],problem=r['prompt'],answer=r['answer'],source='gsm8k_train_calibration') for r in rows],role='calibration',name='gsm8k-v1-calibration')
    texts=[backend.chat([dict(role='user',content=r['prompt'])],False,generate=True)+r['answer']+tokenizer.eos_token for r in rows]
    policy=ForwardPolicy(thinking=False)
    request=dict(type='teacher_forced',model='original',texts=texts,**{'return':dict(token_logprobs=True,topk=3,hidden_states=[a.hidden_layer],mlp_act=[a.hidden_layer])})
    service=ForwardService(backend,store,practice,a.output/'service',policy)
    protocol=dict(model=str(a.model.resolve()),calibration_ids=[r['id'] for r in rows],calibration_sha256=sha(data),
        dtype='float32',device='cpu',thinking=False,request=request,source_sha256={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')},
        scope='Real 0.6B tool tutorial on calibration only. Reference text likelihood, not generation accuracy or BF16 public benchmark comparability.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(dict(event='observing',rows=len(rows),device='cpu')),flush=True)
    result=service.request(request);folder=Path(result['artifact_dir']);lengths=np.load(folder/'lengths.npy')
    lp=np.load(folder/'token_logprobs.npy',mmap_mode='r');top=np.load(folder/'topk_logprobs.npy',mmap_mode='r')
    hidden=np.load(folder/f'hidden_{a.hidden_layer}.npy',mmap_mode='r');mlp=np.load(folder/f'mlp_{a.hidden_layer}.npy',mmap_mode='r');summaries=[]
    for index,(row,length) in enumerate(zip(rows,lengths)):
        length=int(length);summaries.append(dict(id=row['id'],tokens=length,full_text_mean_logprob=float(lp[index,1:length].mean()),
            residual_rms=float(np.sqrt(np.mean(np.square(hidden[index,:length],dtype=np.float64)))),
            mlp_input_rms=float(np.sqrt(np.mean(np.square(mlp[index,:length],dtype=np.float64)))),
            mean_top3_probability_mass=float(np.exp(top[index,:length]).sum(-1).mean())))
    (a.output/'observations.json').write_text(json.dumps(summaries,indent=2))
    (a.output/'complete.json').write_text(json.dumps(dict(status='real_checkpoint_observation_tutorial_complete',n=len(rows),budget=result['budget'],
        elapsed_seconds=time.perf_counter()-started,returned_metadata=str(folder/'meta.json'),scope=protocol['scope']),indent=2))
    print(json.dumps(dict(event='real_observation_complete',n=len(rows),seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
