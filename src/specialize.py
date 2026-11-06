"""Train a standard LoRA reference and save only a delta plus optimizer state.

This is a baseline/direction source for later controlled tool-search experiments;
it does not query benchmark tests or claim an improved workflow.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from weightlab import Lab,write_json
from paper_methods import LoRALinear
from delta_bank import write_delta
from contract_registry import ContractRegistry,ToolRequest
from prepare_training_views import read_jsonl

IMPLEMENTATION_NAMES=['paper_methods.py','delta_bank.py','contract_registry.py','weightlab.py',
    'prepare_training_views.py','finqa_views.py']
SOURCE_BUNDLE={name:(Path(__file__).parent/name).read_bytes()
    for name in ['specialize.py']+IMPLEMENTATION_NAMES}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--view',required=True,type=Path)
    p.add_argument('--train-file',default='train.jsonl')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--alpha',type=float,default=16)
    p.add_argument('--lr',type=float,default=.0002)
    p.add_argument('--max-length',type=int,default=1536)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.steps<=0 or a.batch<=0 or a.rank<=0 or a.lr<=0:raise ValueError('Invalid training budget')
    manifest=json.loads((a.view/'manifest.json').read_text())
    training=a.view/a.train_file
    if sha(training)!=manifest['files'].get(a.train_file):raise ValueError('Unregistered or changed training view')
    rows=read_jsonl(training)
    if not rows or any(r['view_role']!='train' or r['source_split']!='train' for r in rows):
        raise ValueError('Only official training examples may update parameters')
    if a.batch>len(rows):raise ValueError('Effective batch exceeds the training view')
    registry=ContractRegistry()
    spec=registry.methods['lora.functional_delta.v1']
    registry.validate(ToolRequest(spec['id'],spec['observation'],spec['selection'],spec['mutation'],'train','train'))
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='resume'}
    config.update(training_sha256=sha(training),
        code_sha256=hashlib.sha256(SOURCE_BUNDLE['specialize.py']).hexdigest(),contract_id=spec['id'])
    config['implementation_files']={name:hashlib.sha256(SOURCE_BUNDLE[name]).hexdigest()
        for name in IMPLEMENTATION_NAMES}
    config['contract_spec_sha256']=registry.spec_sha256
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'protocol.json').exists():
        if json.loads((a.output/'protocol.json').read_text())!=config:raise ValueError('Training protocol changed')
        if not a.resume:raise FileExistsError('Existing run requires --resume')
        if (a.output/'complete.json').exists():raise FileExistsError('Training already complete')
    else:write_json(a.output/'protocol.json',config)
    (a.output/'code').mkdir(exist_ok=True)
    for name,data in SOURCE_BUNDLE.items():(a.output/'code'/name).write_bytes(data)
    (a.output/'contracts').mkdir(exist_ok=True)
    spec_bytes=registry.path.read_bytes()
    if hashlib.sha256(spec_bytes).hexdigest()!=registry.spec_sha256:
        raise ValueError('Method specification changed while constructing the run')
    (a.output/'contracts/methods.json').write_bytes(spec_bytes)
    lab=Lab(a.model)
    source=lab.manifest()
    if (a.output/'base_manifest.json').exists() and json.loads((a.output/'base_manifest.json').read_text())!=source:
        raise ValueError('Base model changed since the previous training state')
    write_json(a.output/'base_manifest.json',source)
    encoded=[]
    too_long=[]
    for row in rows:
        prefix=lab.tokenizer.encode(lab.prefix(row),add_special_tokens=False)
        target=lab.tokenizer.encode(row['answer'],add_special_tokens=False)+[lab.tokenizer.eos_token_id]
        if len(prefix)+len(target)>a.max_length:too_long.append(row['id'])
        encoded.append((prefix,target))
    if too_long:
        write_json(a.output/'length_violations.json',dict(n=len(too_long),ids=too_long,max_length=a.max_length))
        raise ValueError('Length audit must be resolved before training; no silent example truncation or dropping')
    torch.manual_seed(a.seed)
    base_parameters=sum(v.numel() for v in lab.model.parameters())
    lab.model.requires_grad_(False)
    wrapped={}
    for i,layer in enumerate(lab.layers):
        for name in ['q_proj','v_proj']:
            parent=layer.self_attn
            wrapper=LoRALinear(getattr(parent,name),rank=a.rank,alpha=a.alpha)
            setattr(parent,name,wrapper)
            wrapped[f'model.layers.{i}.self_attn.{name}']=wrapper
    lab.model.train()
    lab.model.config.use_cache=False
    lab.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable=[v for v in lab.model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=a.lr,weight_decay=0)
    rng=random.Random(a.seed)
    start_step=0
    stats=dict(input_tokens=0,supervised_tokens=0,examples=0)
    checkpoint=a.output/'state.pt'
    if a.resume and checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        with torch.no_grad():
            for name,module in wrapped.items():
                module.A.copy_(state['factors'][name+'.A'].to(module.A.device))
                module.B.copy_(state['factors'][name+'.B'].to(module.B.device))
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        rng.setstate(state['python_rng'])
        start_step=state['step'];stats=state['stats']
    began=time.perf_counter()
    for step in range(start_step,a.steps):
        optimizer.zero_grad(set_to_none=True)
        batch=rng.sample(range(len(rows)),a.batch)
        total_target=sum(len(encoded[i][1]) for i in batch)
        loss_total=0.
        for index in batch:
            prefix,target=encoded[index]
            ids=torch.tensor([prefix+target],device='cuda')
            logits=lab.model(ids,use_cache=False).logits[0,len(prefix)-1:-1].float()
            loss=F.cross_entropy(logits,torch.tensor(target,device='cuda'),reduction='sum')/total_target
            loss.backward();loss_total+=loss.item()
            stats['input_tokens']+=len(prefix)+len(target)
            stats['supervised_tokens']+=len(target);stats['examples']+=1
            del logits,ids,loss
        norm=float(torch.nn.utils.clip_grad_norm_(trainable,1.))
        lr=a.lr*min(1.,(step+1)/10)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        record=dict(step=step+1,nll=loss_total,gradient_norm=norm,lr=lr,elapsed_seconds=time.perf_counter()-began,**stats)
        print(json.dumps(record),flush=True)
        with open(a.output/'training.jsonl','a') as stream:stream.write(json.dumps(record)+'\n')
        if (step+1)%10==0 or step+1==a.steps:
            factors={name+'.'+side:getattr(module,side).detach().cpu().clone()
                for name,module in wrapped.items() for side in ['A','B']}
            state=dict(step=step+1,factors=factors,optimizer=optimizer.state_dict(),stats=stats,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=rng.getstate())
            torch.save(state,a.output/'state.pt.tmp');(a.output/'state.pt.tmp').replace(checkpoint)
    factors={name+'.'+side:getattr(module,side).detach().cpu().clone()
        for name,module in wrapped.items() for side in ['A','B']}
    targets={name:dict(in_features=module.base.in_features,out_features=module.base.out_features,
        rank=a.rank,alpha=a.alpha) for name,module in wrapped.items()}
    if not (a.output/'delta/manifest.json').exists():
        write_delta(factors,targets,source,a.output/'delta',1.,dict(method='LoRA SFT',
            paper='https://arxiv.org/abs/2106.09685',contract_id=spec['id'],seed=a.seed,
            train_view_sha256=sha(training),objective='global effective-batch mean target-token CE including EOS',
            scope='public training reference; no test or development feedback used'))
    write_json(a.output/'complete.json',dict(status='trained_delta_only',steps=a.steps,
        trainable_parameters=sum(v.numel() for v in trainable),training_stats=stats,
        base_parameters=base_parameters,trainable_fraction=sum(v.numel() for v in trainable)/base_parameters,
        elapsed_training_seconds=time.perf_counter()-began,delta_directory=str((a.output/'delta').resolve()),
        full_checkpoint_saved=False))


if __name__=='__main__':main()
