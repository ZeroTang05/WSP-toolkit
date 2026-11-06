"""Same trained direction and action space for fixed, random, BO and LLM policies.

This is a finite, declared layerwise task-arithmetic adaptation. It does not
grant arbitrary tensor writes or use benchmark tests for candidate selection.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from weightlab import Lab,write_json
from delta_bank import DeltaBank,fingerprint,sha
from prepare_training_views import read_jsonl
from task_plan import METHOD,compile_coefficients
from search_policies import GRID,FIXED,available,propose,incumbent
from contract_registry import ContractRegistry,ToolRequest


NAMES=['optimize_public.py','search_policies.py','task_plan.py','delta_bank.py','weightlab.py',
    'prepare_training_views.py','finqa_views.py','contract_registry.py']
SOURCES={name:(Path(__file__).parent/name).read_bytes() for name in NAMES}


@torch.inference_mode()
def losses(lab,domain,guards):
    total=0.;nbytes=0
    for row in domain:
        prefix=lab.tokenizer.encode(lab.prefix(row),add_special_tokens=False)
        target=lab.tokenizer.encode(row['answer'],add_special_tokens=False)+[lab.tokenizer.eos_token_id]
        ids=torch.tensor([prefix+target],device='cuda')
        logits=lab.model(ids,use_cache=False).logits[0,len(prefix)-1:-1].float()
        total+=F.cross_entropy(logits,torch.tensor(target,device='cuda'),reduction='sum').item()
        nbytes+=len(row['answer'].encode('utf-8'))
    guard=[]
    for row in guards:
        choices=row['choices'];texts=choices['text'];labels=choices['label']
        prefix=lab.tokenizer.encode(lab.prefix(dict(prompt='Question: '+row['question']+'\nAnswer:')),add_special_tokens=False)
        scores=[]
        for text in texts:
            target=lab.tokenizer.encode(' '+text,add_special_tokens=False)
            if not target:raise ValueError('Empty ARC answer')
            ids=torch.tensor([prefix+target],device='cuda')
            logits=lab.model(ids,use_cache=False).logits[0,len(prefix)-1:-1].float()
            scores.append(-F.cross_entropy(logits,torch.tensor(target,device='cuda'),reduction='sum'))
        guard.append(float(-F.log_softmax(torch.stack(scores),dim=0)[labels.index(row['answerKey'])]))
    return total/(math.log(2)*nbytes),guard


def observation_norms(lab,meta,data):
    values=[dict(delta_squared=0.,base_squared=0.,targets=0) for _ in range(4)]
    for name,spec in meta['targets'].items():
        layer=int(name.split('.')[2]);band=min(3,4*layer//len(lab.layers))
        A=data[name+'.A'];B=data[name+'.B']
        squared=float(((A@A.T)*(B.T@B).T).sum())*(meta['coefficient']*spec['alpha']/spec['rank'])**2
        values[band]['delta_squared']+=max(0.,squared)
        values[band]['base_squared']+=float(lab.model.get_submodule(name).weight.detach().float().square().sum())
        values[band]['targets']+=1
    for row in values:row['relative_delta_norm']=math.sqrt(row['delta_squared']/row['base_squared'])
    return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--delta',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--guard',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',choices=['fixed','random','greedy','bo','llm'],required=True)
    p.add_argument('--controller',type=Path);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--queries',type=int,default=16);p.add_argument('--domain-n',type=int,default=64)
    p.add_argument('--guard-n',type=int,default=32);p.add_argument('--guard-margin',type=float,default=.02)
    p.add_argument('--without-guard',action='store_true');p.add_argument('--without-internal-observations',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if not 2<=a.queries<=16 or a.guard_margin<0:raise ValueError('Invalid search budget')
    if a.policy=='llm' and a.controller is None:raise ValueError('Local controller path required')
    registry=ContractRegistry();spec=registry.methods[METHOD]
    registry.validate(ToolRequest(METHOD,spec['observation'],spec['selection'],spec['mutation'],'none','development'))
    view_meta=json.loads((a.view/'manifest.json').read_text())
    if sha(a.view/'development.jsonl')!=view_meta['files']['development.jsonl']:raise ValueError('Changed development view')
    domain=read_jsonl(a.view/'development.jsonl')
    if any(r['view_role']!='development' or r['source_split']!='train' for r in domain):raise ValueError('Not a training-derived development view')
    if a.guard.name!='validation.jsonl':raise ValueError('Guard must be official ARC validation, never test')
    guards=read_jsonl(a.guard)
    if len(guards)!=299:raise ValueError('Not the full ARC-Challenge validation source')
    if not 1<=a.domain_n<=len(domain) or not 2<=a.guard_n<=len(guards):raise ValueError('Invalid observation subset size')
    guards=sorted(guards,key=lambda row:hashlib.sha256(row['id'].encode()).hexdigest())[:a.guard_n]
    domain=domain[:a.domain_n]
    config=dict(model=str(a.model.resolve()),delta=str(a.delta.resolve()),method_id=METHOD,policy=a.policy,
        seed=a.seed,queries=a.queries,domain_n=a.domain_n,guard_n=a.guard_n,guard_margin=a.guard_margin,
        without_guard=a.without_guard,without_internal_observations=a.without_internal_observations,smoke=a.smoke,
        controller=str(a.controller.resolve()) if a.controller else None,
        delta_manifest_sha256=sha(a.delta/'manifest.json'),domain_sha256=sha(a.view/'development.jsonl'),
        shared_training_protocol_sha256=sha(a.delta.parent/'protocol.json'),contract_spec_sha256=registry.spec_sha256,
        guard_sha256=sha(a.guard),domain_ids=[r['id'] for r in domain],guard_ids=[r['id'] for r in guards],
        coordinates='four normalized-depth bands; each coefficient in {0,.5,1,1.5}; shared trained LoRA direction',
        domain_proxy='teacher-forced reference-rationale CE including EOS, in bits per UTF-8 answer byte',
        guard_proxy='conditional-choice CE over ARC validation answer-text continuations; not harness acc_norm',
        guard_test='paired percentile bootstrap upper95 CE increase <= margin; finite-sample heuristic, not population guarantee',
        permission='No backward or training during search; prior LoRA training cost shared across policies and reported separately',
        selection='best feasible development risk including base; no benchmark test feedback',
        code_sha256={name:hashlib.sha256(data).hexdigest() for name,data in SOURCES.items()})
    if a.output.exists():raise FileExistsError('Fresh run required; preserve failed traces')
    a.output.mkdir(parents=True);write_json(a.output/'protocol.json',config)
    (a.output/'code').mkdir()
    for name,data in SOURCES.items():(a.output/'code'/name).write_bytes(data)
    (a.output/'contracts').mkdir();(a.output/'contracts/methods.json').write_bytes(registry.path.read_bytes())
    setup_started=time.perf_counter()
    lab=Lab(a.model);lab.model.requires_grad_(False);base=lab.manifest();write_json(a.output/'base_manifest.json',base)
    bank=DeltaBank(lab.model,fingerprint(base['files']));bank.register('direction',a.delta)
    meta,data=bank.roles['direction'];norms=observation_norms(lab,meta,data);write_json(a.output/'internal_observations.json',norms)
    controller=tokenizer=None
    if a.policy=='llm':
        from transformers import AutoModelForCausalLM,AutoTokenizer
        controller=AutoModelForCausalLM.from_pretrained(str(a.controller),local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
        tokenizer=AutoTokenizer.from_pretrained(str(a.controller),local_files_only=True)
        write_json(a.output/'controller_fingerprint.json',{p.name:sha(p) for p in a.controller.glob('*') if p.suffix in ['.json','.safetensors','.jinja']})
    rng=np.random.default_rng(a.seed);records=[];baseline_guard=None;bootstrap_indices=None
    setup_seconds=time.perf_counter()-setup_started
    started=time.perf_counter();controller_tokens=0
    for index in range(a.queries):
        proposal=dict(policy=a.policy,controller_seconds=0.,invalid_llm_proposal=False)
        if a.policy!='llm' or index<2:
            point=propose('fixed' if a.policy=='llm' else a.policy,records,rng,not a.without_guard)
        else:
            prompt=('Choose the next model-edit experiment. You may only scale a trained LoRA task vector '
                'in four normalized-depth bands. Each coordinate must be 0, 0.5, 1, or 1.5. '
                'Minimize domain risk while guard_upper95 <= guard_margin. No benchmark test data is available. '
                'Do not repeat previous coordinates. Return only JSON: {"coordinates":[a,b,c,d],"reason":"brief reasoning"}.\n'
                +json.dumps(dict(history=[{k:r[k] for k in ['coordinates','domain_risk','guard_upper95','guard_margin','feasible']} for r in records],
                    internal_observations=None if a.without_internal_observations else norms)))
            text=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer(text,return_tensors='pt',add_special_tokens=False).to('cuda');t=time.perf_counter()
            with torch.inference_mode():generated=controller.generate(**ids,max_new_tokens=192,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            response=tokenizer.decode(generated[0,ids.input_ids.shape[1]:],skip_special_tokens=True)
            controller_tokens+=generated.shape[1]-ids.input_ids.shape[1]
            proposal.update(controller_seconds=time.perf_counter()-t,prompt=prompt,response=response,
                input_tokens=ids.input_ids.shape[1],output_tokens=generated.shape[1]-ids.input_ids.shape[1])
            try:
                parsed=json.loads(response[response.index('{'):response.rindex('}')+1]);point=tuple(parsed['coordinates'])
                if point not in available(records):raise ValueError('Illegal or repeated coordinates')
            except (ValueError,KeyError,TypeError):
                proposal['invalid_llm_proposal']=True
                point=propose('random',records,rng,not a.without_guard)
        coefficients=compile_coefficients(meta['targets'],len(lab.layers),point)
        torch.cuda.synchronize();t=time.perf_counter()
        with bank.activate('direction',module_coefficients=coefficients):domain_risk,guard=losses(lab,domain,guards)
        torch.cuda.synchronize();evaluate_seconds=time.perf_counter()-t
        if baseline_guard is None:
            baseline_guard=np.asarray(guard);bootstrap_indices=np.random.default_rng(20261106).integers(0,len(guard),size=(2000,len(guard)))
        difference=np.asarray(guard)-baseline_guard
        upper=float(np.quantile(difference[bootstrap_indices].mean(1),.95))
        record=dict(index=index,coordinates=list(point),domain_risk=domain_risk,guard_ce=float(np.mean(guard)),
            guard_ce_per_item=guard,guard_upper95=upper,guard_margin=a.guard_margin,feasible=upper<=a.guard_margin,
            evaluate_seconds=evaluate_seconds,proposal=proposal,elapsed_seconds=time.perf_counter()-started)
        records.append(record)
        with open(a.output/'trace.jsonl','a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps({k:record[k] for k in ['index','coordinates','domain_risk','guard_upper95','feasible']}),flush=True)
    best=incumbent(records,not a.without_guard)
    plan=dict(method_id=METHOD,base_id=meta['base_id'],delta_sha256=meta['tensor_sha256'],n_layers=len(lab.layers),
        coordinates=best['coordinates'],selection_protocol_sha256=sha(a.output/'protocol.json'),policy=a.policy,
        implementation_grade='declared depth-wise task-arithmetic adaptation, not full reproduction of Task Arithmetic experiments',
        papers=['https://arxiv.org/abs/2212.04089','https://arxiv.org/abs/2106.09685'])
    write_json(a.output/'selected_plan.json',plan)
    write_json(a.output/'complete.json',dict(status='smoke_complete' if a.smoke else 'development_search_complete',
        selected_index=best['index'],coordinates=best['coordinates'],queries=len(records),policy=a.policy,
        evaluator_seconds=sum(r['evaluate_seconds'] for r in records),controller_tokens=controller_tokens,
        controller_input_tokens=sum(r['proposal'].get('input_tokens',0) for r in records),
        controller_seconds=sum(r['proposal']['controller_seconds'] for r in records),
        elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        setup_seconds_including_gpu_lock_wait=setup_seconds,
        note='Equal query counts only; total-cost matched comparisons and full heldout confirmation remain required. Domain/guard scores are development proxies, not public test accuracy.'))


if __name__=='__main__':main()
