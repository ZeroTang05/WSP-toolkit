"""Controlled FinQA deployment: one base plus independently trained role deltas.

The inference input export contains public question, text, table, ID only.
Labels are read by the final scorer after all predictions have been committed.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from weightlab import Lab,write_json
from delta_bank import DeltaBank,fingerprint
from prepare_training_views import read_jsonl
from finqa_roles import retrieve_prompt,plan_prompt,verify_prompt,parse_retrieval,select_evidence
from finqa_executor import execute
from finqa_views import public_input


SOURCE_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parameter_hash(model):
    value=hashlib.sha256()
    for name,tensor in model.named_parameters():
        value.update(name.encode());value.update(tensor.detach().view(torch.uint8).cpu().numpy().tobytes())
    return value.hexdigest()


def adapter_identity(path):
    path=Path(path)
    return dict(path=str(path.resolve()),manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,type=Path)
    p.add_argument('--inputs',required=True,type=Path);p.add_argument('--labels',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--partition',choices=['dev','test'],required=True)
    p.add_argument('--mode',choices=['single','prompt','roles','multitask','replicas','random_roles'],required=True)
    p.add_argument('--retriever',type=Path);p.add_argument('--planner',type=Path);p.add_argument('--verifier',type=Path)
    p.add_argument('--multitask',type=Path);p.add_argument('--smoke',type=int)
    p.add_argument('--retry',type=int,choices=[0,1],default=1)
    a=p.parse_args()
    role_paths={}
    if a.mode in ['roles','replicas','random_roles']:
        role_paths={role:getattr(a,role) for role in ['retriever','planner','verifier']}
        if not all(role_paths.values()):raise ValueError('Three independently trained role deltas required')
    if a.mode=='multitask':
        if a.multitask is None:raise ValueError('Multitask delta required')
        role_paths={role:a.multitask for role in ['retriever','planner','verifier']}
    if a.mode=='random_roles':
        # Fixed cyclic derangement, reproducible and never tuned on labels.
        role_paths=dict(retriever=a.verifier,planner=a.retriever,verifier=a.planner)
    rows=read_jsonl(a.inputs)
    if any(set(row)!={'id','question','evidence','table'} for row in rows):raise ValueError('Input export contains unauthorized fields')
    if a.smoke is not None:
        if a.smoke<=0:raise ValueError('Positive smoke size required')
        rows=rows[:a.smoke]
    elif len(rows)!={'dev':883,'test':1147}[a.partition]:raise ValueError('Not a complete official partition')
    if len({row['id'] for row in rows})!=len(rows):raise ValueError('Duplicate inference IDs')
    helpers=['finqa_roles.py','finqa_executor.py','finqa_views.py','delta_bank.py','weightlab.py','prepare_training_views.py']
    source_files={name:(Path(__file__).parent/name).read_bytes() for name in helpers}
    protocol=dict(mode=a.mode,partition=a.partition,model=str(a.model.resolve()),inputs_sha256=hashlib.sha256(a.inputs.read_bytes()).hexdigest(),
        n=len(rows),smoke=a.smoke,retry=a.retry,roles={key:adapter_identity(path) for key,path in role_paths.items()},
        seed=20261106,thinking=False,dtype='bfloat16',generation='greedy, fresh KV per call',
        budgets=dict(retriever=256,planner=256,verifier=8),source_sha256=SOURCE_HASH,
        helpers_sha256={name:hashlib.sha256(data).hexdigest() for name,data in source_files.items()},
        verifier_policy='accept only literal accept; one planner repair after rejection when retry=1; score final attempted program',
        labels_used='final offline scorer only; no feedback into inference')
    a.output.mkdir(parents=True,exist_ok=True)
    config_path=a.output/'protocol.json'
    if config_path.exists() and json.loads(config_path.read_text())!=protocol:raise ValueError('Changed workflow protocol')
    if (a.output/'complete.json').exists():raise FileExistsError('Run already complete')
    write_json(config_path,protocol)
    (a.output/'code').mkdir(exist_ok=True)
    for name,data in source_files.items():(a.output/'code'/name).write_bytes(data)
    (a.output/'code/finqa_workflow.py').write_bytes(Path(__file__).read_bytes())
    lab=Lab(a.model);base=lab.manifest();write_json(a.output/'base_manifest.json',base)
    models={role:lab.model for role in ['retriever','planner','verifier']}
    if a.mode=='replicas':
        models['planner']=copy.deepcopy(lab.model)
        models['verifier']=copy.deepcopy(lab.model)
    banks={}
    if role_paths:
        if a.mode=='replicas':
            for role in models:
                bank=DeltaBank(models[role],fingerprint(base['files']))
                bank.register(role,role_paths[role]);banks[role]=bank
        else:
            bank=DeltaBank(lab.model,fingerprint(base['files']))
            for role in models:bank.register(role,role_paths[role]);banks[role]=bank
    else:
        for role in models:banks[role]=None
    before_hash=parameter_hash(lab.model)

    @torch.inference_mode()
    def generate(role,prompt,budget):
        prefix=lab.prefix(dict(prompt=prompt))
        ids=lab.tokenizer(prefix,return_tensors='pt',add_special_tokens=False).to('cuda')
        if ids.input_ids.shape[1]+budget>lab.model.config.max_position_embeddings:
            raise ValueError('Context budget exceeded; no silent input truncation')
        torch.cuda.synchronize();start=time.perf_counter()
        kwargs=dict(**ids,max_new_tokens=budget,do_sample=False,use_cache=True,pad_token_id=lab.tokenizer.pad_token_id)
        if banks[role] is None:output=models[role].generate(**kwargs)
        else:output=banks[role].generate(role,**kwargs)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        generated=output[0,ids.input_ids.shape[1]:]
        text=lab.tokenizer.decode(generated,skip_special_tokens=True)
        return dict(role=role,text=text,input_tokens=ids.input_ids.shape[1],output_tokens=generated.numel(),
            reached_token_budget=generated.numel()>=budget,elapsed_seconds=elapsed)

    # Warm up every resident execution path; reported separately.
    warmup_start=time.perf_counter()
    for role in models:generate(role,'Return the word ready.',8)
    warmup_seconds=time.perf_counter()-warmup_start
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    path=a.output/'predictions.jsonl';prior=read_jsonl(path) if path.exists() else []
    if len({r['id'] for r in prior})!=len(prior):raise ValueError('Duplicate committed predictions')
    expected={row['id'] for row in rows}
    if any(r['id'] not in expected for r in prior):raise ValueError('Unexpected resumed sample')
    done={row['id'] for row in prior};outputs=list(prior)
    for row in rows:
        if row['id'] in done:continue
        view={key:row[key] for key in ['question','evidence']};calls=[];selected_ids=[];valid_retrieval=None
        if a.mode=='single':selected=view
        else:
            retrieval=generate('retriever',retrieve_prompt(view),256);calls.append(retrieval)
            selected_ids,valid_retrieval=parse_retrieval(retrieval['text'],view)
            # No fallback to gold or full document; parse failures remain observable.
            selected=select_evidence(view,selected_ids)
        plan=generate('planner',plan_prompt(selected),256);calls.append(plan)
        program=plan['text'].strip();outcome=execute(program,row['table']);verdict=None
        if a.mode!='single':
            verification=generate('verifier',verify_prompt(selected,program,outcome),8);calls.append(verification)
            verdict=verification['text'].strip()
            if verdict!='accept' and a.retry:
                repair=generate('planner',plan_prompt(selected)+'\nThe previous program was rejected: '+program+'\nReturn a corrected program.',256)
                calls.append(repair);program=repair['text'].strip();outcome=execute(program,row['table'])
        record=dict(id=row['id'],input_hash=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest(),
            retrieved_ids=selected_ids,retrieval_parse_valid=valid_retrieval,program=program,outcome=outcome,
            first_program=plan['text'].strip(),verdict=verdict,calls=calls)
        with open(path,'a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
        outputs.append(record)
        print(json.dumps(dict(committed=len(outputs),total=len(rows),id=row['id'])),flush=True)
    if {row['id'] for row in outputs}!=expected:raise AssertionError('Incomplete inference')
    # All inference is finished before loading evaluation annotations.
    labels={row['id']:row for row in json.loads(a.labels.read_text())}
    if not expected<=set(labels):raise ValueError('Input IDs absent from evaluation labels')
    if not a.smoke and set(labels)!=expected:raise ValueError('Mismatched official partition IDs')
    input_by_id={row['id']:row for row in rows}
    for row in outputs:
        label=labels[row['id']]
        public=dict(id=label['id'],**public_input(label),table=label['table'])
        if public!=input_by_id[row['id']] or hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()!=row['input_hash']:
            raise ValueError('Inference export or committed prediction does not match official public input')
    after_hash=parameter_hash(lab.model)
    if before_hash!=after_hash:raise AssertionError('Workflow mutated base parameters')
    correct=first_correct=0;gold_support=hits=retrieval_examples=0;scored=[]
    for row in outputs:
        label=labels[row['id']];value=execute(row['program'],label['table'])
        success=value['valid'] and value['result']==label['qa']['exe_ans'];correct+=success
        first=execute(row['first_program'],label['table']);first_correct+=first['valid'] and first['result']==label['qa']['exe_ans']
        support=set(label['qa']['gold_inds']);gold_support+=len(support)
        hits+=len(support&set(row['retrieved_ids']));retrieval_examples+=bool(support<=set(row['retrieved_ids']))
        scored.append(dict(id=row['id'],execution_correct=bool(success)))
    write_json(a.output/'scored.json',scored)
    calls=[call for row in outputs for call in row['calls']]
    base_bytes=sum(v.numel()*v.element_size() for v in lab.model.parameters())
    write_json(a.output/'complete.json',dict(status='smoke_complete' if a.smoke else 'full_complete',
        n=len(outputs),execution_accuracy=correct/len(outputs),first_program_execution_accuracy=first_correct/len(outputs),
        retrieval_support_recall=hits/gold_support if gold_support and a.mode!='single' else None,
        retrieval_all_support_rate=retrieval_examples/len(outputs) if a.mode!='single' else None,
        role_calls=len(calls),generated_tokens=sum(call['output_tokens'] for call in calls),
        prompt_tokens=sum(call['input_tokens'] for call in calls),generation_seconds=sum(call['elapsed_seconds'] for call in calls),
        current_process_wall_seconds=time.perf_counter()-started,warmup_seconds=warmup_seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),base_tensor_bytes=base_bytes,
        resident_base_copies=3 if a.mode=='replicas' else 1,
        delta_file_bytes=sum((p/'adapter.safetensors').stat().st_size for p in set(role_paths.values())),
        base_parameter_hash=before_hash,base_parameters_immutable=True,
        labels_sha256=hashlib.sha256(a.labels.read_bytes()).hexdigest(),
        note='Role learning uses synthetic verifier negatives and teacher-forced evidence; no claim these labels prove grounding. Wall/warmup timing describes this process only when resuming.'))


if __name__=='__main__':main()
