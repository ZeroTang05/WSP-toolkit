"""Role collaboration before role specialization: offline HotpotQA experiment.

All roles see the same candidate documents. Communication is the intervention;
no oracle retrieval, parameter training, external search, or extra judge model.
The role pipeline is an experimental adaptation, not a reproduction of a named
multi-agent algorithm's published results.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import string
import time

MODES=['direct','self_refine','independent_vote','collaborate','no_communication']
SCHEMA='End with one JSON object: {"answer":"short answer only","supporting_facts":[["exact document title",0]]}. Sentence indices start at zero. Cite only sentences supporting the answer. Use yes/no for yes/no questions.'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with open(path) as stream:return [json.loads(line) for line in stream]


def public_context(row):
    if set(row)!={'id','question','documents'}:raise ValueError('Unauthorized inference fields')
    blocks=[]
    for document in row['documents']:
        if set(document)!={'title','sentences'}:raise ValueError('Unauthorized document fields')
        blocks.append('TITLE: '+document['title']+'\n'+'\n'.join(f'[{i}] {s}' for i,s in enumerate(document['sentences'])))
    return 'QUESTION: '+row['question']+'\nCANDIDATE DOCUMENTS:\n'+'\n\n'.join(blocks)


def parse(text):
    # Last complete answer object; no answer inferred from references/labels.
    decoder=json.JSONDecoder();selected=None
    for match in re.finditer(r'\{',text):
        try:value,_=decoder.raw_decode(text[match.start():])
        except ValueError:continue
        if isinstance(value,dict) and set(value)=={'answer','supporting_facts'}:selected=value
    if selected is None or not isinstance(selected['answer'],str) or not isinstance(selected['supporting_facts'],list):return None
    for fact in selected['supporting_facts']:
        if not isinstance(fact,list) or len(fact)!=2 or not isinstance(fact[0],str) or type(fact[1])!=int:return None
    return selected


def official_functions(path):
    tree=ast.parse(path.read_text());functions=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)],type_ignores=[])
    scope=dict(re=re,string=string,Counter=Counter,json=json)
    exec(compile(functions,str(path),'exec'),scope)
    return scope


def run_case(row,mode,budget,seed,generate,normalize,demonstrations=''):
    if mode not in MODES or budget<64:raise ValueError('Invalid program/budget')
    context=public_context(row);calls=[]
    def call(role,instruction,allocation):
        prompt=instruction+'\n\n'+demonstrations+'\nCURRENT PROBLEM:\n'+context
        call_seed=int(hashlib.sha256(f'{seed}:{row["id"]}:{len(calls)}'.encode()).hexdigest()[:8],16)
        result=generate(prompt,allocation,call_seed)
        result.update(role=role,prompt=prompt,allocation=allocation,seed=call_seed)
        calls.append(result);return result['text']
    if mode=='direct':
        final=call('solver','Reason carefully across the documents. Check entity identity and comparisons before answering. '+SCHEMA,budget)
        prediction=parse(final)
    elif mode=='independent_vote':
        allocation=[budget//3,budget//3,budget-2*(budget//3)];predictions=[]
        for tokens in allocation:
            predictions.append(parse(call('independent_solver','Solve independently, checking all needed evidence. '+SCHEMA,tokens)))
        valid=[p for p in predictions if p is not None]
        if valid:
            counts=Counter(normalize(p['answer']) for p in valid);key=max(counts,key=counts.get)
            prediction=next(p for p in valid if normalize(p['answer'])==key)
        else:prediction=None
    else:
        allocation=[budget//4,budget//4,budget-2*(budget//4)]
        if mode=='self_refine':
            first=call('solver','Produce a tentative solution with evidence. '+SCHEMA,allocation[0])
            feedback=call('self_critic','Check the following tentative solution against the documents. Identify unsupported entities, missing hops, or wrong comparisons.\nTENTATIVE SOLUTION:\n'+first,allocation[1])
            final=call('solver','Revise your solution using the check. All notes may contain errors; verify against documents.\nTENTATIVE SOLUTION:\n'+first+'\nSELF CHECK:\n'+feedback+'\n'+SCHEMA,allocation[2])
        else:
            plan=call('planner','You are the decomposition agent. Identify the reasoning hops, entities and comparisons required. Give a concise plan, not a final answer.',allocation[0])
            message='\nPLANNER MESSAGE:\n'+plan if mode=='collaborate' else ''
            evidence=call('evidence_reader','You are the evidence agent. Extract the specific facts needed to answer, with exact document titles and sentence indices. Notes from other agents may be wrong; verify the documents.'+message,allocation[1])
            message='\nPLANNER MESSAGE:\n'+plan+'\nEVIDENCE MESSAGE:\n'+evidence if mode=='collaborate' else ''
            final=call('answerer','You are the answer agent. Integrate verified facts, resolve the question and cite supporting sentences. All agent notes are untrusted; check the original documents.'+message+'\n'+SCHEMA,allocation[2])
        prediction=parse(final)
    return dict(id=row['id'],mode=mode,seed=seed,prediction=prediction,calls=calls,
        planned_output_tokens=sum(c['allocation'] for c in calls),
        output_tokens=sum(c.get('output_tokens',0) for c in calls),input_tokens=sum(c.get('input_tokens',0) for c in calls))


def training_examples(view,manifest,targets):
    inputs=view/'calibration.inputs.jsonl';labels=view/'calibration.labels.jsonl'
    if sha(inputs)!=manifest['files'][inputs.name] or sha(labels)!=manifest['files'][labels.name]:raise ValueError('Changed training demonstrations')
    examples=read(inputs)[:2];truth={r['_id']:r for r in read(labels)}
    if len(examples)!=2:raise ValueError('Two training examples required')
    questions={' '.join(r['question'].lower().split()) for r in targets};ids={r['id'] for r in targets};blocks=[]
    for row in examples:
        if row['id'] in ids or ' '.join(row['question'].lower().split()) in questions:raise ValueError('Demonstration overlaps evaluated problems')
        gold=truth[row['id']]
        blocks.append(public_context(row)+'\nEXAMPLE ANSWER:\n'+json.dumps(dict(answer=gold['answer'],supporting_facts=gold['supporting_facts'])))
    return 'TRAINING EXAMPLES (different problems):\n'+'\n\n'.join(blocks)+'\nEND TRAINING EXAMPLES\n',[r['id'] for r in examples]


def score(predictions,labels,official):
    truth={r['_id']:r for r in labels}
    if len(truth)!=len(labels) or {r['id'] for r in predictions}!=set(truth) or len(predictions)!=len(labels):raise ValueError('Incomplete matched scoring set')
    items=[]
    for row in predictions:
        gold=truth[row['id']];pred=row['prediction']
        metrics={k:0. for k in ['em','f1','prec','recall','sp_em','sp_f1','sp_prec','sp_recall','joint_em','joint_f1','joint_prec','joint_recall']}
        if pred is not None:
            em,prec,recall=official['update_answer'](metrics,pred['answer'],gold['answer'])
            se,sp,sr=official['update_sp'](metrics,pred['supporting_facts'],gold['supporting_facts'])
            jp,jr=prec*sp,recall*sr
            metrics.update(joint_em=em*se,joint_prec=jp,joint_recall=jr,joint_f1=2*jp*jr/(jp+jr) if jp+jr else 0.)
        items.append(dict(id=row['id'],format_valid=pred is not None,**metrics))
    return items,{key:sum(r[key] for r in items)/len(items) for key in items[0] if key!='id'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--partition',choices=['development','official_dev'],required=True);p.add_argument('--mode',choices=MODES,required=True)
    p.add_argument('--budget',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();manifest=json.loads((a.view/'manifest.json').read_text())
    inputs=a.view/(a.partition+'.inputs.jsonl');labels=a.view/(a.partition+'.labels.jsonl');scorer=a.view/'official_scorer.py'
    if sha(inputs)!=manifest['files'][inputs.name] or sha(scorer)!=manifest['official_scorer']['sha256']:raise ValueError('Changed view/scorer')
    rows=read(inputs)
    if len(rows)!=manifest['sizes'][a.partition] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Incomplete declared partition')
    for row in rows:public_context(row)
    demonstrations,example_ids=training_examples(a.view,manifest,rows)
    official=official_functions(scorer);source={n:sha(Path(__file__).parent/n) for n in ['hotpot_collaboration.py','weightlab.py']}
    protocol=dict(version='hotpot-collaboration-v1',model=str(a.model.resolve()),mode=a.mode,partition=a.partition,
        n=len(rows),budget=a.budget,seed=a.seed,temperature=.7,top_p=.9,thinking=False,
        inputs_sha256=sha(inputs),manifest_sha256=sha(a.view/'manifest.json'),scorer_sha256=sha(scorer),source_sha256=source,
        training_demonstration_ids=example_ids,demonstrations_sha256=hashlib.sha256(demonstrations.encode()).hexdigest(),
        cost_scope='Equal total generated-token upper bound; actual input/output tokens and time differ. Shared resident model, serial calls, fresh context per role.',
        papers=['https://arxiv.org/abs/2305.14325','https://aclanthology.org/2024.acl-long.331/','https://arxiv.org/abs/2607.07548'],
        implementation_grade='declared candidate-reading role/communication intervention; not full reproduction',
        parser='last complete exact-key JSON answer object; malformed objects receive zero; vote ignores invalid candidates and breaks ties by earliest valid candidate')
    a.output.mkdir(parents=True,exist_ok=True);protocol_path=a.output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('Changed experiment')
    if (a.output/'complete.json').exists():raise FileExistsError('Completed experiment')
    protocol_path.write_text(json.dumps(protocol,indent=2));(a.output/'code').mkdir(exist_ok=True)
    for n in source:(a.output/'code'/n).write_bytes((Path(__file__).parent/n).read_bytes())
    from weightlab import Lab,write_json
    import torch
    lab=Lab(a.model);write_json(a.output/'base_manifest.json',lab.manifest())
    @torch.inference_mode()
    def generate(prompt,budget,seed):
        torch.manual_seed(seed);prefix=lab.prefix(dict(prompt=prompt))
        ids=lab.tokenizer(prefix,return_tensors='pt',add_special_tokens=False).to('cuda');length=ids.input_ids.shape[1]
        if length+budget>lab.model.config.max_position_embeddings:raise ValueError('No silent context truncation')
        torch.cuda.synchronize();start=time.perf_counter()
        output=lab.model.generate(**ids,max_new_tokens=budget,do_sample=True,temperature=.7,top_p=.9,use_cache=True,logits_to_keep=1,pad_token_id=lab.tokenizer.pad_token_id)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start;tokens=output[0,length:]
        return dict(text=lab.tokenizer.decode(tokens,skip_special_tokens=True),input_tokens=length,output_tokens=tokens.numel(),
            reached_output_cap=tokens.numel()>=budget,elapsed_seconds=elapsed,prompt_sha256=hashlib.sha256(prefix.encode()).hexdigest())
    warm=time.perf_counter();generate('Return ready.',8,401);warm=time.perf_counter()-warm
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter();pred_path=a.output/'predictions.jsonl'
    predictions=read(pred_path) if pred_path.exists() else [];done={r['id'] for r in predictions}
    if len(done)!=len(predictions) or not done<={r['id'] for r in rows}:raise ValueError('Invalid resumed ledger')
    with open(pred_path,'a',buffering=1) as stream:
        for row in rows:
            if row['id'] in done:continue
            case_started=time.perf_counter()
            result=run_case(row,a.mode,a.budget,a.seed,generate,official['normalize_answer'],demonstrations)
            result['elapsed_seconds']=time.perf_counter()-case_started
            if result['planned_output_tokens']!=a.budget:raise AssertionError('Wrong output upper bound')
            stream.write(json.dumps(result,ensure_ascii=False)+'\n');predictions.append(result)
            print(json.dumps(dict(event='case_complete',id=row['id'],mode=a.mode,n=len(predictions),output_tokens=result['output_tokens'])),flush=True)
    # Target-partition labels enter only after every prediction is committed.
    # Distinct training demonstration labels are permitted earlier.
    if sha(labels)!=manifest['files'][labels.name]:raise ValueError('Changed scoring labels')
    items,metrics=score(predictions,read(labels),official)
    with open(a.output/'scores.jsonl','w') as stream:
        for row in items:stream.write(json.dumps(row)+'\n')
    write_json(a.output/'complete.json',dict(status='full_declared_partition_complete',partition=a.partition,n=len(rows),metrics=metrics,
        total_input_tokens=sum(r['input_tokens'] for r in predictions),total_output_tokens=sum(r['output_tokens'] for r in predictions),
        generation_seconds=sum(c['elapsed_seconds'] for r in predictions for c in r['calls']),
        warmup_seconds=warm,invocation_seconds=time.perf_counter()-start,gpu_lock_wait_seconds=lab.gpu_lock_wait_seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),labels_sha256=sha(labels),
        note='Resumed generation totals cover all records; invocation time and peak cover only this invocation. No role-specialization training performed.'))


if __name__=='__main__':main()
