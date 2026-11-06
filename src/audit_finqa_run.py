"""Recompute FinQA scores and verifier decisions from committed raw predictions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from finqa_executor import execute
from finqa_views import public_input
from finqa_roles import parse_retrieval


def audit(run,labels_path,verify_prompts=False):
    protocol=json.loads((run/'protocol.json').read_text());complete=json.loads((run/'complete.json').read_text())
    with open(run/'predictions.jsonl') as stream:predictions=[json.loads(line) for line in stream if line.strip()]
    labels={row['id']:row for row in json.loads(labels_path.read_text())}
    if len(predictions)!=len({row['id'] for row in predictions}) or len(predictions)!=protocol['n']:
        raise ValueError('Duplicate IDs or incomplete committed predictions')
    if complete['n']!=len(predictions):raise ValueError('Completion count mismatch')
    expected={'dev':883,'test':1147}[protocol['partition']]
    if protocol.get('smoke') is None and (len(predictions)!=expected or {r['id'] for r in predictions}!=set(labels)):
        raise ValueError('Incomplete official partition')
    for name,digest in protocol['helpers_sha256'].items():
        if hashlib.sha256((run/'code'/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed helper snapshot')
    entries=list((run/'code').glob('finqa_workflow*.py'))
    if len(entries)!=1 or hashlib.sha256(entries[0].read_bytes()).hexdigest()!=protocol['source_sha256']:
        raise ValueError('Workflow source mismatch')
    if hashlib.sha256(labels_path.read_bytes()).hexdigest()!=complete['labels_sha256']:raise ValueError('Changed official labels')
    scored={row['id']:row for row in json.loads((run/'scored.json').read_text())}
    if set(scored)!={row['id'] for row in predictions}:raise ValueError('Scored sample IDs mismatch')
    correct=valid=first_correct=support_total=support_hits=support_complete=0
    repairs=wrong_to_correct=correct_to_wrong=0
    calls=[];confusion=dict(accept_correct=0,accept_wrong=0,reject_correct=0,reject_wrong=0)
    for row in predictions:
        label=labels[row['id']];public=dict(id=label['id'],**public_input(label),table=label['table'])
        if hashlib.sha256(json.dumps(public,sort_keys=True).encode()).hexdigest()!=row['input_hash']:
            raise ValueError('Public input fingerprint mismatch')
        final=execute(row['program'],label['table']);first=execute(row['first_program'],label['table'])
        if final!=row['outcome']:raise ValueError('Logged execution does not match program')
        success=bool(final['valid'] and final['result']==label['qa']['exe_ans'])
        first_success=bool(first['valid'] and first['result']==label['qa']['exe_ans'])
        if bool(scored[row['id']]['execution_correct'])!=success:raise ValueError('Per-item score mismatch')
        correct+=success;valid+=final['valid'];first_correct+=first_success
        support=set(label['qa']['gold_inds']);support_total+=len(support)
        support_hits+=len(support&set(row['retrieved_ids']));support_complete+=support<=set(row['retrieved_ids'])
        if row['verdict'] is not None:
            key=('accept_' if row['verdict']=='accept' else 'reject_')+('correct' if first_success else 'wrong')
            confusion[key]+=1
        row_calls=row['calls'];mode=protocol['mode']
        planner_calls=[call for call in row_calls if call['role']=='planner']
        if len(planner_calls)>1:
            repairs+=1;wrong_to_correct+=not first_success and success;correct_to_wrong+=first_success and not success
        verifier_calls=[call for call in row_calls if call['role']=='verifier']
        if (not planner_calls or planner_calls[0]['text'].strip()!=row['first_program']
            or planner_calls[-1]['text'].strip()!=row['program']):raise ValueError('Programs do not match raw planner outputs')
        if (verifier_calls[0]['text'].strip() if verifier_calls else None)!=row['verdict']:
            raise ValueError('Verdict does not match raw verifier output')
        retrieval_calls=[call for call in row_calls if call['role']=='retriever']
        if retrieval_calls:
            ids,valid_parse=parse_retrieval(retrieval_calls[0]['text'],public_input(label))
            if ids!=row['retrieved_ids'] or valid_parse!=row['retrieval_parse_valid']:
                raise ValueError('Retrieval does not match raw output')
        expected_roles=['planner'] if mode=='single' else ['planner']*4 if mode=='base_refine' else (
            ([] if mode in ['no_retriever','oracle_retrieval'] else ['retriever'])+['planner']+
            ([] if mode=='no_verifier' else ['verifier'])+
            (['planner'] if mode not in ['no_verifier'] and row['verdict']!='accept' and protocol['retry'] else []))
        if [call['role'] for call in row_calls]!=expected_roles:raise ValueError('Unexpected workflow calls')
        for call in row_calls:
            budget=194 if mode=='base_refine' else protocol['budgets'][call['role']]
            if not 0<=call['output_tokens']<=budget or call['input_tokens']<1 or call['elapsed_seconds']<0:
                raise ValueError('Invalid call budget/cost')
        calls.extend(row_calls)
    calculated=dict(execution_accuracy=correct/len(predictions),first_program_execution_accuracy=first_correct/len(predictions),
        program_validity_rate=valid/len(predictions),role_calls=len(calls),
        generated_tokens=sum(call['output_tokens'] for call in calls),prompt_tokens=sum(call['input_tokens'] for call in calls),
        generation_seconds=sum(call['elapsed_seconds'] for call in calls))
    mode=protocol['mode']
    if support_total and mode not in ['single','base_refine','no_retriever']:
        calculated.update(retrieval_support_recall=support_hits/support_total,retrieval_all_support_rate=support_complete/len(predictions))
    for key,value in calculated.items():
        if key in complete and not math.isclose(value,complete[key],rel_tol=1e-10,abs_tol=1e-10):raise ValueError('Aggregate mismatch: '+key)
    prompt_audit=None
    if verify_prompts:
        from audit_finqa_prompts import audit_prompts
        prompt_audit=audit_prompts(run,protocol,labels,predictions)
    return dict(status='raw_execution_and_cost_verified',n=len(predictions),metrics=calculated,verifier_confusion=confusion,
        correct=correct,first_correct=first_correct,valid=valid,
        repair_audit=dict(samples_with_multiple_planner_calls=repairs,wrong_to_correct=wrong_to_correct,correct_to_wrong=correct_to_wrong),
        prompt_audit=prompt_audit,
        diagnostic_oracle=protocol.get('diagnostic_oracle',False),
        scope='Raw execution, official public-input lineage, source snapshots, per-item scores and logged budgets/cost. Timings are checked for accounting consistency, not independently remeasured.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--labels',required=True,type=Path)
    p.add_argument('--verify-prompts',action='store_true')
    a=p.parse_args();result=audit(a.run,a.labels,a.verify_prompts);(a.run/'raw_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
