"""Host-owned final math evaluation, separate from the programmer tool service.

No evaluation labels or results are returned to a ProgrammingSession. Raw sample
records stream to disk; repeated samples are summarized at the problem level.
"""
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time
from .forward_service import INSTRUCTION
from .grading import loop_diagnostics


def evaluate_math(backend,store,rows,output,policy,grader,*,samples=4,case_context=None,edited=False):
    output=Path(output)
    if output.exists():raise FileExistsError('Fresh final evaluation required; inspect partial runs explicitly')
    if type(samples)!=int or samples<1:raise ValueError('Positive samples per problem required')
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Unique final-evaluation problem IDs required')
    for row in rows:
        if row.get('data_role')!='evaluation' or row.get('source_split')!='test':raise ValueError('Explicit test/evaluation provenance required')
        if any(not isinstance(row.get(k),str) or not row[k].strip() for k in ['id','problem','answer']):raise ValueError('Canonical evaluation fields required')
    output.mkdir(parents=True);snapshot=store.snapshot(with_edits=edited)
    protocol=dict(base_id=snapshot['base_id'],variant_id=snapshot['variant_id'],native_tensor_sha256=snapshot['native_tensor_sha256'],
        edited=edited,policy=asdict(policy),samples_per_problem=samples,ids=[r['id'] for r in rows],
        dataset_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),backend=backend.identity,grader=grader.identity,
        scope='Final host evaluation. No search feedback. Accuracy averages samples, not pass@k.')
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2));summary=[];started=time.perf_counter()
    request=dict(max_tokens=policy.max_new_tokens,activations=dict(layers=[],mlp=False))
    try:
        with (output/'samples.jsonl').open('x') as stream:
            for position,row in enumerate(rows):
                # Reference solution/answer never enters the generation prompt.
                item=dict(id=row['id'],source=row.get('source','math'),prompt=row['problem']+'\n'+INSTRUCTION)
                prepared=backend.prepare_generation([item],request,policy)[0];records=[]
                with (case_context(position) if case_context is not None else nullcontext()):
                    with backend.activate(snapshot):
                        for sample in range(samples):
                            record=backend.generate(prepared,sample,request,policy,output,f'item_{position:04d}_sample_{sample}')
                            record.update(grader(row,record['text'],record['finish_reason']))
                            if type(record.get('correct'))!=bool:raise ValueError('Every final sample must have a boolean grade')
                            record['loop_diagnostics']=loop_diagnostics(record['text'])
                            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                            records.append({k:record[k] for k in ['correct','output_tokens','hit_max_tokens','generation_seconds']})
                            records[-1]['periodic_tail']=record['loop_diagnostics']['verbatim_periodic_tail']
                item_result=dict(id=row['id'],samples=samples,accuracy=statistics.mean(r['correct'] for r in records),
                    correct_samples=sum(r['correct'] for r in records),output_tokens=sum(r['output_tokens'] for r in records),
                    cutoffs=sum(r['hit_max_tokens'] for r in records),periodic_tails=sum(r['periodic_tail'] for r in records),
                    generation_seconds=sum(r['generation_seconds'] for r in records))
                summary.append(item_result)
                with (output/'items.jsonl').open('a') as items:items.write(json.dumps(item_result)+'\n')
                print(json.dumps(dict(event='final_evaluation_item',position=position+1,items=len(rows),**item_result)),flush=True)
        accuracy=statistics.mean(r['accuracy'] for r in summary);total=len(rows)*samples
        result=dict(status='final_math_evaluation_complete',problems=len(rows),samples=total,accuracy=accuracy,
            problem_cluster_standard_error=statistics.stdev(r['accuracy'] for r in summary)/len(rows)**.5 if len(rows)>1 else None,
            cutoff_share=sum(r['cutoffs'] for r in summary)/total,periodic_tail_share=sum(r['periodic_tails'] for r in summary)/total,
            mean_output_tokens=sum(r['output_tokens'] for r in summary)/total,
            generation_seconds=sum(r['generation_seconds'] for r in summary),wall_seconds=time.perf_counter()-started,
            variant_id=snapshot['variant_id'],scope=protocol['scope'])
        (output/'complete.json').write_text(json.dumps(result,indent=2));return result
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,message=str(error),completed_problems=len(summary)),indent=2))
        raise
