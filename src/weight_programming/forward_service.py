"""A budgeted request boundary around actual forward execution and frozen edits.

The host owns dataset authorization, process isolation, scheduling and the model.
This Python service is not an operating-system sandbox or an HTTP deployment.
"""
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import statistics
import threading
import time
from .protocol import ForwardPolicy,BudgetLedger,validate_request
from .grading import loop_diagnostics,extract

INSTRUCTION='Please reason step by step, and put your final answer within \\boxed{}.'


class PracticeSet:
    def __init__(self,rows,role,name):
        if role not in ['train','calibration','development']:raise ValueError('Final evaluation data cannot be registered as programmer practice')
        self.rows={};self.role=role;self.name=name
        for row in rows:
            if not isinstance(row,dict) or not isinstance(row.get('id'),str) or not isinstance(row.get('problem'),str) or not row['problem'].strip():raise ValueError('Practice ID and problem required')
            if row.get('source_split')=='test' or row.get('data_role')=='evaluation':raise ValueError('Conflicting final-evaluation provenance in practice row')
            if row['id'] in self.rows:raise ValueError('Duplicate practice ID')
            self.rows[row['id']]=json.loads(json.dumps(row,allow_nan=False))
        digest=hashlib.sha256(json.dumps(self.rows,sort_keys=True).encode()).hexdigest()
        self.identity=dict(name=name,role=role,sha256=digest,items=len(rows),authorization='caller-supplied training-role registry; not a sandbox against relabeled test data')

    def generation_items(self,request):
        problems=request.get('problems');prompts=request.get('prompts');items=[]
        if problems is not None:
            for position,key in enumerate(problems):
                if key not in self.rows:raise ValueError('Unknown/unregistered practice ID')
                row=self.rows[key];prompt=prompts[position] if prompts is not None else row['problem']+'\n'+INSTRUCTION
                items.append(dict(row,prompt=prompt))
        else:
            for prompt in prompts:
                key='custom:'+hashlib.sha256(prompt.encode()).hexdigest()
                items.append(dict(id=key,prompt=prompt,source='custom',answer=None))
            if len({r['id'] for r in items})!=len(items):raise ValueError('Use n for repeated identical custom prompts')
        return items


def summarize_generations(rows):
    graded=[r for r in rows if r.get('correct') is not None]
    return dict(samples=len(rows),graded_samples=len(graded),accuracy=sum(r['correct'] for r in graded)/len(graded) if graded else None,
        mean_output_tokens=statistics.mean(r['output_tokens'] for r in rows),
        cutoff_share=sum(r['hit_max_tokens'] for r in rows)/len(rows),
        periodic_tail_share=sum(r['loop_diagnostics']['verbatim_periodic_tail'] for r in rows)/len(rows),
        failures=dict(cut_without_correct=sum(r['hit_max_tokens'] and r.get('correct') is False for r in rows),
            finished_wrong=sum(not r['hit_max_tokens'] and r.get('correct') is False and r.get('boxed',False) for r in rows),
            finished_without_box=sum(not r['hit_max_tokens'] and not r.get('boxed',False) for r in rows)),
        generation_seconds=sum(r['generation_seconds'] for r in rows),input_tokens=sum(r['input_tokens'] for r in rows),output_tokens=sum(r['output_tokens'] for r in rows))


def paired_comparison(original,edited):
    key=lambda r:(r['id'],r['sample'])
    a={key(r):r for r in original};b={key(r):r for r in edited}
    if set(a)!=set(b) or len(a)!=len(original) or len(b)!=len(edited):raise ValueError('Mismatched paired generations')
    differences={};rescued=regressed=0
    for item in a:
        if a[item]['seed']!=b[item]['seed'] or a[item]['prompt_sha256']!=b[item]['prompt_sha256']:raise ValueError('Different prompt/random seed in pair')
        if a[item].get('correct') is None or b[item].get('correct') is None:continue
        delta=int(b[item]['correct'])-int(a[item]['correct']);differences.setdefault(item[0],[]).append(delta)
        rescued+=delta>0;regressed+=delta<0
    means=[statistics.mean(values) for values in differences.values()]
    return dict(graded_problems=len(means),gain=statistics.mean(means) if means else None,
        paired_item_standard_error=statistics.stdev(means)/(len(means)**.5) if len(means)>1 else None,
        rescued_samples=rescued,regressed_samples=regressed,
        scope='Mean per-problem sample effect; standard error clusters repeated samples by problem, excludes editor-seed variation')


class ForwardService:
    def __init__(self,backend,store,practice,work_dir,policy=None,grader=None):
        self.backend=backend;self.store=store;self.practice=practice;self.work_dir=Path(work_dir);self.policy=policy or ForwardPolicy();self.grader=grader
        self.work_dir.mkdir(parents=True,exist_ok=True);self._lock=threading.RLock()
        base=store.snapshot(with_edits=False)
        sources={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ['forward_service.py','protocol.py','hf_backend.py','grading.py','tensor_store.py']}
        backend_source=Path(inspect.getfile(type(backend)))
        sources[backend_source.name]=hashlib.sha256(backend_source.read_bytes()).hexdigest()
        self.binding=dict(base_id=base['base_id'],backend=backend.identity,practice=practice.identity,
            grader=grader.identity if grader is not None else None,source_sha256=sources)
        self.ledger=BudgetLedger(self.work_dir/'budget.json',self.policy,self.binding)

    def request(self,raw):
        with self._lock:
            started=time.perf_counter();request=validate_request(raw,self.policy,self.backend.num_layers,self.backend.full_attention_layers)
            snapshot=self.store.snapshot(with_edits=request['model']!='original');original=self.store.snapshot(with_edits=False)
            if request['type']=='teacher_forced':prepared=self.backend.prepare_teacher(request,self.policy)
            else:prepared=self.backend.prepare_generation(self.practice.generation_items(request),request,self.policy)
            attempt=self.ledger.begin(request,snapshot['variant_id']);stage=self.work_dir/f'.request_{attempt:04d}.partial';output=self.work_dir/f'request_{attempt:04d}'
            created=False
            try:
                if stage.exists() or output.exists():raise FileExistsError('Existing request artifacts require investigation')
                stage.mkdir();created=True
                if request['type']=='teacher_forced':
                    selected=original if request['model']=='original' else snapshot
                    with self.backend.activate(selected):result=self.backend.teacher_forced(request,prepared,stage)
                else:
                    variants=['original','edited'] if request['model']=='both' else [request['model']];generated={};summaries={}
                    for variant in variants:
                        selected=original if variant=='original' else snapshot;rows=[]
                        with self.backend.activate(selected),open(stage/(variant+'.jsonl'),'w',buffering=1) as stream:
                            for position,item in enumerate(prepared):
                                for sample in range(request['n']):
                                    record=self.backend.generate(item,sample,request,self.policy,stage,f'{variant}_{position}_{sample}')
                                    record['model']=variant;record['variant_id']=selected['variant_id'];record['loop_diagnostics']=loop_diagnostics(record['raw_text'])
                                    if self.grader is not None:record.update(self.grader(item,record['raw_text'],record['finish_reason']))
                                    else:
                                        prediction,boxes=extract(record['raw_text'])
                                        record.update(correct=None,prediction=prediction,boxed=prediction is not None,boxes=boxes,grading='unavailable; no accuracy claimed')
                                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');rows.append(record)
                        generated[variant]=rows;summaries[variant]=summarize_generations(rows)
                    result=dict(generation_files={v:v+'.jsonl' for v in variants},summaries=summaries)
                    if request['model']=='both':result['paired']=paired_comparison(generated['original'],generated['edited'])
                result.update(request=request,variant_id=snapshot['variant_id'],original_variant_id=original['variant_id'],
                    binding=self.binding,policy=asdict(self.policy),attempt=attempt,elapsed_seconds=time.perf_counter()-started)
                (stage/'meta.json').write_text(json.dumps(result,indent=2))
                actual_bytes=sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())
                if actual_bytes>self.policy.max_return_bytes:raise ValueError('Actual artifact exceeds return quota')
                stage.replace(output);self.ledger.finish(attempt,True,dict(artifact=output.name,elapsed_seconds=result['elapsed_seconds'],artifact_bytes=actual_bytes))
                return dict(result,artifact_dir=str(output),budget=self.ledger.summary())
            except Exception as error:
                # A failed request must not expose partial uncharged observations.
                if created:shutil.rmtree(stage,ignore_errors=True)
                self.ledger.finish(attempt,False,dict(error_type=type(error).__name__,error=str(error)[:500],elapsed_seconds=time.perf_counter()-started))
                raise
