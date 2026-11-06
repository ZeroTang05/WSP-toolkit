"""Posthoc output-style/complexity diagnostics for full GSM parameter programs.

Character length is not tokenizer cost or measured speed. Reference-step
buckets describe annotation style, not intrinsic difficulty or causal skill.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import random


def analyze(paths):
    cases={};summaries=[];binding=None
    filename='idea4_gsm8k_cot_zeroshot_v1.samples.jsonl'
    for name,path in paths.items():
        rows=[r for r in map(json.loads,(path/filename).read_text().splitlines()) if r['filter']=='flexible-extract']
        by_id={r['doc_id']:r for r in rows}
        if len(rows)!=1319 or set(by_id)!=set(range(1319)):raise ValueError('Expected full official partition')
        protocol=json.loads((path/'protocol.json').read_text());complete=json.loads((path/'complete.json').read_text())
        settings={k:protocol[k] for k in ['model','batch_size','max_new_tokens','smoke_limit','evaluation_seed','task_files','implementation_files']}
        key=(settings,complete['code_sha256'],tuple(by_id[i]['doc_hash'] for i in range(1319)))
        if binding is None:binding=key
        elif key!=binding:raise ValueError('Different model/data/runtime conditions')
        correct=int(sum(r['exact_match'] for r in rows))
        result=json.loads((path/'results.json').read_text())['results']['idea4_gsm8k_cot_zeroshot_v1']['exact_match,flexible-extract']
        if abs(correct/1319-result)>1e-12:raise ValueError('Score mismatch')
        texts=[r['resps'][0][0] for r in rows];lengths=sorted(map(len,texts))
        record=dict(program=name,correct=correct,n=1319,median_output_characters=statistics.median(lengths),
            p90_output_characters=lengths[int(.9*(len(lengths)-1))],arithmetic_markup_count=sum('<<' in t for t in texts),
            answer_phrase_count=sum('the answer is' in t.lower() for t in texts),
            source_sha256=hashlib.sha256((path/filename).read_bytes()).hexdigest(),reference_step_groups=[])
        for low,high,label in [(0,2,'0-2'),(3,4,'3-4'),(5,10**6,'5+')]:
            bucket=[r for r in rows if low<=r['doc']['answer'].count('<<')<=high]
            record['reference_step_groups'].append(dict(reference_steps=label,n=len(bucket),correct=int(sum(r['exact_match'] for r in bucket))))
        summaries.append(record);cases[name]=by_id
    contrasts=[]
    for program in ['greedy','bo']:
        for low,high,label in [(0,2,'0-2'),(3,4,'3-4'),(5,10**6,'5+')]:
            ids=[i for i in range(1319) if low<=cases['fixed'][i]['doc']['answer'].count('<<')<=high]
            delta=[cases[program][i]['exact_match']-cases['fixed'][i]['exact_match'] for i in ids]
            rng=random.Random(20261106);draws=sorted(sum(rng.choices(delta,k=len(delta)))/len(delta) for _ in range(2000))
            contrasts.append(dict(program=program,reference='fixed',reference_steps=label,n=len(delta),
                gain=sum(delta)/len(delta),improved=sum(d>0 for d in delta),regressed=sum(d<0 for d in delta),
                item_bootstrap95=[draws[49],draws[1949]],resamples=2000,
                scope='Posthoc subgroup, no multiplicity correction, one trained source seed.'))
    return dict(status='complete_full_partition_posthoc_diagnostics',programs=summaries,posthoc_paired_groups=contrasts,
        scope='One trained source seed. Output style and reference-step strata were not used to select programs. Conditional associations are not causal mechanisms. Characters are not generation tokens or latency.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.root/'confirmation-v2/Qwen3-0.6B';paths={m:root/d/'gsm8k' for m,d in [('base','base'),('lora','lora-seed11'),('fixed','fixed08-seed11'),('random','random-seed11')]}
    paths.update({m:a.root/'programming-expansion-v1/existing-effects'/m for m in ['greedy','bo']})
    result=analyze(paths);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
