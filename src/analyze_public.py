"""Audit unique per-filter predictions and aggregate paired benchmark effects."""
import argparse
import json
import math
from pathlib import Path
from prepare_training_views import read_jsonl


def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-width,center+width]


def read_scores(directory,task,filter_name,metric):
    rows=read_jsonl(directory/(task+'.samples.jsonl'))
    chosen=[r for r in rows if r.get('filter')==filter_name]
    if not chosen:raise ValueError('Missing per-item filter')
    scores={}
    for row in chosen:
        if row['doc_id'] in scores:raise ValueError('Duplicate document/filter record')
        value=float(row[metric])
        if value not in [0.,1.]:raise ValueError('Expected binary per-item metric')
        scores[row['doc_id']]=dict(score=value,doc_hash=row['doc_hash'])
    result=json.loads((directory/'results.json').read_text())['results'][task]
    actual=sum(v['score'] for v in scores.values())/len(scores)
    if not math.isclose(actual,result[metric+','+filter_name],abs_tol=1e-12):raise ValueError('Aggregate does not match raw predictions')
    return scores


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--task',default='idea4_gsm8k_cot_zeroshot_v1')
    p.add_argument('--filter',default='flexible-extract');p.add_argument('--metric',default='exact_match')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    data=read_scores(a.run,a.task,a.filter,a.metric);n=len(data);k=int(sum(r['score'] for r in data.values()))
    report=dict(run=str(a.run),n=n,correct=k,accuracy=k/n,per_item_wilson95=wilson(k,n),
        uncertainty_scope='Item sampling only; this single-run interval excludes optimization-seed variation and training contamination uncertainty.')
    if a.reference:
        import numpy as np
        reference=read_scores(a.reference,a.task,a.filter,a.metric)
        if set(data)!=set(reference) or any(data[i]['doc_hash']!=reference[i]['doc_hash'] for i in data):
            raise ValueError('Runs do not evaluate the same documents')
        differences=np.array([data[i]['score']-reference[i]['score'] for i in sorted(data)])
        rng=np.random.default_rng(20261106)
        means=[]
        for _ in range(5000):means.append(float(rng.choice(differences,size=n,replace=True).mean()))
        report['paired_against_reference']=dict(reference=str(a.reference),gain=float(differences.mean()),
            paired_item_bootstrap95=np.quantile(means,[.025,.975]).tolist(),resamples=5000,seed=20261106,
            improved=int((differences>0).sum()),regressed=int((differences<0).sum()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
