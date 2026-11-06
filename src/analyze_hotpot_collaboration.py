"""Matched-item, repeated-seed collaboration effects; no specialization claims."""
import argparse
import json
from pathlib import Path
import numpy as np


def load(path):
    protocol=json.loads((path/'protocol.json').read_text());complete=json.loads((path/'complete.json').read_text())
    rows=[json.loads(line) for line in (path/'scores.jsonl').read_text().splitlines()]
    if complete['status']!='full_declared_partition_complete' or len(rows)!=protocol['n'] or len({r['id'] for r in rows})!=len(rows):raise ValueError('Incomplete scores')
    for key,value in complete['metrics'].items():
        if not np.isclose(np.mean([r[key] for r in rows]),value,rtol=0,atol=1e-12):raise ValueError('Score/summary mismatch')
    return protocol,complete,{r['id']:r for r in rows},json.loads((path/'base_manifest.json').read_text())['files']


def effect(left,right,draws=2000):
    delta=np.asarray(left)-np.asarray(right)
    rng=np.random.default_rng(20261106);means=[]
    for _ in range(draws):
        values=delta[rng.integers(len(delta),size=len(delta))].mean(0)
        means.append(float(values[rng.integers(len(values),size=len(values))].mean()))
    return dict(estimate=float(delta.mean()),seed_item_bootstrap95=np.quantile(means,[.025,.975]).tolist(),
        seed_gains=delta.mean(1).tolist(),improved_by_seed=(delta>0).sum(1).tolist(),regressed_by_seed=(delta<0).sum(1).tolist(),
        uncertainty='paired empirical seed-and-item bootstrap; only three inference seeds, no training-seed evidence')


def analyze(root,partition,budget,seeds=(11,22,33),modes=('direct','self_refine','independent_vote','collaborate','no_communication')):
    matrix={};summaries=[];binding=None
    for mode in modes:
        scores=[];costs=[]
        for seed in seeds:
            path=root/partition/f'budget{budget}'/f'seed{seed}'/mode;p,c,rows,files=load(path)
            if (p['partition'],p['budget'],p['seed'],p['mode'])!=(partition,budget,seed,mode):raise ValueError('Wrong condition')
            settings={k:p[k] for k in ['model','n','temperature','top_p','thinking','inputs_sha256','manifest_sha256','scorer_sha256','source_sha256','training_demonstration_ids','demonstrations_sha256']}
            key=(settings,files,tuple(sorted(rows)))
            if binding is None:binding=key
            elif key!=binding:raise ValueError('Unmatched data/model/execution')
            scores.append([rows[i]['em'] for i in sorted(rows)])
            costs.append(dict(seed=seed,answer_em=c['metrics']['em'],answer_f1=c['metrics']['f1'],support_f1=c['metrics']['sp_f1'],joint_f1=c['metrics']['joint_f1'],
                format_valid=c['metrics']['format_valid'],input_tokens=c['total_input_tokens'],output_tokens=c['total_output_tokens'],
                generation_seconds=c['generation_seconds'],peak_allocated_bytes=c['peak_allocated_bytes']))
        matrix[mode]=scores;summaries.append(dict(mode=mode,answer_em=float(np.mean(scores)),seeds=costs))
    comparisons=[dict(reference=mode,**effect(matrix['collaborate'],matrix[mode])) for mode in modes if mode!='collaborate']
    return dict(partition=partition,budget=budget,summaries=summaries,collaboration_effects=comparisons,
        scope='Complete declared partition. Collaboration benefit requires comparison to prespecified strong baseline; equal output caps do not equal actual compute. No multiple-comparison-adjusted discovery claim.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--partition',choices=['development','official_dev'],required=True)
    p.add_argument('--budget',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze(a.root,a.partition,a.budget);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
