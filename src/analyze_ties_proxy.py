"""Paired, descriptive analysis of the frozen five-condition CPU proxy panel."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    complete=json.loads((a.run/'complete.json').read_text())
    if complete['status']!='cpu_fp32_proxy_panel_complete' or complete['conditions']!=5:raise ValueError('Complete five-condition panel required')
    protocol=json.loads((a.run/'protocol.json').read_text())
    data=[json.loads(line) for line in (a.run/'results.jsonl').read_text().splitlines()]
    expected={'base','ordinary_mean','ties','ties_mean_norm','mean_ties_norm'}
    if len(data)!=5 or {r['condition'] for r in data}!=expected:raise ValueError('Wrong condition coverage')
    indexed={r['condition']:r for r in data};summary=[]
    for row in data:
        if [r['id'] for r in row['domain']]!=protocol['development_ids'] or [r['id'] for r in row['guard']]!=protocol['guard_ids']:raise ValueError('Different item alignment')
        nll=sum(r['nll'] for r in row['domain'])/(math.log(2)*sum(r['answer_bytes'] for r in row['domain']))
        ce=sum(r['norm_ce'] for r in row['guard'])/len(row['guard'])
        if abs(nll-row['domain_bits_per_answer_byte'])>1e-10 or abs(ce-row['guard_norm_ce'])>1e-10:raise ValueError('Wrong aggregate')
        summary.append({k:v for k,v in row.items() if k not in ['domain','guard']})
    rng=np.random.default_rng(401);n=len(protocol['development_ids']);g=len(protocol['guard_ids'])
    di=rng.integers(0,n,size=(10000,n));gi=rng.integers(0,g,size=(10000,g));contrasts=[]
    for first,second in [('ties','ordinary_mean'),('ties_mean_norm','ordinary_mean'),('mean_ties_norm','ordinary_mean'),('ties','mean_ties_norm'),('ordinary_mean','base')]:
        aa=indexed[first];bb=indexed[second]
        ab=np.array([r['answer_bytes'] for r in aa['domain']]);bbytes=np.array([r['answer_bytes'] for r in bb['domain']])
        if not np.array_equal(ab,bbytes) or np.any(ab<=0):raise ValueError('Different answer lengths')
        delta=np.array([r['nll']-s['nll'] for r,s in zip(aa['domain'],bb['domain'])])/math.log(2)
        samples=delta[di].sum(1)/ab[di].sum(1)
        guards=np.array([r['norm_ce']-s['norm_ce'] for r,s in zip(aa['guard'],bb['guard'])])
        contrasts.append(dict(first=first,second=second,domain_delta=float(delta.sum()/ab.sum()),
            domain_nominal95=np.quantile(samples,[.025,.975]).tolist(),domain_improved_items=int((delta<0).sum()),domain_worsened_items=int((delta>0).sum()),
            guard_ce_delta=float(guards.mean()),guard_nominal95=np.quantile(guards[gi].mean(1),[.025,.975]).tolist()))
    result=dict(status='paired_development_proxy_analysis',source_sha256=hashlib.sha256((a.run/'results.jsonl').read_bytes()).hexdigest(),
        conditions=summary,contrasts=contrasts,development_items=n,guard_items=g,
        uncertainty='Paired item bootstrap, 10000 replicates, seed401. Nominal intervals, no multiplicity correction or training-seed uncertainty; development results are exploratory.',
        interpretation='Negative domain/CE deltas improve these proxies. No generation-accuracy or native-BF16 equivalence claim.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
