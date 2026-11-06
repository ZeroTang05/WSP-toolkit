"""Independent bookkeeping and selection audit; not proof of model/task efficacy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from prepare_training_views import read_jsonl
from delta_bank import fingerprint


def audit(directory):
    protocol=json.loads((directory/'protocol.json').read_text())
    complete=json.loads((directory/'complete.json').read_text())
    plan=json.loads((directory/'selected_plan.json').read_text())
    records=read_jsonl(directory/'trace.jsonl')
    budget_study=protocol.get('protocol_version')=='resident-budget-v1'
    if protocol.get('protocol_version') not in ['paper-grid-v2','resident-budget-v1']:raise ValueError('Unknown search protocol')
    for name,value in protocol['code_sha256'].items():
        if hashlib.sha256((directory/'code'/name).read_bytes()).hexdigest()!=value:raise ValueError('Changed source '+name)
    if hashlib.sha256((directory/'contracts/methods.json').read_bytes()).hexdigest()!=protocol['contract_spec_sha256']:
        raise ValueError('Changed contract snapshot')
    n=len(records) if budget_study else protocol['queries']
    if len(records)!=n or complete['queries']!=n or n>protocol['queries']:raise ValueError('Incomplete trace')
    if budget_study and complete['budget_seconds']!=protocol['budget_seconds']:raise ValueError('Budget mismatch')
    seen=set();base=None;bootstrap=None;previous_active=0.
    for index,row in enumerate(records):
        point=tuple(row['coordinates'])
        if row['index']!=index or len(point)!=4 or point in seen:raise ValueError('Invalid trace order or repeated point')
        if any(not np.isfinite(x) or not 0<=x<=1 or abs(x*20-round(x*20))>1e-8 for x in point):raise ValueError('Illegal coordinate')
        seen.add(point)
        if budget_study:
            active=float(row['active_seconds']);budget=protocol['budget_seconds']
            if not np.isfinite(active) or active<previous_active:raise ValueError('Nonmonotone budget clock')
            if bool(row['within_budget'])!=(budget is None or active<=budget):raise ValueError('Incorrect deadline eligibility')
            previous_active=active
        risk=row['domain_risk']
        values=np.asarray(row['guard_ce_per_item'],dtype=float)
        if not np.isfinite(risk) or risk<0 or len(values)!=protocol['guard_n'] or not np.isfinite(values).all():
            raise ValueError('Invalid proxy risk')
        if not np.allclose(values,row['guard_diagnostics'][protocol['guard_mode']],rtol=0,atol=1e-12):
            raise ValueError('Wrong guard proxy')
        if index==0:
            if point!=(0.,)*4:raise ValueError('Missing identity anchor')
            base=values;bootstrap=np.random.default_rng(20261106).integers(0,len(values),size=(2000,len(values)))
        if index==1 and point!=(1.,)*4:raise ValueError('Missing full-LoRA anchor')
        upper=float(np.quantile((values-base)[bootstrap].mean(1),.95))
        if (not np.isclose(upper,row['guard_upper95'],rtol=0,atol=1e-12)
            or not np.isclose(values.mean(),row['guard_ce'],rtol=0,atol=1e-12)
            or row['guard_margin']!=protocol['guard_margin']
            or bool(row['feasible'])!=(upper<=protocol['guard_margin'])):raise ValueError('Guard statistic mismatch')
    eligible=[r for r in records if not budget_study or r['within_budget']]
    valid=[r for r in eligible if protocol['without_guard'] or r['feasible']]
    chosen=min(valid,key=lambda r:(r['domain_risk'],r['index'])) if valid else dict(index=None,coordinates=[0.]*4)
    if not valid and not budget_study:raise ValueError('Missing feasible identity anchor')
    if budget_study:
        if complete['eligible_queries']!=len(eligible):raise ValueError('Eligible query count mismatch')
        if bool(complete['fallback_identity_without_completed_observation'])!=(not bool(eligible)):
            raise ValueError('Default-identity accounting mismatch')
        active=float(complete['active_seconds']);budget=protocol['budget_seconds']
        if not np.isfinite(active) or active<previous_active:raise ValueError('Invalid final resident time')
        expected_overrun=max(0.,active-budget) if budget is not None else 0.
        if not np.isclose(complete['overrun_seconds'],expected_overrun,atol=1e-9,rtol=0):raise ValueError('Overrun mismatch')
    if (complete['selected_index']!=chosen['index'] or complete['coordinates']!=chosen['coordinates']
        or plan['coordinates']!=chosen['coordinates']):raise ValueError('Incorrect selected incumbent')
    if hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest()!=plan['selection_protocol_sha256']:
        raise ValueError('Plan protocol mismatch')
    base_meta=json.loads((directory/'base_manifest.json').read_text())
    if fingerprint(base_meta['files'])!=plan['base_id']:raise ValueError('Plan base identity mismatch')
    delta=Path(protocol['delta']);manifest_bytes=(delta/'manifest.json').read_bytes()
    meta=json.loads(manifest_bytes)
    if (hashlib.sha256(manifest_bytes).hexdigest()!=protocol['delta_manifest_sha256']
        or hashlib.sha256((delta/'adapter.safetensors').read_bytes()).hexdigest()!=meta['tensor_sha256']
        or meta['tensor_sha256']!=plan['delta_sha256'] or meta['base_id']!=plan['base_id']):
        raise ValueError('Plan trained-direction binding mismatch')
    return dict(status='selection_and_trace_verified',queries=n,selected_index=chosen['index'],
        coordinates=chosen['coordinates'],identity_selected=all(x==0 for x in chosen['coordinates']),
        budget_study=budget_study,
        scope='Source snapshots, candidate legality, guard bootstrap and incumbent selection only; requires independent heldout benchmark evaluation')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    result=audit(a.run);(a.run/'selection_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
