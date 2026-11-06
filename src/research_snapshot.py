"""CPU evidence inventory; counts never imply the paper goal is complete."""
import argparse
import datetime
import json
from pathlib import Path
from audit_search import audit
from analyze_public import read_scores


def snapshot(root):
    report=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),searches=[],budget_searches=[],public_results=[],
        application_results=[],training=[],errors=[],matrix_flags={})
    for path in (root/'benchmarks/search-v2').glob('*/seed*/*/complete.json'):
        try:report['searches'].append(dict(path=str(path.parent),**audit(path.parent)))
        except Exception as error:report['errors'].append(dict(path=str(path),error=str(error)))
    for path in (root/'benchmarks/search-budget-v1/full').glob('*/seed*/*/complete.json'):
        try:report['budget_searches'].append(dict(path=str(path.parent),**audit(path.parent)))
        except Exception as error:report['errors'].append(dict(path=str(path),error=str(error)))
    for study in ['confirm-v2','structured-v2','budget-confirm-v1']:
        for path in (root/'benchmarks'/study).rglob('complete.json'):
            try:
                complete=json.loads(path.read_text())
                if complete.get('status')!='full_complete':continue
                results={}
                for task,expected,filter_name,metric in [('idea4_gsm8k_cot_zeroshot_v1',1319,'flexible-extract','exact_match'),
                    ('idea4_arc_challenge_v1',1172,'none','acc_norm'),('idea4_hellaswag_v1',10042,'none','acc_norm')]:
                    if task not in complete['samples']:continue
                    rows=read_scores(path.parent,task,filter_name,metric)
                    if len(rows)!=expected or set(rows)!=set(range(expected)):raise ValueError('Incomplete official IDs')
                    results[task]=dict(n=len(rows),correct=int(sum(r['score'] for r in rows.values())),
                        accuracy=sum(r['score'] for r in rows.values())/len(rows))
                report['public_results'].append(dict(path=str(path.parent),results=results))
            except Exception as error:report['errors'].append(dict(path=str(path),error=str(error)))
    for study in ['finqa-v2','finqa-peft-v1','finqa-header-control-v1']:
        for path in (root/'benchmarks'/study).rglob('complete.json'):
            value=json.loads(path.read_text())
            if value.get('status') not in ['smoke_complete','full_complete'] or 'execution_accuracy' not in value:continue
            report['application_results'].append(dict(path=str(path.parent),
                **{k:value[k] for k in ['status','n','execution_accuracy','program_validity_rate','diagnostic_oracle'] if k in value}))
    train_paths=list((root/'benchmarks/optimization').rglob('complete.json'))
    train_paths+=list((root/'benchmarks/finqa-header-control-v1/training').rglob('complete.json'))
    for path in train_paths:
        value=json.loads(path.read_text())
        if value.get('status')=='trained_delta_only':
            report['training'].append(dict(path=str(path.parent),steps=value['steps'],trainable_fraction=value.get('trainable_fraction')))
    for study in ['search-v2','search-budget-v1','confirm-v2','budget-confirm-v1','structured-v2','finqa-v2','finqa-peft-v1','finqa-header-control-v1']:
        report['matrix_flags'][study]=(root/'benchmarks'/study/'matrix_complete.json').exists()
    report['scope']='Raw prediction aggregation and search selection audits; application scores inventoried but require separate raw-outcome audit. Live process state is checked through SSH jobs, not inferred from these files.'
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=snapshot(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True);temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(a.output)
    print(json.dumps(dict(searches=len(result['searches']),budget_searches=len(result['budget_searches']),public_runs=len(result['public_results']),
        training_runs=len(result['training']),application_runs=len(result['application_results']),errors=result['errors'],matrices=result['matrix_flags']),indent=2))
