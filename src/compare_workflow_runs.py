"""Compare actual shared/replicated or PEFT runs without assuming equivalence."""
import argparse
import json
from pathlib import Path
from audit_finqa_run import audit


def compare(left,right,labels):
    audits=[audit(path,labels) for path in [left,right]]
    protocols=[json.loads((path/'protocol.json').read_text()) for path in [left,right]]
    for field in ['model','inputs_sha256','roles','coefficients','retry','budgets','n']:
        if protocols[0].get(field)!=protocols[1].get(field):raise ValueError('Not a matched comparison: '+field)
    def rows(path):
        with open(path/'predictions.jsonl') as stream:return {r['id']:r for r in map(json.loads,stream)}
    first,second=rows(left),rows(right)
    if set(first)!=set(second):raise ValueError('Different evaluation samples')
    differences=[]
    for key in first:
        l,r=first[key],second[key]
        lc=[(call.get('prompt_hash'),call['text']) for call in l['calls']]
        rc=[(call.get('prompt_hash'),call['text']) for call in r['calls']]
        if lc!=rc or l['program']!=r['program']:
            differences.append(dict(id=key,first_different_stage=next((i for i in range(min(len(lc),len(rc))) if lc[i]!=rc[i]),min(len(lc),len(rc)))))
    complete=[json.loads((path/'complete.json').read_text()) for path in [left,right]]
    costs=[]
    for item in complete:
        costs.append(dict(peak_allocated_bytes=item['peak_allocated_bytes'],
            generation_seconds=item['generation_seconds'],generated_tokens=item['generated_tokens'],
            logical_dense_parameter_bytes=item['base_tensor_bytes']*item['resident_base_copies'],
            delta_file_bytes=item['delta_file_bytes'],resident_base_copies=item['resident_base_copies']))
    return dict(status='matched_comparison_audited',n=len(first),exact_prompt_and_output_match=not differences,
        differences=differences,audits=audits,costs=costs,
        scope='Single-run descriptive memory/time comparison. Logical parameter bytes are distinct from physical disk storage; repeated randomized runs are needed for performance claims. Exact matching here covers only the evaluated inputs.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    value=compare(a.left,a.right,a.labels);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(value,indent=2));print(json.dumps({k:v for k,v in value.items() if k!='audits'},indent=2))
