"""Differential validation against archived, pinned upstream pure functions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
from weight_programming.merging import ties_merge


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=(a.reference/'merge_utils.py').read_bytes();provenance=json.loads((a.reference/'provenance.json').read_text())
    if hashlib.sha256(source).hexdigest()!=provenance['sha256']:raise ValueError('Reference source changed')
    wanted={'topk_values_mask','resolve_zero_signs','resolve_sign','disjoint_merge'}
    definitions=[node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    if {node.name for node in definitions}!=wanted:raise ValueError('Missing reference functions')
    namespace={'torch':torch};exec(compile(ast.Module(body=definitions,type_ignores=[]),'<pinned-upstream-functions>','exec'),namespace)
    generator=torch.Generator().manual_seed(8301);records=[]
    cases=[torch.randn(3,101,generator=generator),torch.randint(-3,4,(3,97),generator=generator).float(),
        torch.zeros(3,13),torch.tensor([[1.,2.,-3.,4.],[-1.,-2.,3.,-4.]])]
    for case,x in enumerate(cases):
        for density in [.2,.5,.9]:
            trimmed=namespace['topk_values_mask'](x.clone(),K=density)[0]
            signs=namespace['resolve_sign'](trimmed,'mass')
            expected=namespace['disjoint_merge'](trimmed,'dis-mean',signs)
            split=x.shape[1]//2
            actual,info=ties_merge([{'a':row[:split],'b':row[split:]} for row in x],['shared-base']*len(x),density=density)
            actual=torch.cat([actual['a'],actual['b']]);error=float((actual-expected).abs().max())
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            records.append(dict(case=case,density=density,shape=list(x.shape),max_absolute_error=error,retained_mask_counts=info['retained_mask_counts']))
    result=dict(status='exact_reference_agreement',reference=provenance,cases=records,
        excluded='Density endpoints are explicit extensions; abstain zero-sign variant intentionally differs.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],comparisons=len(records))))


if __name__=='__main__':main()
