"""LoRA update geometry; parameter statistics are not causal capability evidence."""
import argparse
import json
import math
from pathlib import Path
import torch
from safetensors.torch import load_file
from delta_bank import sha


def factors(directory):
    meta=json.loads((directory/'manifest.json').read_text())
    if sha(directory/'adapter.safetensors')!=meta['tensor_sha256']:raise ValueError('Corrupt adapter')
    return meta,load_file(str(directory/'adapter.safetensors'))


def norm_squared(A,B,scale):return max(0.,float(((A@A.T)*(B.T@B).T).sum())*scale**2)


def compare(first,second):
    m1,d1=factors(first);m2,d2=factors(second)
    if m1['base_id']!=m2['base_id'] or set(m1['targets'])!=set(m2['targets']):
        raise ValueError('Parameter cosine requires identical verified coordinates')
    dot=n1=n2=0.
    for name,s1 in m1['targets'].items():
        s2=m2['targets'][name];A1,B1=d1[name+'.A'],d1[name+'.B'];A2,B2=d2[name+'.A'],d2[name+'.B']
        c1=m1['coefficient']*s1['alpha']/s1['rank'];c2=m2['coefficient']*s2['alpha']/s2['rank']
        dot+=float(((A1@A2.T)*(B1.T@B2)).sum())*c1*c2
        n1+=norm_squared(A1,B1,c1);n2+=norm_squared(A2,B2,c2)
    return dict(cosine=dot/math.sqrt(n1*n2) if n1 and n2 else None,
        first_frobenius=math.sqrt(n1),second_frobenius=math.sqrt(n2),
        scope='Same base and target coordinates only; geometric similarity does not prove matching task behavior')


def describe(directory):
    meta,data=factors(directory);records=[]
    for name,spec in meta['targets'].items():
        A,B=data[name+'.A'],data[name+'.B'];scale=meta['coefficient']*spec['alpha']/spec['rank']
        _,Ra=torch.linalg.qr(A.T,mode='reduced');_,Rb=torch.linalg.qr(B,mode='reduced')
        singular=torch.linalg.svdvals(Rb@Ra.T).abs()*abs(scale)
        energy=singular.square();total=float(energy.sum())
        rank95=int(torch.searchsorted(energy.cumsum(0),torch.tensor(.95*total)).item()+1) if total else 0
        records.append(dict(module=name,layer=int(name.split('.')[2]),nominal_rank=spec['rank'],
            energy95_rank=rank95,singular_values=singular.tolist(),frobenius=math.sqrt(total)))
    return dict(base_id=meta['base_id'],n_targets=len(records),
        stored_factor_parameters=sum(t.numel() for t in data.values()),
        factor_bytes=sum(t.numel()*t.element_size() for t in data.values()),
        total_delta_frobenius=math.sqrt(sum(r['frobenius']**2 for r in records)),modules=records,
        note='Effective update spectrum, not the rank of the full model or a compression/speedup claim')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--delta',type=Path,required=True)
    p.add_argument('--other',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);result=describe(a.delta)
    if a.other:result['comparison']=compare(a.delta,a.other)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='modules'},indent=2))
