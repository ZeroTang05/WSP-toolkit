"""Paper-derived algebra on trained updates, not arithmetic on LoRA factors.

Sources: Task Arithmetic (2212.04089), Model Soups (2203.05482), and the
activation-weighted reconstruction objective of SVD-LLM (2403.07378).
Weighted delta truncation is a declared adaptation, not full SVD-LLM.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from analyze_delta import factors,describe,compare
from delta_bank import write_delta,sha

SOURCE_FILES={name:(Path(__file__).parent/name).read_bytes() for name in ['delta_programming.py','delta_bank.py','analyze_delta.py']}


def concatenate(items,weights):
    """items contain effective (A,B), so delta=B@A, with scale already in B."""
    if len(items)!=len(weights) or not items or not all(math.isfinite(float(w)) for w in weights):
        raise ValueError('Finite weight for every source required')
    if len({(A.shape[1],B.shape[0]) for A,B in items})!=1:raise ValueError('Different target dimensions')
    return torch.cat([A.double() for A,B in items]),torch.cat([float(w)*B.double() for (A,B),w in zip(items,weights)],dim=1)


def norm(A,B):
    _,ra=torch.linalg.qr(A.double().T,mode='reduced');_,rb=torch.linalg.qr(B.double(),mode='reduced')
    return float(torch.linalg.matrix_norm(rb@ra.T))


def residual(A,B,a,b):return norm(torch.cat([A.double(),a.double()]),torch.cat([B.double(),-b.double()],dim=1))


def truncate(A,B,rank):
    if not 1<=rank<=min(A.shape[0],A.shape[1],B.shape[0]):raise ValueError('Invalid retained rank')
    qa,ra=torch.linalg.qr(A.double().T,mode='reduced');qb,rb=torch.linalg.qr(B.double(),mode='reduced')
    u,s,vh=torch.linalg.svd(rb@ra.T,full_matrices=False);root=s[:rank].sqrt()
    a=(root[:,None]*vh[:rank])@qa.T;b=(qb@u[:,:rank])*root
    return a,b,dict(singular_values=s.tolist(),discarded_frobenius_squared=float(s[rank:].square().sum()))


def weighted_truncate(A,B,C,rank):
    """Minimize E||BAx-B'A'x||² for fixed C=E[(Ax)(Ax)^T].

    The leading output eigenspace of B C B^T attains the rank-constrained
    reconstruction optimum, even for singular C. No inverse or full input
    covariance is needed. This is a local calibration objective only.
    """
    if C.shape!=(A.shape[0],A.shape[0]) or not torch.isfinite(C).all():raise ValueError('Wrong latent second moment')
    C=(C.double()+C.double().T)/2;e,v=torch.linalg.eigh(C)
    if float(e.min()) < -1e-8*max(1.,float(e.abs().max())):raise ValueError('Second moment is not PSD')
    root=v*e.clamp_min(0).sqrt();u,s,_=torch.linalg.svd(B.double()@root,full_matrices=False)
    if not 1<=rank<=min(A.shape[0],B.shape[0]):raise ValueError('Invalid rank')
    b=u[:,:rank];a=b.T@B.double()@A.double()
    # Rebalance the factors without dropping any additional directions.
    a,b,_=truncate(a,b,rank)
    return a,b,dict(calibration_discarded_output_mse=float(s[rank:].square().sum()),
        calibration_total_output_energy=float(s.square().sum()),latent_second_moment_rank=int((e>e.max()*1e-10).sum()),
        objective='local source-variant activation reconstruction; no sequential recovery update')


def emit(output,base,pairs,method,details,*,method_provenance=None):
    """Save an effective low-rank update as generic factors with unit scaling."""
    targets={};data={}
    for name,(A,B) in pairs.items():
        A=A.float().contiguous();B=B.float().contiguous();rank=A.shape[0]
        targets[name]=dict(in_features=A.shape[1],out_features=B.shape[0],rank=rank,alpha=rank)
        data[name+'.A']=A;data[name+'.B']=B
    if method_provenance is None:
        method_provenance = dict(
            papers=['https://arxiv.org/abs/2212.04089','https://arxiv.org/abs/2203.05482','https://arxiv.org/abs/2403.07378'],
            implementation_grade='exact update concatenation / declared low-rank reconstruction adaptation')
    return write_delta(data, targets, base, output, 1., dict(method=method, details=details, **method_provenance))


def load_effective(directory):
    meta,data=factors(Path(directory));pairs={}
    for name,spec in meta['targets'].items():
        A=data[name+'.A'];B=data[name+'.B'];scale=meta['coefficient']*spec['alpha']/spec['rank']
        pairs[name]=(A.double(),B.double()*scale)
    return meta,pairs


def build(sources,output):
    if output.exists():raise FileExistsError(output)
    loaded=[load_effective(path) for path in sources];first=loaded[0][0]
    for meta,pairs in loaded:
        if meta['base_id']!=first['base_id'] or set(meta['targets'])!=set(first['targets']):raise ValueError('Different model coordinates')
    base=dict(source=first['base_source'],files=first['base_files']);names=list(first['targets'])
    output.mkdir(parents=True)
    (output/'code').mkdir()
    for name,data in SOURCE_FILES.items():(output/'code'/name).write_bytes(data)
    weights=[1/len(sources)]*len(sources)
    origin=[dict(path=str(path.resolve()),manifest_sha256=sha(path/'manifest.json'),tensor_sha256=meta['tensor_sha256'],
        training_protocol_sha256=sha(path.parent/'protocol.json')) for path,(meta,_) in zip(sources,loaded)]
    soup={name:concatenate([pairs[name] for _,pairs in loaded],weights) for name in names}
    records=[]
    emit(output/'soup'/'delta',base,soup,'uniform trained-update soup',dict(sources=origin,weights=weights))
    for rank in [8,4]:
        compressed={};evidence={}
        for name,(A,B) in soup.items():
            a,b,record=truncate(A,B,rank);compressed[name]=(a,b)
            record.update(relative_weight_error=residual(A,B,a.float(),b.float())/max(norm(A,B),1e-30))
            evidence[name]=record
        emit(output/f'frobenius-r{rank}'/'delta',base,compressed,'Frobenius truncated update SVD',dict(sources=origin,rank=rank,modules=evidence))
    soup_norm=math.sqrt(sum(norm(*pair)**2 for pair in soup.values()))
    original=loaded[0][1];single_norm=math.sqrt(sum(norm(*pair)**2 for pair in original.values()))
    scale=soup_norm/single_norm if single_norm else 0.
    emit(output/'norm-matched-seed11'/'delta',base,{n:(A,B*scale) for n,(A,B) in original.items()},
        'Task Arithmetic amplitude control',dict(reference=str(sources[0]),coefficient=scale,matched_to='uniform soup global Frobenius norm'))
    for name in ['soup','frobenius-r8','frobenius-r4','norm-matched-seed11']:
        description=describe(output/name/'delta');(output/name/'geometry.json').write_text(json.dumps(description,indent=2))
        records.append(dict(program=name,parameters=description['stored_factor_parameters'],factor_bytes=description['factor_bytes'],norm=description['total_delta_frobenius']))
    pairwise=[dict(first=str(sources[i]),second=str(sources[j]),**compare(sources[i],sources[j])) for i in range(len(sources)) for j in range(i+1,len(sources))]
    report=dict(source_directions=origin,source_descriptions=[describe(path) for path in sources],programs=records,pairwise=pairwise,
        code_sha256={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_FILES.items()},
        soup_norm=soup_norm,single_norm=single_norm,norm_matched_scale=scale,
        scope='Parameter geometry and storage only, not task efficacy. Soup sources cost multiple training runs; not an equal-training-cost comparison. Compression reduces adapter storage, not dense base parameters.')
    (output/'geometry.json').write_text(json.dumps(report,indent=2));(output/'complete.json').write_text(json.dumps(dict(status='compiled_update_programs',programs=records),indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--sources',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4);report=build(a.sources,a.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_descriptions']},indent=2))


if __name__=='__main__':main()
