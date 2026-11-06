"""TIES on effective parameter deltas in a declared shared coordinate scope.

Independent implementation of Yadav et al. (NeurIPS 2023), arXiv:2306.01708.
Reference behavior checked against prateeky2806/ties-merging revision
44e7891fc84f3de7e4caa52664cd864ca3715e91, src/utils/merge_utils.py.
Trimming is global over the supplied modules, not independently per matrix.
This consumes trained updates and is not original-WSP independent discovery.
"""
import math
import torch


@torch.no_grad()
def ties_merge(deltas,base_ids,*,density=.2,zero_sign='reference_majority'):
    """Return (dense effective delta mapping, evidence), CPU FP32.

    Reference threshold retains |x| >= kth_smallest(|x|, d-floor(d*density)).
    This can retain one extra coordinate even without ties. Endpoints are
    explicitly extended to retain none/all; upstream density=1 has kthvalue(0).
    ``reference_majority`` implements the upstream zero-sign fallback, including
    its negative selection when the global sign vote remains exactly zero.
    ``abstain`` instead leaves exact mass ties at zero; this is a declared variant.
    Sources must be effective deltas, never separately supplied LoRA factors.
    """
    if not isinstance(deltas,(tuple,list)) or not deltas:raise ValueError('Nonempty delta list required')
    if len(base_ids)!=len(deltas) or any(not isinstance(v,str) or not v for v in base_ids) or len(set(base_ids))!=1:
        raise ValueError('Every delta must declare the same nonempty base identity')
    if isinstance(density,bool) or not math.isfinite(float(density)) or not 0<=density<=1:raise ValueError('Density must be in [0,1]')
    if zero_sign not in ['reference_majority','abstain']:raise ValueError('Unknown zero-sign policy')
    if any(not isinstance(d,dict) or not d for d in deltas):raise ValueError('Nonempty named effective deltas required')
    names=sorted(deltas[0])
    if any(set(d)!=set(names) for d in deltas):raise ValueError('Different delta coordinate scopes')
    shapes={};vectors=[]
    for d in deltas:
        parts=[]
        for name in names:
            x=d[name]
            if not isinstance(name,str) or not isinstance(x,torch.Tensor) or not x.is_floating_point() or x.numel()==0:
                raise ValueError('Named nonempty floating tensors required')
            if name in shapes and tuple(x.shape)!=shapes[name]:raise ValueError('Different target shapes')
            shapes[name]=tuple(x.shape)
            x=x.detach().to(device='cpu',dtype=torch.float32).reshape(-1)
            if not torch.isfinite(x).all():raise ValueError('Nonfinite delta')
            parts.append(x)
        vectors.append(torch.cat(parts))
    x=torch.stack(vectors);del vectors
    size=x.shape[1];source_nonzero=(x!=0).sum(1).tolist();thresholds=[];retained=[]
    source_norm=[float(row.double().norm()) for row in x]
    for i in range(len(x)):
        if density==0:thresholds.append(None);x[i].zero_();retained.append(0);continue
        if density==1:thresholds.append(0.);retained.append(size);continue
        order=size-int(size*density)
        threshold=float(x[i].abs().kthvalue(order).values);keep=x[i].abs()>=threshold
        thresholds.append(threshold);retained.append(int(keep.sum()));x[i].masked_fill_(~keep,0)
    positive=(x>0).any(0);negative=(x<0).any(0)
    conflicts=positive&negative;active=positive|negative
    signs=x.sum(0).sign();zero_before=signs==0;global_vote=float(signs.sum().sign())
    if zero_sign=='reference_majority':signs[zero_before]=global_vote
    selected=[]
    for i in range(len(x)):
        if zero_sign=='reference_majority':keep=torch.where(signs>0,x[i]>0,x[i]<0)
        else:keep=(x[i].sign()==signs)&(x[i]!=0)
        selected.append(keep)
    counts=torch.stack(selected).sum(0)
    merged=torch.zeros(size,dtype=torch.float32)
    for i,keep in enumerate(selected):merged.add_(x[i]*keep)
    merged.div_(counts.clamp_min(1));result={};modules={};offset=0
    if not torch.isfinite(merged).all():raise ValueError('Nonfinite merged update')
    for name in names:
        n=math.prod(shapes[name]);sl=slice(offset,offset+n)
        result[name]=merged[sl].reshape(shapes[name]).clone()
        modules[name]=dict(parameters=n,conflict_coordinates=int(conflicts[sl].sum()),
            active_coordinates=int(active[sl].sum()),merged_nonzero=int((merged[sl]!=0).sum()))
        offset+=n
    evidence=dict(method='TIES mass-sign disjoint mean',paper='https://arxiv.org/abs/2306.01708',
        reference_revision='44e7891fc84f3de7e4caa52664cd864ca3715e91',base_id=base_ids[0],
        scope='Sorted union of explicitly supplied effective-delta modules; no other model parameters',
        dtype='CPU FP32',density=density,zero_sign=zero_sign,parameters=size,sources=len(x),
        trim_rule='upstream inclusive kth threshold; density endpoints explicitly extended',
        thresholds=thresholds,retained_mask_counts=retained,source_nonzero=source_nonzero,
        retained_nonzero_counts=(x!=0).sum(1).tolist(),source_norms=source_norm,
        conflicting_coordinates=int(conflicts.sum()),active_coordinates=int(active.sum()),
        exact_mass_ties=int((zero_before&active).sum()),global_sign_vote=global_vote,
        merged_norm=float(merged.double().norm()),merged_nonzero=int((merged!=0).sum()),modules=modules,
        scope_warning='May increase delta rank/storage. No inference acceleration or behavioral benefit is implied.')
    return result,evidence
