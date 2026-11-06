"""Contrast and output-direction edits, with explicit algebra and no layer defaults.

Sources: MIT WSP p50c final_report formula; direction removal also appears in
Arditi et al., https://arxiv.org/abs/2406.11717. No published edit is hard-coded.
"""
import math
import torch


def _finite_cpu(value):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or not value.is_floating_point() or not torch.isfinite(value).all():raise ValueError('Finite CPU floating tensor required')


@torch.no_grad()
def contrast_direction(source,target):
    _finite_cpu(source);_finite_cpu(target)
    if source.ndim!=2 or source.shape!=target.shape or not source.shape[0]:raise ValueError('Matched activation rows required')
    direction=(target.double()-source.double()).mean(0);length=direction.norm()
    if not length:raise ValueError('Zero contrast direction')
    return direction/length


@torch.no_grad()
def project_output_direction(weight,direction,columns,strength):
    _finite_cpu(weight);_finite_cpu(direction)
    if weight.ndim!=2 or direction.ndim!=1 or direction.numel()!=weight.shape[0]:raise ValueError('Direction must be in output coordinates')
    if not math.isfinite(float(strength)) or not direction.double().norm():raise ValueError('Finite strength and nonzero direction required')
    columns=list(columns)
    if not columns or len(set(columns))!=len(columns) or any(type(i)!=int or not 0<=i<weight.shape[1] for i in columns):raise ValueError('Unique legal selected columns required')
    u=direction.double()/direction.double().norm();result=weight.double().clone()
    result[:,columns]-=float(strength)*u[:,None]*(u@result[:,columns])[None,:]
    return result.float()


@torch.no_grad()
def compare_edits(base,first,second):
    for value in [base,first,second]:_finite_cpu(value)
    if base.ndim!=2 or base.shape!=first.shape or base.shape!=second.shape:raise ValueError('Same matrix coordinates required')
    a=first.double()-base.double();b=second.double()-base.double();na,nb=a.norm(),b.norm()
    sa=a.ne(0).any(0);sb=b.ne(0).any(0);union=int((sa|sb).sum())
    return dict(first_norm=float(na),second_norm=float(nb),signed_weight_cosine=float((a*b).sum()/(na*nb)) if na and nb else None,
        changed_column_jaccard=int((sa&sb).sum())/union if union else None,
        first_changed_columns=int(sa.sum()),second_changed_columns=int(sb.sum()),
        scope='Same base/coordinates only. Geometric agreement alone does not establish a shared behavioral mechanism.')
