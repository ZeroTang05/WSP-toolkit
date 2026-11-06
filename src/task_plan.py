"""Declared layerwise adaptation of trained task-vector arithmetic.

Task Arithmetic (Ilharco et al., arXiv:2212.04089) supplies trained parameter
directions. Four normalized-depth coefficients are an explicit search-space
adaptation here, not a reproduction of a novel optimizer from that paper.
"""
import math
import re


METHOD='taskvector.depth_scaled_lora.v1'


def compile_coefficients(targets,n_layers,coordinates):
    if n_layers<1 or len(coordinates)!=4 or any(not math.isfinite(float(x)) or not 0<=x<=1.5 for x in coordinates):
        raise ValueError('Expected four finite depth coefficients in [0,1.5]')
    result={}
    for name in targets:
        found=re.fullmatch(r'model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)',name)
        if not found or int(found[1])>=n_layers:raise ValueError('Unsupported task-vector target')
        group=min(3,4*int(found[1])//n_layers)
        result[name]=float(coordinates[group])
    return result


def validate_plan(plan,delta,n_layers):
    if (plan['method_id']!=METHOD or plan['base_id']!=delta['base_id'] or
        plan['delta_sha256']!=delta['tensor_sha256'] or plan['n_layers']!=n_layers):
        raise ValueError('Task plan belongs to a different base, delta, or compiler')
    return compile_coefficients(delta['targets'],n_layers,plan['coordinates'])
