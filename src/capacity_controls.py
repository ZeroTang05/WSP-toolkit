"""Compute architecture-specific LoRA capacity without allocating a GPU model."""
import argparse
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open


def describe(directory,target=.002):
    config=json.loads((directory/'config.json').read_text());shapes={};files={}
    for path in directory.glob('*.safetensors'):
        with safe_open(path,framework='pt',device='cpu') as handle:
            for name in handle.keys():
                if name in shapes:raise ValueError('Duplicate checkpoint tensor')
                shapes[name]=handle.get_slice(name).get_shape();files[name]=path
    tied=config.get('tie_word_embeddings',False)
    if tied and 'lm_head.weight' in shapes:
        with safe_open(files['lm_head.weight'],framework='pt',device='cpu') as handle:a=handle.get_tensor('lm_head.weight')
        with safe_open(files['model.embed_tokens.weight'],framework='pt',device='cpu') as handle:b=handle.get_tensor('model.embed_tokens.weight')
        if not torch.equal(a,b):raise ValueError('Declared tied head does not match embedding')
    total=sum(math.prod(shape) for name,shape in shapes.items() if not(tied and name=='lm_head.weight'))
    targets={name:shape for name,shape in shapes.items() if name.endswith(('.self_attn.q_proj.weight','.self_attn.v_proj.weight'))}
    if len(targets)!=2*config['num_hidden_layers']:raise ValueError('Unexpected target module coverage')
    per_rank=sum(sum(shape) for shape in targets.values())
    ideal=target*total/per_rank
    options={max(1,math.floor(ideal)),max(1,math.ceil(ideal))}
    rank=min(options,key=lambda r:abs(r*per_rank/total-target))
    return dict(model=directory.name,base_parameters=total,target_modules=len(targets),parameters_per_rank=per_rank,
        fixed_rank8_parameters=per_rank*8,fixed_rank8_fraction=per_rank*8/total,
        relative_control=dict(target_fraction=target,rank=rank,alpha=2*rank,parameters=rank*per_rank,
            actual_fraction=rank*per_rank/total,absolute_fraction_error=rank*per_rank/total-target),
        tied_weight_storage_deduplicated=tied,
        note='Capacity design only; does not establish equal expressivity, optimization dynamics, latency or trained outcomes')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    rows=[describe(a.models_root/name) for name in ['Qwen3-0.6B','Qwen3-1.7B','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct']]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
