"""ShortGPT BI-based depth programming, with explicit Qwen3/Llama adaptation."""
import contextlib
import math
import random
import torch


def block_influence(before, after):
    """返回每个词元的 ShortGPT 层影响分数。"""
    return 1 - torch.nn.functional.cosine_similarity(before.float(), after.float(), dim=-1)


def choose_layers(scores,drop,mode='bi',seed=11):
    if not 0<drop<len(scores) or not all(math.isfinite(float(x)) for x in scores):raise ValueError('Invalid layer budget/scores')
    if mode=='bi':removed=sorted(range(len(scores)),key=lambda i:(scores[i],i))[:drop]
    elif mode=='tail':removed=list(range(len(scores)-drop,len(scores)))
    elif mode=='random':removed=random.Random(seed).sample(range(len(scores)),drop)
    else:raise ValueError('Unknown layer policy')
    return [i for i in range(len(scores)) if i not in set(removed)]


@contextlib.contextmanager
def depth_variant(model,kept):
    original=model.model.layers;n=len(original)
    if model.config.model_type not in ['qwen3','llama']:raise ValueError('Unsupported decoder architecture')
    if not kept or kept!=sorted(set(kept)) or any(type(i)!=int or not 0<=i<n for i in kept):raise ValueError('Invalid original layer IDs')
    configs=list({id(c):c for c in [model.config,model.model.config]}.values())
    state=[(c,c.num_hidden_layers,getattr(c,'layer_types',None)) for c in configs]
    indices=[(layer.self_attn,layer.self_attn.layer_idx) for layer in original]
    try:
        model.model.layers=torch.nn.ModuleList([original[i] for i in kept])
        for i,layer in enumerate(model.model.layers):layer.self_attn.layer_idx=i
        for config,_,types in state:
            config.num_hidden_layers=len(kept)
            if types is not None:config.layer_types=[types[i] for i in kept]
        yield
    finally:
        model.model.layers=original
        for attention,index in indices:attention.layer_idx=index
        for config,count,types in state:
            config.num_hidden_layers=count
            if types is not None:config.layer_types=types
