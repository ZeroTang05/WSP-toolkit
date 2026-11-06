"""Portable base-plus-bitmask representation; dense execution, no speed claim."""
import json
from pathlib import Path
import numpy as np
import torch
from delta_bank import fingerprint,sha


def save_masks(masks,base,output,provenance):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if not masks:raise ValueError('Empty pruning mask')
    packed={};targets={}
    for index,(name,value) in enumerate(sorted(masks.items())):
        if value.dtype!=torch.bool or value.ndim!=2:raise ValueError('Expected 2D boolean mask')
        key='mask_'+str(index)
        packed[key]=np.packbits(value.cpu().numpy().reshape(-1),bitorder='little')
        targets[name]=dict(key=key,shape=list(value.shape),masked=int(value.sum()))
    output.mkdir(parents=True)
    np.savez(output/'masks.npz',**packed)
    meta=dict(version=1,kind='zero_weight_bitmask',base_id=fingerprint(base['files']),base_files=base['files'],
        targets=targets,tensor_sha256=sha(output/'masks.npz'),provenance=provenance,
        semantics='W_new=W_base with selected entries exactly zero; shape unchanged; dense inference')
    (output/'manifest.json').write_text(json.dumps(meta,indent=2))
    return meta


@torch.no_grad()
def apply_masks(model,directory,base_files):
    directory=Path(directory)
    meta=json.loads((directory/'manifest.json').read_text())
    actual={key:base_files.get(key) for key in meta['base_files']}
    if meta['version']!=1 or meta['kind']!='zero_weight_bitmask':raise ValueError('Unsupported mask format')
    if fingerprint(actual)!=meta['base_id'] or fingerprint(meta['base_files'])!=meta['base_id']:
        raise ValueError('Pruning mask requires a different base')
    if sha(directory/'masks.npz')!=meta['tensor_sha256']:raise ValueError('Mask payload changed')
    # Validate all entries before committing any mutation.
    checked=[]
    with np.load(directory/'masks.npz',allow_pickle=False) as archive:
        if set(archive.files)!={spec['key'] for spec in meta['targets'].values()}:raise ValueError('Mask keys mismatch')
        for name,spec in meta['targets'].items():
            module=model.get_submodule(name)
            if not isinstance(module,torch.nn.Linear) or list(module.weight.shape)!=spec['shape']:
                raise ValueError('Mask module or shape mismatch')
            n=module.weight.numel();packed=archive[spec['key']]
            if packed.dtype!=np.uint8 or packed.shape!=((n+7)//8,):raise ValueError('Invalid packed mask')
            bits=np.unpackbits(packed,bitorder='little')
            if bits[n:].any() or int(bits[:n].sum())!=spec['masked']:raise ValueError('Mask count or padding mismatch')
            checked.append((module,torch.from_numpy(bits[:n].reshape(spec['shape']).astype(bool))))
    for module,mask in checked:module.weight.masked_fill_(mask.to(module.weight.device),0.)
    return meta
