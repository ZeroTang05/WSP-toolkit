"""Source-bound structural selectors; no full variant checkpoint required."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from delta_bank import fingerprint
from weightlab import Lab


@torch.no_grad()
def apply_structure(model,path,base_files):
    meta=json.loads(Path(path).read_text())
    actual={key:base_files.get(key) for key in meta['base_files']}
    if (meta['version']!=1 or meta['kind']!='gated_mlp_channel_selector'
        or fingerprint(actual)!=meta['base_id'] or fingerprint(meta['base_files'])!=meta['base_id']):
        raise ValueError('Structural selector does not match the base')
    keep=[torch.tensor(indices,dtype=torch.int64) for indices in meta['kept_indices']]
    # The adapter validates ALL selectors and dependencies before any slicing.
    Lab.structural_prune(SimpleNamespace(layers=model.model.layers,model=model),keep)
    return meta
