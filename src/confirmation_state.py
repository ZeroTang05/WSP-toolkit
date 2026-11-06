"""Identity of executable native-base / functional-LoRA confirmation states.

Float conversion follows task_plan.compile_coefficients. This is not a
similarity cache: no rounding of nonzero coefficients or tensor distances.
"""
import hashlib
import json
import math
from pathlib import Path
from task_plan import validate_plan


def sha(path):
    value=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def canonical_key(base_files,delta_manifest_sha=None,coordinates=None):
    point=([1.]*4 if delta_manifest_sha else [0.]*4) if coordinates is None else [float(x) for x in coordinates]
    if len(point)!=4 or any(not math.isfinite(x) or not 0<=x<=1 for x in point):
        raise ValueError('Expected four finite coefficients in [0,1]')
    point=[0. if x==0 else float(x) for x in point]
    state=dict(base=base_files)
    if any(point):
        if not delta_manifest_sha:raise ValueError('Nonzero plan requires a trained delta')
        state.update(delta_manifest=delta_manifest_sha,coordinates=point)
    return hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()


def checked_plan_key(base_files,delta_path,plan,model_path):
    delta_path=Path(delta_path);raw=(delta_path/'manifest.json').read_bytes();meta=json.loads(raw)
    if sha(delta_path/'adapter.safetensors')!=meta['tensor_sha256']:raise ValueError('Trained factors changed')
    base_id=hashlib.sha256(json.dumps(base_files,sort_keys=True).encode()).hexdigest()
    if meta['base_id']!=base_id:raise ValueError('Different base from trained direction')
    layers=json.loads((Path(model_path)/'config.json').read_text())['num_hidden_layers']
    validate_plan(plan,meta,layers)
    return canonical_key(base_files,hashlib.sha256(raw).hexdigest(),plan['coordinates'])


def recorded_key(run):
    run=Path(run);protocol=json.loads((run/'protocol.json').read_text())
    if any(key in protocol for key in ['mask','structure']):raise ValueError('Unsupported state composition')
    base=json.loads((run/'model_fingerprint.json').read_text())
    if base['source']!=protocol['model']:raise ValueError('Model source mismatch')
    if 'delta' not in protocol:
        if 'plan' in protocol:raise ValueError('Plan without delta')
        return canonical_key(base['files'])
    delta=protocol['delta'];directory=Path(delta['directory'])
    if delta['coefficient_override'] is not None:raise ValueError('Global override needs a separate identity rule')
    if sha(directory/'manifest.json')!=delta['manifest_sha256'] or sha(directory/'adapter.safetensors')!=delta['factors_sha256']:
        raise ValueError('Changed confirmation direction')
    meta=json.loads((directory/'manifest.json').read_text())
    if meta['base_id']!=hashlib.sha256(json.dumps(base['files'],sort_keys=True).encode()).hexdigest():
        raise ValueError('Recorded direction/base mismatch')
    if 'plan' not in protocol:return canonical_key(base['files'],delta['manifest_sha256'])
    path=Path(protocol['plan']['file'])
    if sha(path)!=protocol['plan']['sha256']:raise ValueError('Changed confirmation plan')
    return checked_plan_key(base['files'],directory,json.loads(path.read_text()),protocol['model'])
