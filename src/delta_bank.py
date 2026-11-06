"""One immutable base model with validated, request-scoped LoRA deltas.

Storage/runtime foundation for workflow experiments, not evidence that experts
or adapter routing improve a task. No KV cache is shared across role requests.
"""
import contextlib
import hashlib
import json
import threading
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file, save_file


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(files):
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def write_delta(factors,targets,base_manifest,output,coefficient,provenance):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    expected={name+'.'+side for name in targets for side in ['A','B']}
    if set(factors)!=expected or not targets:raise ValueError('Incomplete delta factors')
    for name,spec in targets.items():
        A,B=factors[name+'.A'],factors[name+'.B']
        if spec['rank']<=0 or tuple(A.shape)!=(spec['rank'],spec['in_features']):raise ValueError('Invalid A')
        if tuple(B.shape)!=(spec['out_features'],spec['rank']):raise ValueError('Invalid B')
        if not torch.isfinite(A).all() or not torch.isfinite(B).all():raise ValueError('Nonfinite factor')
    if not torch.isfinite(torch.tensor(coefficient)):raise ValueError('Invalid coefficient')
    output.mkdir(parents=True)
    save_file({k:v.detach().float().cpu().contiguous() for k,v in factors.items()},str(output/'adapter.safetensors'))
    meta=dict(version=1,kind='functional_lora',base_id=fingerprint(base_manifest['files']),
        base_source=base_manifest['source'],base_files=base_manifest['files'],targets=targets,
        coefficient=coefficient,tensor_sha256=sha(output/'adapter.safetensors'),provenance=provenance,
        semantics='unmerged W*x + coefficient*(alpha/r)*B*A*x; BF16 merge need not be bitwise equivalent')
    (output/'manifest.json').write_text(json.dumps(meta,indent=2))
    return meta


def export_legacy_adapter(adapter_path, base_manifest_path, output, alpha=16, coefficient=1.):
    """Convert existing pilot factors without copying a full model checkpoint."""
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=json.loads(Path(base_manifest_path).read_text())
    old=torch.load(adapter_path,map_location='cpu',weights_only=True)
    factors={}
    targets={}
    for name,A in old.items():
        if not name.endswith('.A'):continue
        layer,module,_=name.split('.')
        B=old[f'{layer}.{module}.B']
        path=f'model.layers.{layer}.self_attn.{module}'
        if A.ndim!=2 or B.ndim!=2 or A.shape[0]!=B.shape[1]:raise ValueError('Invalid factor shape')
        factors[path+'.A']=A.float().contiguous()
        factors[path+'.B']=B.float().contiguous()
        targets[path]=dict(in_features=A.shape[1],out_features=B.shape[0],rank=A.shape[0],alpha=alpha)
    if len(factors)!=len(old):raise ValueError('Unknown/extra adapter keys')
    if not targets:raise ValueError('Empty adapter')
    output.mkdir(parents=True)
    save_file(factors,str(output/'adapter.safetensors'))
    meta=dict(version=1,kind='functional_lora',base_id=fingerprint(manifest['files']),
        base_source=manifest['source'],base_files=manifest['files'],targets=targets,
        coefficient=coefficient,tensor_sha256=sha(output/'adapter.safetensors'),
        provenance=dict(method='LoRA',paper='https://arxiv.org/abs/2106.09685',
            input_adapter_sha256=sha(adapter_path),scope='converted synthetic pilot, not a new workflow expert'),
        semantics='unmerged W*x + coefficient*(alpha/r)*B*A*x; BF16 merge need not be bitwise equivalent')
    (output/'manifest.json').write_text(json.dumps(meta,indent=2))
    return meta


class DeltaBank:
    def __init__(self, model, base_id):
        self.model=model
        self.base_id=base_id
        self.roles={}
        self._active=None
        self._lock=threading.RLock()

    @classmethod
    def from_verified_base(cls, model, base_manifest_path):
        meta=json.loads(Path(base_manifest_path).read_text())
        base=Path(meta['source'])
        if Path(model.config._name_or_path).resolve()!=base.resolve():
            raise ValueError('Loaded model is not from the declared base directory')
        if not all(sha(base/name)==value for name,value in meta['files'].items()):
            raise ValueError('Base file fingerprint mismatch')
        # Model loading must use this verified directory; tensor semantics are
        # separately checked by the caller against the materialized reference.
        return cls(model,fingerprint(meta['files']))

    def register(self, role, directory, coefficient_override=None):
        with self._lock:
            if self._active is not None:raise RuntimeError('Cannot register during an active request')
            if role in self.roles:raise ValueError('Duplicate role')
            path=Path(directory)
            meta=json.loads((path/'manifest.json').read_text())
            if coefficient_override is not None:
                if not torch.isfinite(torch.tensor(coefficient_override)):raise ValueError('Invalid coefficient override')
                meta['coefficient']=coefficient_override
            if (meta['version']!=1 or meta['kind']!='functional_lora' or meta['base_id']!=self.base_id
                or fingerprint(meta['base_files'])!=meta['base_id'] or not meta['targets']):
                raise ValueError('Wrong base or delta type')
            if sha(path/'adapter.safetensors')!=meta['tensor_sha256']:
                raise ValueError('Corrupt delta file')
            data=load_file(str(path/'adapter.safetensors'),device='cpu')
            modules=dict(self.model.named_modules())
            expected=set()
            for name,spec in meta['targets'].items():
                module=modules.get(name)
                if not isinstance(module,torch.nn.Linear):raise ValueError('Unsupported or missing target '+name)
                if (spec['rank']<=0 or spec['in_features']!=module.in_features
                    or spec['out_features']!=module.out_features):raise ValueError('Declared target shape mismatch')
                A,B=data[name+'.A'],data[name+'.B']
                if tuple(A.shape)!=(spec['rank'],module.in_features):raise ValueError('Input/rank shape mismatch')
                if tuple(B.shape)!=(module.out_features,spec['rank']):raise ValueError('Output/rank shape mismatch')
                if not torch.isfinite(A).all() or not torch.isfinite(B).all():raise ValueError('Nonfinite delta')
                if not (torch.isfinite(torch.tensor(meta['coefficient'])) and torch.isfinite(torch.tensor(spec['alpha']))):
                    raise ValueError('Nonfinite scale')
                expected.update([name+'.A',name+'.B'])
            if expected!=set(data):raise ValueError('Unexpected factor keys')
            self.roles[role]=(meta,data)

    @contextlib.contextmanager
    def activate(self, role, module_coefficients=None):
        """Serialize role requests; hooks are removed even after an exception."""
        with self._lock:
            if self._active is not None:raise RuntimeError('Nested role activation is unsupported')
            if role not in self.roles:raise KeyError(role)
            meta,data=self.roles[role]
            module_coefficients={} if module_coefficients is None else module_coefficients
            if set(module_coefficients)-set(meta['targets']):raise ValueError('Unknown module coefficient')
            if any(not torch.isfinite(torch.tensor(v)) for v in module_coefficients.values()):
                raise ValueError('Nonfinite module coefficient')
            handles=[]
            self._active=role
            try:
                modules=dict(self.model.named_modules())
                for name,spec in meta['targets'].items():
                    module=modules[name]
                    scale=meta['coefficient']*module_coefficients.get(name,1.)*spec['alpha']/spec['rank']
                    if scale==0.:continue
                    A=data[name+'.A'].to(module.weight.device)
                    B=data[name+'.B'].to(module.weight.device)
                    def correction(module,inputs,output,A=A,B=B,scale=scale):
                        return output+(F.linear(F.linear(inputs[0].float(),A),B)*scale).to(output.dtype)
                    handles.append(module.register_forward_hook(correction))
                yield self.model
            finally:
                for handle in handles:handle.remove()
                self._active=None

    @torch.inference_mode()
    def generate(self, role, **kwargs):
        if kwargs.get('past_key_values') is not None:
            raise ValueError('Cross-role KV reuse is not permitted')
        with self.activate(role):
            return self.model.generate(**kwargs)

    def storage(self):
        return dict(base_id=self.base_id,roles=len(self.roles),
            factor_bytes=sum(v.numel()*v.element_size() for _,data in self.roles.values() for v in data.values()),
            policy='CPU factor banks; active factors copied to device per role; no full checkpoint copies')
