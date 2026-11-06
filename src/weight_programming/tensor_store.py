"""Direct tensor read/write primitives motivated by MIT WSP Experiment 8.

Source contract: RikiyaT/lm_understand-logs, exp8_math/runs/p50c/turns/t001/prompt.txt.
This is an independent implementation, not the unpublished author harness.
It never loads an executable model, changes a source checkpoint, or trains.
API checks do not constitute a sandbox against arbitrary Python execution.
"""
from dataclasses import asdict,dataclass
import hashlib
import json
from pathlib import Path
import threading
import torch
from safetensors import safe_open
from safetensors.torch import load_file,save_file


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class EditPolicy:
    protected_names:tuple=(
        'model.language_model.norm.weight','model.language_model.embed_tokens.weight','lm_head.weight',
        'model.norm.weight','model.embed_tokens.weight')
    forbid_global_rescaling:bool=True
    rescaling_relative_tolerance:float=1e-6
    allowed_prefixes:tuple=()


class WeightStore:
    def __init__(self,model_dir,edit_dir,policy=None):
        self.model_dir=Path(model_dir).resolve();self.edit_dir=Path(edit_dir).resolve();self.policy=policy or EditPolicy()
        if self.model_dir==self.edit_dir or self.model_dir in self.edit_dir.parents:raise ValueError('Edit storage must be outside the source checkpoint')
        if not 0<=self.policy.rescaling_relative_tolerance<1:raise ValueError('Invalid rescaling tolerance')
        index=self.model_dir/'model.safetensors.index.json'
        if index.exists():
            declared=json.loads(index.read_text())['weight_map'];files=sorted(set(declared.values()))
        else:
            declared=None;files=[p.name for p in sorted(self.model_dir.glob('*.safetensors'))]
        if not files:raise FileNotFoundError('No source safetensors')
        self._locations={}
        for filename in files:
            path=(self.model_dir/filename).resolve()
            if path.parent!=self.model_dir:raise ValueError('Unsupported external/nested source shard')
            with safe_open(str(path),framework='pt',device='cpu') as stream:
                for name in stream.keys():
                    if name in self._locations:raise ValueError('Duplicate source tensor')
                    self._locations[name]=path
        if declared is not None and (set(declared)!=set(self._locations) or any(self._locations[n].name!=f for n,f in declared.items())):raise ValueError('Shard index mismatch')
        self.names=tuple(sorted(self._locations));fingerprints={f:sha(self.model_dir/f) for f in files}
        self._source_stats={str(self.model_dir/f):((self.model_dir/f).stat().st_size,(self.model_dir/f).stat().st_mtime_ns) for f in files}
        for filename in ['config.json','generation_config.json','tokenizer_config.json','chat_template.jinja',index.name]:
            if (self.model_dir/filename).exists():fingerprints[filename]=sha(self.model_dir/filename)
        base_id=hashlib.sha256(json.dumps(fingerprints,sort_keys=True).encode()).hexdigest()
        policy_record=asdict(self.policy)
        for key in ['protected_names','allowed_prefixes']:policy_record[key]=list(policy_record[key])
        header=dict(version=1,base_id=base_id,source_files=fingerprints,policy=policy_record,
            semantics='FP32 tensor proposals, evaluated only after explicit native-dtype materialization; source immutable')
        self.edit_dir.mkdir(parents=True,exist_ok=True);self._lock=threading.RLock();self._path=self.edit_dir/'manifest.json'
        if self._path.exists():
            self._manifest=json.loads(self._path.read_text())
            if any(self._manifest.get(k)!=v for k,v in header.items()):raise ValueError('Different base checkpoint or edit policy')
        else:
            self._manifest=dict(**header,edits={},history=[]);self._commit()

    def _commit(self):
        temporary=self._path.with_suffix('.json.tmp');temporary.write_text(json.dumps(self._manifest,indent=2));temporary.replace(self._path)

    def _native(self,name):
        if name not in self._locations:raise KeyError(name)
        stat=self._locations[name].stat()
        if (stat.st_size,stat.st_mtime_ns)!=self._source_stats[str(self._locations[name])]:raise ValueError('Source checkpoint changed since opening')
        with safe_open(str(self._locations[name]),framework='pt',device='cpu') as stream:return stream.get_tensor(name)

    @torch.no_grad()
    def load_original(self,name):return self._native(name).float().clone()

    @torch.no_grad()
    def load(self,name):
        with self._lock:
            if name not in self._locations:raise KeyError(name)
            record=self._manifest['edits'].get(name)
            if record is None:return self.load_original(name)
            path=self.edit_dir/record['file']
            if sha(path)!=record['sha256']:raise ValueError('Corrupt edited tensor')
            return load_file(str(path),device='cpu')['value'].clone()

    def _validate_edit(self,name,tensor):
        if name in self.policy.protected_names:raise ValueError('Protected source tensor')
        if self.policy.allowed_prefixes and not name.startswith(self.policy.allowed_prefixes):raise ValueError('Outside allowed model scope')
        if not isinstance(tensor,torch.Tensor) or tensor.device.type!='cpu' or not tensor.is_floating_point():raise ValueError('Finite CPU floating tensor required')
        if tensor.requires_grad or tensor.grad_fn is not None:raise ValueError('Training-derived tensors are outside this API permission')
        native=self._native(name)
        if tuple(tensor.shape)!=tuple(native.shape) or not native.is_floating_point():raise ValueError('Source shape/type mismatch')
        proposal=tensor.detach().float().contiguous()
        if not torch.isfinite(proposal).all():raise ValueError('Nonfinite edit')
        original=native.float();delta=proposal.double()-original.double();base_norm=original.double().norm();relative=float(delta.norm()/base_norm) if base_norm else None
        if self.policy.forbid_global_rescaling and base_norm and not torch.equal(proposal,original):
            scale=(original.double()*proposal.double()).sum()/base_norm.square()
            residual=(proposal.double()-scale*original.double()).norm()
            if residual<=self.policy.rescaling_relative_tolerance*float(delta.norm()):raise ValueError('Whole-tensor scalar rescaling is forbidden by this policy')
        if not torch.isfinite(proposal.to(native.dtype)).all():raise ValueError('Edit overflows source precision')
        return proposal,native,relative

    @torch.no_grad()
    def save_edit(self,name,tensor):
        with self._lock:
            proposal,native,relative=self._validate_edit(name,tensor);original=native.float();evaluated=proposal.to(native.dtype)
            # Content-addressed proposals make earlier committed manifest entries stable.
            temporary=self.edit_dir/'proposal.tmp.safetensors';save_file({'value':proposal},str(temporary));digest=sha(temporary)
            filename=digest+'.safetensors';temporary.replace(self.edit_dir/filename)
            record=dict(file=filename,sha256=digest,shape=list(native.shape),native_dtype=str(native.dtype),
                changed_float32=int((proposal!=original).sum()),changed_native_dtype=int((evaluated!=native).sum()),
                relative_frobenius_change=relative,proposal_dtype='torch.float32')
            self._manifest['edits'][name]=record;self._manifest['history'].append(dict(action='save_edit',name=name,sha256=digest))
            self._commit();return dict(record)

    def drop_edit(self,name):
        with self._lock:
            if name not in self._locations:raise KeyError(name)
            self._manifest['edits'].pop(name,None);self._manifest['history'].append(dict(action='drop_edit',name=name));self._commit()

    def edited(self):return tuple(sorted(self._manifest['edits']))

    @torch.no_grad()
    def snapshot(self,with_edits=True):
        """Read committed edits from another process and validate their payloads.

        Values are immutable for a forward request even if the notebook writes
        its next proposal while that request is running. Not a sandbox boundary.
        """
        with self._lock:
            current=json.loads(self._path.read_text())
            for key in ['version','base_id','source_files','policy','semantics']:
                if current.get(key)!=self._manifest[key]:raise ValueError('Changed base/policy binding')
            values={};native_hashes={}
            for name,record in (current['edits'] if with_edits else {}).items():
                path=(self.edit_dir/record['file']).resolve()
                if path.parent!=self.edit_dir or sha(path)!=record['sha256']:raise ValueError('Corrupt edit snapshot')
                payload=load_file(str(path),device='cpu')
                if set(payload)!={'value'}:raise ValueError('Unexpected edit payload')
                proposal,native,_=self._validate_edit(name,payload['value']);values[name]=proposal
                bits=proposal.to(native.dtype).contiguous().view(torch.uint8).numpy().tobytes()
                native_hashes[name]=hashlib.sha256(bits).hexdigest()
            identity=dict(base_id=current['base_id'],native_tensor_sha256=native_hashes)
            return dict(**identity,variant_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest(),values=values)

    def kernel_bindings(self,tokenizer=None):
        bindings=dict(NAMES=self.names,load=self.load,load_original=self.load_original,
            save_edit=self.save_edit,drop_edit=self.drop_edit,edited=self.edited)
        if tokenizer is not None:bindings['tok']=tokenizer
        return bindings

    @torch.no_grad()
    def materialize_edits(self,output):
        output=Path(output)
        resolved=output.resolve()
        if resolved==self.model_dir or self.model_dir in resolved.parents:raise ValueError('Cannot export into the source checkpoint')
        if output.exists():raise FileExistsError(output)
        output.mkdir(parents=True);records={}
        for name in self.edited():
            native=self._native(name);value=self.load(name).to(native.dtype).contiguous()
            filename=hashlib.sha256(name.encode()).hexdigest()+'.safetensors';save_file({name:value},str(output/filename))
            records[name]=dict(file=filename,sha256=sha(output/filename),dtype=str(value.dtype),shape=list(value.shape))
        (output/'manifest.json').write_text(json.dumps(dict(base_id=self._manifest['base_id'],tensors=records,
            kind='replacement_tensors_not_a_complete_model',proposal_manifest_sha256=sha(self._path)),indent=2))
        return records
