"""Executable method contracts. Metadata is provenance, not a proof of utility."""
from dataclasses import dataclass
import json
import hashlib
import sys
from pathlib import Path
import torch


@dataclass(frozen=True)
class ToolRequest:
    method_id:str
    observation:str
    selection:str
    mutation:str
    gradient_permission:str
    data_role:str


class ContractRegistry:
    def __init__(self,path=None):
        if path is None:
            candidates=[Path(__file__).resolve().parents[1]/'contracts/methods.json',
                        Path(sys.prefix)/'share/idea4-weightlab/contracts/methods.json']
            path=next((p for p in candidates if p.exists()),None)
            if path is None:raise FileNotFoundError('Method registry not installed; pass its path explicitly')
        self.path=Path(path).resolve()
        data=json.loads(self.path.read_text())
        self.spec_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if data['version']!=1:raise ValueError('Unsupported registry version')
        self.methods={}
        for item in data['methods']:
            if item['id'] in self.methods:raise ValueError('Duplicate method ID')
            if not item['paper'].startswith('https://arxiv.org/abs/'):raise ValueError('Missing paper provenance')
            for field in ['preconditions','postconditions','required_evidence']:
                if not item[field]:raise ValueError('Empty '+field)
            self.methods[item['id']]=item

    def validate(self,request):
        spec=self.methods[request.method_id]
        for field in ['observation','selection','mutation','gradient_permission']:
            if getattr(request,field)!=spec[field]:
                raise ValueError('Contract mismatch: '+field)
        if request.data_role not in spec['data_roles']:
            raise ValueError('Forbidden data role for this optimization tool')
        return spec

    @torch.no_grad()
    def apply_wanda_linear(self,request,module,energy,fraction):
        from paper_methods import wanda_mask
        spec=self.validate(request)
        if request.method_id!='wanda.row_unstructured.v1':raise ValueError('Wrong executable method')
        W=module.weight
        if W.ndim!=2 or not torch.isfinite(W).all():raise ValueError('Invalid weight')
        if energy.shape!=(W.shape[1],) or not torch.isfinite(energy).all() or torch.any(energy<0):
            raise ValueError('Invalid input energy')
        # Compute and validate completely before committing a mutation.
        mask=wanda_mask(W,energy,fraction)
        count=int(W.shape[1]*fraction)
        if not torch.all(mask.sum(1)==count):raise AssertionError('Per-row count failed')
        replacement=W.detach().clone()
        replacement[mask]=0
        if not torch.equal(replacement[~mask],W[~mask]):raise AssertionError('Unexpected edit')
        W.copy_(replacement)
        return dict(contract_id=spec['id'],masked=int(mask.sum()),total=mask.numel(),
            contract_spec_sha256=self.spec_sha256,contract_code_sha256=self.code_sha256,
            postconditions_checked=['unchanged_shape','nonmasked_weights_exact','floor_width_times_fraction_masked_per_row'],
            scope='local linear operation; reference conformance and task quality require separate evidence')
