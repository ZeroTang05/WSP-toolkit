"""Paired activation patching, following Zhang & Nanda (ICLR 2024).

https://arxiv.org/abs/2309.16042v2: clean/corrupt/patched runs, explicit
corruption and metric choices. This intervention extension is NOT part of the
original WSP no-activation-intervention permission track. It does not edit weights.
"""
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch
from .protocol import ForwardPolicy,BudgetLedger


@contextlib.contextmanager
def replace_activation(backend,kind,layer,clean,positions,channels=None):
    """Temporarily replace selected coordinates at one site, batch size one.

    Residual 0=embedding; residual l=raw block l-1 output. MLP input means the
    down_proj input. Position alignment is explicit, never inferred from text.
    """
    if kind not in ['residual','mlp_input']:raise ValueError('Unknown activation site')
    maximum=backend.num_layers if kind=='residual' else backend.num_layers-1
    if type(layer)!=int or not 0<=layer<=maximum:raise ValueError('Invalid layer')
    width=backend.hidden_size if kind=='residual' else backend.mlp_width
    if (not isinstance(clean,torch.Tensor) or clean.ndim!=3 or clean.shape[0]!=1
            or clean.shape[-1]!=width or not torch.isfinite(clean).all()):
        raise ValueError('One finite clean activation with matching coordinates required')
    positions=list(positions)
    if not positions or len(set(positions))!=len(positions) or any(type(i)!=int or not 0<=i<clean.shape[1] for i in positions):
        raise ValueError('Unique aligned positions required')
    if channels is None:channels=list(range(width))
    else:channels=list(channels)
    if not channels or len(set(channels))!=len(channels) or any(type(i)!=int or not 0<=i<width for i in channels):
        raise ValueError('Unique legal activation channels required')
    def patched(value):
        if value.shape!=clean.shape:raise ValueError('Corrupt/clean activation shapes differ')
        output=value.clone();p=torch.tensor(positions,device=value.device);c=torch.tensor(channels,device=value.device)
        reference=clean.to(value.device,dtype=value.dtype)
        output[0,p[:,None],c[None,:]]=reference[0,p[:,None],c[None,:]]
        return output
    if kind=='mlp_input':
        module=backend.layers[layer].mlp.down_proj
        def hook(module,args):return (patched(args[0]),)+args[1:]
        handle=module.register_forward_pre_hook(hook)
    else:
        module=backend.decoder.embed_tokens if layer==0 else backend.layers[layer-1]
        def hook(module,args,out):
            if isinstance(out,tuple):return (patched(out[0]),)+out[1:]
            if isinstance(out,list):return [patched(out[0])]+out[1:]
            return patched(out)
        handle=module.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def metrics(logits,query,target,alternative):
    row=logits[query].float();normalizer=torch.logsumexp(row,dim=-1)
    return dict(logit_difference=float(row[target]-row[alternative]),
        target_probability=float(torch.exp(row[target]-normalizer)),
        target_log_probability=float(row[target]-normalizer))


def prepare_pair(backend, policy, clean_text, corrupt_text, target_token, alternative_token, query_position=None):
    """校验成对输入的词元对齐、目标分数与上下文预算。"""
    clean_ids, corrupt_ids = backend.encode(clean_text), backend.encode(corrupt_text)
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError("Paired same-token-length corruption required; no guessed alignment")
    if clean_ids == corrupt_ids:
        raise ValueError("Corruption must change token IDs")
    length = len(clean_ids)
    if length > min(policy.max_text_tokens, backend.context_length):
        raise ValueError("Token/context budget exceeded")
    if any(type(token) is not int or not 0 <= token < backend.vocab_size
           for token in (target_token, alternative_token)) or target_token == alternative_token:
        raise ValueError("Distinct valid target and alternative token IDs required")
    query = length - 1 if query_position is None else query_position
    if type(query) is not int or not 0 <= query < length:
        raise ValueError("Explicit legal query position required")
    return clean_ids, corrupt_ids, query


@torch.inference_mode()
def paired_interventions(backend, snapshot, pair, sites, target, alternative, read):
    """一次计算干净和受扰基线，再逐个临时改变站点。"""
    clean_ids, corrupt_ids, query = pair
    hidden = sorted({site["layer"] for site in sites if site["kind"] == "residual"})
    mlp = sorted({site["layer"] for site in sites if site["kind"] == "mlp_input"})
    with backend.activate(snapshot):
        logits, captured = read(clean_ids, hidden, mlp, [], True)
        clean = metrics(logits, query, target, alternative)
        del logits
        logits, _ = read(corrupt_ids, [], [], [], True)
        corrupt = metrics(logits, query, target, alternative)
        del logits
        outcomes = []
        for site in sites:
            kind, layer = site["kind"], site["layer"]
            reference = captured[f"{'hidden' if kind == 'residual' else 'mlp'}_{layer}"]
            if site.get("method") == "ablation":
                reference = torch.zeros_like(reference)
            with replace_activation(backend, kind, layer, reference, site["positions"], site.get("channels")):
                logits, _ = read(corrupt_ids, [], [], [], True)
                patched = metrics(logits, query, target, alternative)
                del logits
            effects = {}
            for name in clean:
                gap, effect = clean[name] - corrupt[name], patched[name] - corrupt[name]
                effects[name] = dict(clean_corrupt_gap=gap, patched_minus_corrupt=effect,
                                     normalized_gap_change=effect / gap if abs(gap) > 1e-8 else None,
                                     restoration_interpretation_valid=gap > 1e-8)
            outcomes.append(dict(clean=clean, corrupt=corrupt, patched=patched, effects=effects))
    return outcomes


class ActivationPatchingStudy:
    def __init__(self,backend,store,output,*,data_role,corruption_description,policy=None):
        if data_role not in ['train','calibration','development']:raise ValueError('Use authorized construction data only')
        if not isinstance(corruption_description,str) or not corruption_description.strip():raise ValueError('State the corruption assumption')
        self.backend=backend;self.store=store;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.policy=policy or ForwardPolicy();self.corruption=corruption_description
        self.binding=dict(base_id=store.snapshot(False)['base_id'],backend=backend.identity,data_role=data_role,
            corruption_description=corruption_description,track='paired_intervention_extension_not_original_WSP',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            backend_source_sha256=hashlib.sha256(Path(inspect.getfile(type(backend))).read_bytes()).hexdigest())
        self.ledger=BudgetLedger(self.output/'budget.json',self.policy,self.binding)

    @torch.inference_mode()
    def run(self,*,clean_text,corrupt_text,kind,layer,positions,target_token,alternative_token,
            channels=None,query_position=None,model='original'):
        if model not in ['original','edited']:raise ValueError('One explicit model variant required')
        positions=list(positions);channels=None if channels is None else list(channels)
        clean_ids,corrupt_ids,query=prepare_pair(self.backend,self.policy,clean_text,corrupt_text,
                                                target_token,alternative_token,query_position)
        length=len(clean_ids)
        # Validate the site before reserving a request or running the model.
        width=self.backend.hidden_size if kind=='residual' else self.backend.mlp_width
        with replace_activation(self.backend,kind,layer,torch.zeros(1,length,width),positions,channels):pass
        snapshot=self.store.snapshot(with_edits=model=='edited')
        request=dict(type='activation_patch',model=model,clean_text=clean_text,corrupt_text=corrupt_text,
            kind=kind,layer=layer,positions=list(positions),channels=None if channels is None else list(channels),
            target_token=target_token,alternative_token=alternative_token,query_position=query)
        attempt=self.ledger.begin(request,snapshot['variant_id']);started=time.perf_counter();attempted=0
        def read(*args):
            nonlocal attempted
            attempted+=1
            return self.backend._read(*args)
        try:
            sites=[dict(kind=kind,layer=layer,positions=positions,channels=channels)]
            outcome,=paired_interventions(self.backend,snapshot,(clean_ids,corrupt_ids,query),sites,
                                           target_token,alternative_token,read)
            result=dict(request=request,variant_id=snapshot['variant_id'],binding=self.binding,
                **outcome,
                input_tokens=3*length,executed_model_forwards=3,elapsed_seconds=time.perf_counter()-started,
                note='Raw and normalized metrics may disagree. Ratios are not clipped; patching alone does not identify a full circuit or a weight edit.')
            file=self.output/f'request_{attempt:04d}.json';file.write_text(json.dumps(result,indent=2))
            self.ledger.finish(attempt,True,dict(artifact=file.name,input_tokens=3*length,executed_model_forwards=3,elapsed_seconds=result['elapsed_seconds']))
            return result
        except Exception as error:
            self.ledger.finish(attempt,False,dict(error=type(error).__name__+': '+str(error),attempted_model_forwards=attempted,elapsed_seconds=time.perf_counter()-started))
            raise
