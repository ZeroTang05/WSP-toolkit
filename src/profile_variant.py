"""Fixed-workload prefill/decode measurements with one resident variant.

Run base and variants in independent processes and randomize their order in
replicated experiments. Logical parameter storage and CUDA allocator memory are
reported separately. Fixed-length decoding ignores EOS by construction.
"""
import argparse
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
import torch
from weightlab import Lab,write_json
from delta_bank import DeltaBank,fingerprint,sha
from structure_bundle import apply_structure
from mask_bundle import apply_masks


SOURCE=Path(__file__).read_bytes()
HELPERS={name:(Path(__file__).parent/name).read_bytes() for name in ['weightlab.py','delta_bank.py','structure_bundle.py','mask_bundle.py']}


def summarize(values,tokens):
    ordered=sorted(values);median=statistics.median(values)
    return dict(milliseconds=[v*1000 for v in values],median_ms=median*1000,
        p05_ms=ordered[int(.05*(len(values)-1))]*1000,p95_ms=ordered[int(.95*(len(values)-1))]*1000,
        tokens_per_second=tokens/median)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--structure',type=Path)
    p.add_argument('--mask',type=Path);p.add_argument('--delta',type=Path)
    p.add_argument('--repeats',type=int,default=15);p.add_argument('--warmup',type=int,default=5)
    a=p.parse_args()
    if sum(x is not None for x in [a.structure,a.mask,a.delta])>1:raise ValueError('One intervention per profiling protocol')
    if a.repeats<3 or a.warmup<1:raise ValueError('Insufficient profiling repetitions')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    (a.output/'profile_variant.py').write_bytes(SOURCE)
    (a.output/'code').mkdir()
    for name,data in HELPERS.items():(a.output/'code'/name).write_bytes(data)
    lab=Lab(a.model);base=lab.manifest();write_json(a.output/'base_manifest.json',base)
    source_parameters=sum(value.numel() for value in lab.model.parameters())
    intervention={};scope=contextlib.nullcontext()
    if a.structure:
        apply_structure(lab.model,a.structure,base['files']);intervention=dict(kind='structure',path=str(a.structure.resolve()),sha256=sha(a.structure))
    if a.mask:
        apply_masks(lab.model,a.mask,base['files']);intervention=dict(kind='mask',path=str(a.mask.resolve()),sha256=sha(a.mask/'manifest.json'))
    if a.delta:
        bank=DeltaBank(lab.model,fingerprint(base['files']));bank.register('profile',a.delta);scope=bank.activate('profile')
        intervention=dict(kind='delta',path=str(a.delta.resolve()),sha256=sha(a.delta/'manifest.json'))
    model=lab.model.requires_grad_(False)
    supports_last='logits_to_keep' in inspect.signature(model.forward).parameters
    kwargs={'logits_to_keep':1} if supports_last else {}
    ids=lab.tokenizer.encode('The purpose of this fixed workload is to measure inference. '*150,add_special_tokens=False)
    if len(ids)<512:raise ValueError('Fixed workload text is too short')
    protocol=dict(intervention=intervention,repeats=a.repeats,warmup=a.warmup,dtype='bfloat16',
        attn_implementation='sdpa',torch=torch.__version__,cuda_build=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        last_token_logits_only=supports_last,decode_tokens=32,decode_policy='greedy tokens, ignore EOS, new cache each repetition',
        resident_variants=1,source_sha256=hashlib.sha256(SOURCE).hexdigest(),
        helper_sha256={name:hashlib.sha256(data).hexdigest() for name,data in HELPERS.items()},
        base_parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters()),
        source_parameters=source_parameters,active_dense_parameter_bytes=sum(v.numel()*v.element_size() for v in model.parameters()),
        active_delta_factor_bytes=bank.storage()['factor_bytes'] if a.delta else 0,
        parameters=sum(v.numel() for v in model.parameters()))
    write_json(a.output/'protocol.json',protocol)
    results=[]

    @torch.inference_mode()
    def prefill(inputs):
        return model(input_ids=inputs,use_cache=True,**kwargs)

    @torch.inference_mode()
    def decode(inputs):
        initial=prefill(inputs);cache=initial.past_key_values;token=initial.logits[:,-1].argmax(-1,keepdim=True)
        del initial
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(32):
            output=model(input_ids=token,past_key_values=cache,use_cache=True,**kwargs)
            cache=output.past_key_values;token=output.logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize()
        return time.perf_counter()-start

    with scope,torch.inference_mode():
        for batch,length in [(1,128),(1,512),(8,128),(8,512)]:
            inputs=torch.tensor([ids[:length]]*batch,device='cuda')
            for _ in range(a.warmup):
                output=prefill(inputs);del output
                decode(inputs)
            torch.cuda.synchronize();resident=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            prefill_times=[];decode_times=[]
            for _ in range(a.repeats):
                torch.cuda.synchronize();start=time.perf_counter()
                output=prefill(inputs);torch.cuda.synchronize();prefill_times.append(time.perf_counter()-start)
                del output
                decode_times.append(decode(inputs))
            row=dict(batch=batch,input_tokens=length,prefill=summarize(prefill_times,batch*length),
                decode=summarize(decode_times,batch*32),resident_allocated_bytes=resident,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            results.append(row);print(json.dumps(row),flush=True)
            del inputs
    write_json(a.output/'complete.json',dict(status='fixed_workload_profile_complete',protocol=protocol,workloads=results,
        note='Single-process warm fixed workloads; repeated-process order randomization and application latency are separate evidence. CUDA allocated/reserved memory is not raw disk storage.'))


if __name__=='__main__':main()
