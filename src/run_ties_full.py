"""Full GSM8K panel for five prespecified dense-update conditions.

Native BF16 merged-weight execution is deliberately distinct from the CPU FP32
proxy and functional LoRA hooks. No reuse of numerically different baselines.
"""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--tasks-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh full evaluation required')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    study=json.loads((a.study/'protocol.json').read_text());analysis=json.loads((a.study/'analysis.json').read_text())
    if sha(a.study/'dense_updates.safetensors')!=analysis['dense_update_sha256']:raise ValueError('Changed dense update')
    reference=json.loads((Path(study['sources'][0]['path'])/'manifest.json').read_text())
    for name,digest in reference['base_files'].items():
        if sha(a.model/name)!=digest:raise ValueError('Wrong model base')
    a.output.mkdir(parents=True);code=a.output/'code';code.mkdir()
    for name in ['run_ties_full.py','delta_programming.py','delta_bank.py','analyze_delta.py']:
        shutil.copy2(Path(__file__).parent/name,code/name)
    sys.path.insert(0,str(code.resolve()))
    # The loader and its dependencies are frozen above; all source hashes enter
    # the root protocol before this process joins the GPU queue.
    conditions=['base','ordinary_mean','ties','ties_mean_norm','mean_ties_norm']
    ratio=analysis['ties_to_mean_norm']
    if not ratio or ratio<=0:raise ValueError('Nonzero source norms required')
    scales={'base':0.,'ordinary_mean':1.,'ties':1.,'ties_mean_norm':1/ratio,'mean_ties_norm':ratio}
    order=sorted(conditions,key=lambda name:hashlib.sha256(('seed401:'+name).encode()).hexdigest())
    task_sources={p.name:sha(p) for p in a.tasks_dir.glob('*.yaml')}
    protocol=dict(conditions=order,scales=scales,base_files=reference['base_files'],study_protocol_sha256=sha(a.study/'protocol.json'),
        merge_sha256=analysis['dense_update_sha256'],source_sha256={p.name:sha(p) for p in code.glob('*.py')},
        execution='FP32 addition to native source then round to BF16 weight; CUDA0 SDPA; no functional LoRA hook',
        task='idea4_gsm8k_cot_zeroshot_v1',expected_items=1319,batch_size=32,max_new_tokens=512,
        thinking=False,do_sample=False,task_sources=task_sources,
        selection='All five conditions fixed before development proxy results; no test-driven program selection')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from delta_programming import load_effective
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    torch.set_num_threads(4)
    updates=[]
    for source in study['sources']:
        path=Path(source['path'])
        if sha(path/'manifest.json')!=source['manifest_sha256']:raise ValueError('Changed source metadata')
        meta,pairs=load_effective(path)
        if meta['base_id']!=study['base_id']:raise ValueError('Wrong update base')
        updates.append({name:(B@A).float() for name,(A,B) in pairs.items()})
    mean={name:torch.stack([d[name] for d in updates]).mean(0) for name in updates[0]};del updates,pairs
    ties=load_file(str(a.study/'dense_updates.safetensors'))
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    modules=dict(model.named_modules());original={name:modules[name].weight.detach().clone() for name in ties}
    alias=Path(lm_eval.__file__).resolve().parent/'tasks/idea4_local';alias.mkdir(exist_ok=True)
    for file in a.tasks_dir.glob('*.yaml'):shutil.copy2(file,alias/file.name)
    lock=open(f'/tmp/idea4-{os.getuid()}-gpu-0.lock','a');results=[]
    for name in order:
        folder=a.output/name;folder.mkdir();scale=scales[name];delta=ties if name.startswith('ties') else mean
        native={key:(value if name=='base' else (value.float()+scale*delta[key]).to(value.dtype)) for key,value in original.items()}
        native_hashes={key:hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() for key,value in native.items()}
        print(json.dumps(dict(event='waiting_gpu_0',condition=name)),flush=True)
        waiting=time.perf_counter();fcntl.flock(lock,fcntl.LOCK_EX);wait=time.perf_counter()-waiting;started=time.perf_counter()
        try:
            with torch.no_grad():
                for key,value in native.items():modules[key].weight.copy_(value)
            model.to('cuda:0');wrapper=HFLM(pretrained=model,tokenizer=tok,batch_size=32,max_length=2048,
                enable_thinking=False,chat_template_args={'enable_thinking':False})
            (folder/'protocol.json').write_text(json.dumps(dict(condition=name,scale=scale,native_tensor_sha256=native_hashes,
                harness=importlib.metadata.version('lm_eval'),transformers=importlib.metadata.version('transformers'),
                torch=torch.__version__,root_protocol_sha256=sha(a.output/'protocol.json')),indent=2))
            result=lm_eval.simple_evaluate(model=wrapper,tasks=[protocol['task']],num_fewshot=0,
                task_manager=TaskManager(include_path=str(alias)),batch_size=32,log_samples=True,
                use_cache=str(folder/'responses'),bootstrap_iters=1000,apply_chat_template=True,
                gen_kwargs=dict(max_gen_toks=512,do_sample=False),random_seed=20261106,numpy_random_seed=20261106,
                torch_random_seed=20261106,fewshot_random_seed=20261106)
            samples=result.pop('samples')[protocol['task']]
            if len(samples)!=1319 or len({r['doc_id'] for r in samples})!=1319:raise ValueError('Incomplete full public partition')
            with (folder/'samples.jsonl').open('w') as stream:
                for row in samples:stream.write(json.dumps(row,default=str)+'\n')
            (folder/'results.json').write_text(json.dumps(result,default=str,indent=2))
            summary=dict(status='full_gsm1319_complete',condition=name,items=1319,gpu_wait_seconds=wait,
                resident_seconds=time.perf_counter()-started,scores=result['results'])
            (folder/'complete.json').write_text(json.dumps(summary,indent=2));results.append(summary)
            print(json.dumps(dict(event='full_gsm_complete',**summary)),flush=True)
            del wrapper
        finally:
            try:
                model.to('cpu')
                with torch.no_grad():
                    for key,value in original.items():modules[key].weight.copy_(value)
                torch.cuda.empty_cache()
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
    (a.output/'complete.json').write_text(json.dumps(dict(status='full_five_condition_gsm_panel_complete',results=results),indent=2))


if __name__=='__main__':main()
