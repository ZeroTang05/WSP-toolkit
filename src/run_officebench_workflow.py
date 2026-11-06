"""Execute frozen unedited-model workflow comparisons on actual OfficeBench.

The optional task subset is explicitly a diagnostic panel, not full benchmark
confirmation. No target model is trained and no evaluation fields enter prompts.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sys
import time
import zipfile

SOURCES=['run_officebench_workflow.py','officebench_runtime.py','officebench_container_worker.py','officebench_workflow.py']


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--acquisition',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--docker-host',required=True);p.add_argument('--image',required=True)
    p.add_argument('--tasks',nargs='+',default=['all']);p.add_argument('--modes',nargs='+',default=['single','self_refine','independent_select','collaborate','no_communication'])
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--dtype',choices=['float32','bfloat16'],default='bfloat16')
    p.add_argument('--seed',type=int,default=11);p.add_argument('--token-budget',type=int,default=8192)
    p.add_argument('--decision-tokens',type=int,default=384);p.add_argument('--max-steps',type=int,default=20)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new frozen run; never restart a live or partial run implicitly')
    if len(set(a.modes))!=len(a.modes) or min(a.token_budget,a.decision_tokens,a.max_steps)<=0:raise ValueError('Invalid run policy')
    os.environ['CUDA_VISIBLE_DEVICES']='0' if a.device=='cuda' else ''
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    acquisition=json.loads((a.acquisition/'manifest.json').read_text());archive=a.acquisition/'source.zip'
    if sha(archive)!=acquisition['source_archive_sha256']:raise ValueError('Changed benchmark archive')
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            if entry.is_dir():continue
            relative=PurePosixPath(*PurePosixPath(entry.filename).parts[1:])
            if '..' in relative.parts or relative.is_absolute():raise ValueError('Invalid archive path')
            if hashlib.sha256(z.read(entry)).hexdigest()!=sha(a.acquisition/'source'/relative):
                raise ValueError('Changed benchmark source: '+str(relative))
    available=sorted([f'{q.parents[1].name}/{q.stem}' for q in (a.acquisition/'source').glob('tasks/*/subtasks/*.json')],
        key=lambda s:tuple(int(x) for x in s.replace('/','-').split('-')))
    tasks=available if a.tasks==['all'] else a.tasks
    if not tasks or len(set(tasks))!=len(tasks) or set(tasks)-set(available):raise ValueError('Unknown or duplicated task')
    a.output.mkdir(parents=True);code=a.output/'code';code.mkdir()
    for name in SOURCES:shutil.copy2(Path(__file__).parent/name,code/name)
    sys.path.insert(0,str(code.resolve()))
    from officebench_workflow import run_case,MODES
    from officebench_runtime import OfficeBenchRuntime
    if set(a.modes)-set(MODES):raise ValueError('Unknown workflow mode')
    import subprocess
    image=subprocess.run(['docker','--host',a.docker_host,'image','inspect',a.image,'--format','{{.Id}}'],capture_output=True,text=True,check=True).stdout.strip()
    protocol=dict(model=str(a.model.resolve()),benchmark_commit=acquisition['commit'],benchmark_archive_sha256=sha(archive),
        image_id=image,tasks=tasks,modes=a.modes,seed=a.seed,device=a.device,dtype=a.dtype,
        token_budget=a.token_budget,decision_tokens=a.decision_tokens,max_steps=a.max_steps,
        context='last 8 executed/proposed actions; first 3000 characters of each tool observation; task and catalog always present; no hidden scoring fields',
        sampling=dict(thinking=False,do_sample=True,temperature=.6,top_p=.95,top_k=20,repetition_penalty=1.),
        source_sha256={name:sha(code/name) for name in SOURCES},
        evaluation_scope='full pinned task inventory' if tasks==available else 'explicit public diagnostic subset; not full confirmation',
        parameter_edits=False,external_model_delegation=False,
        resource_scope='Shared upper token/action budgets, not identical realized FLOPs. All roles same unedited checkpoint. Ported environment, not original leaderboard equivalence.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));lock=None;waiting=time.perf_counter()
    if a.device=='cuda':
        lock=open(f'/tmp/idea4-{os.getuid()}-gpu-0.lock','a');print(json.dumps(dict(event='waiting_gpu_0')),flush=True)
        fcntl.flock(lock,fcntl.LOCK_EX)
    wait_seconds=time.perf_counter()-waiting;started=time.perf_counter()
    import torch
    import transformers
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if transformers.__version__!='4.57.6':raise ValueError('Pinned Qwen3 workflow runtime required')
    torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
        torch_dtype=getattr(torch,a.dtype),attn_implementation='sdpa').to(a.device).eval()
    model.requires_grad_(False)
    if tokenizer.pad_token_id is None:tokenizer.pad_token_id=tokenizer.eos_token_id
    source_files={f.name:sha(f) for f in a.model.iterdir() if f.is_file() and (f.suffix=='.safetensors' or f.name in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja'])}
    (a.output/'model.json').write_text(json.dumps(dict(source_sha256=source_files,torch=torch.__version__,transformers=transformers.__version__,
        generation_config=model.generation_config.to_dict(),parameters=sum(v.numel() for v in model.parameters()),
        setup_seconds=time.perf_counter()-started,gpu_lock_wait_seconds=wait_seconds),indent=2))
    @torch.inference_mode()
    def generate(role,system,prompt,allocation,seed):
        rendered=tokenizer.apply_chat_template([dict(role='system',content=system),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens=tokenizer(rendered,return_tensors='pt',add_special_tokens=False).to(a.device)
        if tokens.input_ids.shape[1]+allocation>model.config.max_position_embeddings:raise ValueError('Context limit exceeded; no silent prompt truncation')
        if a.device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.random.fork_rng(devices=[0] if a.device=='cuda' else []):
            torch.manual_seed(seed)
            output=model.generate(**tokens,max_new_tokens=allocation,do_sample=True,temperature=.6,top_p=.95,top_k=20,
                repetition_penalty=1.,pad_token_id=tokenizer.pad_token_id,use_cache=True)
        if a.device=='cuda':torch.cuda.synchronize()
        new=output[0,tokens.input_ids.shape[1]:].cpu().tolist()
        return dict(text=tokenizer.decode(new,skip_special_tokens=True),raw_text=tokenizer.decode(new,skip_special_tokens=False),
            output_tokens=len(new),input_tokens=int(tokens.input_ids.shape[1]),seconds=time.perf_counter()-start,
            generated_token_ids=new,prompt_token_sha256=hashlib.sha256(tokens.input_ids.cpu().numpy().tobytes()).hexdigest())
    print(json.dumps(dict(event='office_model_ready',device=a.device,tasks=len(tasks),modes=a.modes)),flush=True)
    results=[]
    for task in tasks:
        # Rotate mode order deterministically by task, without using outcomes.
        offset=int(hashlib.sha256(task.encode()).hexdigest()[:8],16)%len(a.modes)
        modes=a.modes[offset:]+a.modes[:offset]
        for mode in modes:
            folder=a.output/'cases'/task.replace('/','_')/mode;case_start=time.perf_counter()
            print(json.dumps(dict(event='office_case_start',task=task,mode=mode)),flush=True)
            with OfficeBenchRuntime(a.acquisition,task,folder,docker_host=a.docker_host,image=image) as env:
                run=run_case(env,mode,generate,seed=a.seed,token_budget=a.token_budget,decision_tokens=a.decision_tokens,max_steps=a.max_steps)
                (folder/'workflow.json').write_text(json.dumps(run,indent=2))
                evaluation=env.evaluate()
            result=dict(task_id=task,mode=mode,all_pass=evaluation['all_pass'],checks=evaluation['checks'],
                output_tokens=run['output_tokens'],input_tokens=run['input_tokens'],model_seconds=run['model_seconds'],
                termination=run['termination'],model_calls=len(run['calls']),decisions=len(run['decisions']),
                end_to_end_seconds=time.perf_counter()-case_start)
            with (a.output/'results.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            results.append(result);print(json.dumps(dict(event='office_case_complete',**result)),flush=True)
    (a.output/'complete.json').write_text(json.dumps(dict(status='office_workflow_comparison_complete',cases=len(results),
        active_seconds=time.perf_counter()-started,gpu_lock_wait_seconds=wait_seconds,
        evaluation_scope=protocol['evaluation_scope'],scope=protocol['resource_scope']),indent=2))


if __name__=='__main__':main()
