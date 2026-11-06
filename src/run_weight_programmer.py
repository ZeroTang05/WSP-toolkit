"""Run a fresh local model-programmer with the original-style weight tools.

Only the model checkpoint, author practice files and own writable work directory
reach the CPU kernel. No reference solution, research notes or prior transcript
is included in the programmer conversation. This is an independent HF experiment,
not a claim of equivalence to the original programmer model or vLLM engine.
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
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def fingerprint(folder):
    files={p.name:sha(p) for p in sorted(folder.iterdir())
        if p.is_file() and (p.suffix in ['.safetensors','.json','.jinja'] or p.name.startswith('tokenizer'))}
    if not any(n.endswith('.safetensors') for n in files):raise ValueError('Local checkpoint required')
    return files


def load_practice(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['revision']!='4d09ba6d6a7e891dd8063974dfe81b3d78e3494f':raise ValueError('Pinned author practice required')
    allowed={'manifest.json','LICENSE','stats.json','gsm8k_train.jsonl','math_train.jsonl','aime.jsonl'}
    if {p.name for p in folder.iterdir()}!=allowed:raise ValueError('Practice must contain only the authorized acquisition')
    rows=[]
    for name,expected in [('gsm8k_train.jsonl',7469),('math_train.jsonl',7275),('aime.jsonl',714)]:
        path=folder/name
        if sha(path)!=manifest['files'][name]['sha256']:raise ValueError('Practice hash mismatch')
        items=[json.loads(line) for line in path.read_text().split('\n') if line.strip()]
        if len(items)!=expected:raise ValueError('Practice count mismatch')
        rows.extend(items)
    return rows,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True)
    p.add_argument('--programmer',type=Path,required=True);p.add_argument('--practice',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=401)
    p.add_argument('--max-turns',type=int,default=500);p.add_argument('--reply-tokens',type=int,default=8192)
    p.add_argument('--context-tokens',type=int,default=32768)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh experiment required; never restart a live discovery')
    if min(a.max_turns,a.reply_tokens,a.context_tokens)<=0:raise ValueError('Positive programmer limits required')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    rows,practice_manifest=load_practice(a.practice)
    target_files=fingerprint(a.target);programmer_files=fingerprint(a.programmer)
    target_type=json.loads((a.target/'config.json').read_text())['model_type']
    if target_type not in ['qwen3','qwen3_5']:raise ValueError('Verified target backend required')
    expected_tf='5.3.0' if target_type=='qwen3_5' else '4.57.6'
    if importlib.metadata.version('transformers')!=expected_tf:raise ValueError('Use the separately pinned target environment')
    # Fail before submitting a session if the advertised CPU stack is incomplete.
    environment={name:importlib.metadata.version(name) for name in
        ['torch','transformers','numpy','scipy','scikit-learn','pandas','sympy','networkx','safetensors','tokenizers','math-verify']}
    if target_type=='qwen3_5':
        base=json.loads((a.target/'idea4_wsp_base_manifest.json').read_text())
        if base['revision']!='2fc06364715b967f1860aea9cf38778875588b17':raise ValueError('Original target revision required')
        if any(sha(a.target/name)!=entry['sha256'] for name,entry in base['files'].items()):raise ValueError('Changed original target')
    a.output.mkdir(parents=True)
    code=a.output/'code';code.mkdir()
    shutil.copy2(__file__,code/Path(__file__).name)
    shutil.copytree(Path(__file__).parent/'weight_programming',code/'weight_programming',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    sys.path.insert(0,str(code.resolve()))
    import weight_programming
    if Path(weight_programming.__file__).resolve().parent!=code.resolve()/'weight_programming':raise RuntimeError('Frozen toolkit import required')
    initial_prompt=(
        'Improve this open-weight model through direct, evidence-based parameter programming. '
        'Study its actual mathematical behavior and failure mechanisms using the supplied tools; '
        'evaluate controlled edits on practice inputs unused to select them. Preserve broad useful behavior. '
        'Final evaluation is separate and unavailable. The target architecture is '+target_type+'. '
        'Read /model/config.json for architecture information. Authorized practice files: '
        '/practice/gsm8k_train.jsonl (7469), /practice/math_train.jsonl (7275), /practice/aime.jsonl (714). '
        'They contain id, source, problem, answer and where available solution. Parse literal LF-delimited JSON. '
        'You have 300 persistent CPU executions and 50 successful forward requests. '
        'Generation has thinking enabled, temperature 0.6, top_p 0.95, top_k 20, generated-token presence penalty 1.5, '
        'up to 32768 new tokens; per request at most 512 samples, or 256 per model for paired original/edited. '
        'Teacher-forced requests allow up to 1024 texts, 32768 tokens per text, 524288 total tokens. '
        'Returned artifacts include metadata and arrays; their paths are provided in each tool result. '
        'Check the published tool schemas for exact request syntax and legal layers. '
        'Use the entire forward-pass budget before submitting your final choice. '
        'The submitted model is exactly the current saved edits. No author solution or prior run analysis is provided.')
    protocol=dict(scope='Fresh local programmer experiment, not held-out effects or original-engine equivalence',
        target=str(a.target.resolve()),target_files=target_files,programmer=str(a.programmer.resolve()),programmer_files=programmer_files,
        practice_manifest=practice_manifest,initial_prompt=initial_prompt,seed=a.seed,
        environment=environment,
        source_sha256={str(f.relative_to(code)):sha(f) for f in code.rglob('*.py')},
        limits=dict(python=300,forward=50,programmer_turns=a.max_turns,programmer_reply_tokens=a.reply_tokens,programmer_context_tokens=a.context_tokens),
        programmer_sampling=dict(thinking=False,do_sample=False),target_engine='pinned Hugging Face, SDPA, BF16, one GPU',
        independence='Fresh messages; isolated kernel mounts contain no reference edit, prior analysis or held-out tasks. Pretraining contamination unknown.')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(dict(event='waiting_gpu_0',target=target_type,programmer=a.programmer.name,practice_items=len(rows))),flush=True)
    lock=open(f'/tmp/idea4-{os.getuid()}-gpu-0.lock','a');waiting=time.perf_counter();fcntl.flock(lock,fcntl.LOCK_EX)
    wait_seconds=time.perf_counter()-waiting
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from weight_programming import ProgrammingSession,PracticeSet,ForwardPolicy
    from weight_programming.programmer import run_programmer
    from weight_programming.grading import MathVerifyGrader
    torch.set_num_threads(4)
    target_tok=AutoTokenizer.from_pretrained(a.target,local_files_only=True)
    if target_type=='qwen3_5':
        from transformers import Qwen3_5ForConditionalGeneration
        from weight_programming.qwen35_backend import Qwen35Backend as Backend
        Model=Qwen3_5ForConditionalGeneration
    else:
        from weight_programming.hf_backend import HuggingFaceBackend as Backend
        Model=AutoModelForCausalLM
    target=Model.from_pretrained(a.target,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval().requires_grad_(False)
    programmer=AutoModelForCausalLM.from_pretrained(a.programmer,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval().requires_grad_(False)
    programmer_tok=AutoTokenizer.from_pretrained(a.programmer,local_files_only=True)
    def render(messages,schemas):
        enriched=[dict(m) for m in messages]
        enriched[0]['content']+='\nTOOL SCHEMAS:\n'+json.dumps(schemas)
        return programmer_tok.apply_chat_template(enriched,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    def count(messages,schemas):return len(programmer_tok.encode(render(messages,schemas),add_special_tokens=False))
    def generate(messages,schemas,cap,seed):
        inputs=programmer_tok(render(messages,schemas),return_tensors='pt',add_special_tokens=False).to('cuda:0')
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.random.fork_rng(devices=[0]),torch.inference_mode():
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            generated=programmer.generate(**inputs,max_new_tokens=cap,do_sample=False,pad_token_id=programmer_tok.eos_token_id)
        torch.cuda.synchronize();suffix=generated[0,inputs.input_ids.shape[1]:]
        response=dict(text=programmer_tok.decode(suffix,skip_special_tokens=True),input_tokens=inputs.input_ids.shape[1],
            output_tokens=len(suffix),seconds=time.perf_counter()-started)
        print(json.dumps(dict(event='programmer_reply',input_tokens=response['input_tokens'],output_tokens=response['output_tokens'],seconds=response['seconds'])),flush=True)
        return response
    with ProgrammingSession(lambda store:Backend(target,target_tok,store),a.target,a.practice,
            PracticeSet(rows,role='train',name='exact-author-practice'),a.output/'work',a.output/'control',
            policy=ForwardPolicy(),grader=MathVerifyGrader()) as session:
        result=run_programmer(session,generate,initial_prompt,
            programmer_identity=dict(model=a.programmer.name,files=programmer_files,sampling=protocol['programmer_sampling']),
            count_tokens=count,max_turns=a.max_turns,max_reply_tokens=a.reply_tokens,max_context_tokens=a.context_tokens,
            seed=a.seed,require_forward_budget=True)
        session.service.store.materialize_edits(a.output/'submitted-edits')
    result.update(gpu_lock_wait_seconds=wait_seconds,scope=protocol['scope'])
    (a.output/'complete.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(event='programmer_complete',status=result['status'],edited_tensors=len(result['edited_names']))),flush=True)


if __name__=='__main__':main()
