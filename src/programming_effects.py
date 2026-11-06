"""Full public capability panel for compiled update/depth programs.

One resident native model, fresh HFLM wrapper/cache per condition. No smoke
subset. All conditions and baselines use this same pinned evaluation path.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import torch
from weightlab import Lab,write_json
from delta_bank import DeltaBank,fingerprint,sha
from depth_programming import depth_variant

NAMES=['programming_effects.py','depth_programming.py','delta_bank.py','weightlab.py']
SOURCES={name:(Path(__file__).parent/name).read_bytes() for name in NAMES}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--programs',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True);p.add_argument('--standard',type=Path,required=True)
    p.add_argument('--tasks-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--group',choices=['gsm8k','guards'],required=True);p.add_argument('--variants',nargs='+',required=True)
    a=p.parse_args();lab=Lab(a.model);lab.model.requires_grad_(False);base=lab.manifest()
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    alias=Path(lm_eval.__file__).resolve().parent/'tasks/idea4_local';alias.mkdir(exist_ok=True)
    for path in a.tasks_dir.glob('*.yaml'):shutil.copy2(path,alias/path.name)
    tasks=['idea4_gsm8k_cot_zeroshot_v1'] if a.group=='gsm8k' else ['idea4_arc_challenge_v1','idea4_hellaswag_v1']
    batch=32 if a.group=='gsm8k' else 8
    bank=DeltaBank(lab.model,fingerprint(base['files']))
    for name in a.variants:
        if name in ['base','bi10','bi20','random10','tail10']:continue
        path=a.standard if name=='standard-seed11' else a.programs/name/'delta'
        bank.register(name,path)
    depth=json.loads((a.probe/'layer_programs.json').read_text())['programs']
    for name in a.variants:
        out=a.output/name/a.group
        if (out/'complete.json').exists():continue
        if out.exists():raise FileExistsError('Inspect partial full evaluation '+str(out))
        out.mkdir(parents=True);(out/'code').mkdir()
        for filename,data in SOURCES.items():(out/'code'/filename).write_bytes(data)
        scope=depth_variant(lab.model,depth[name]['kept']) if name in depth else bank.activate(name) if name!='base' else contextlib.nullcontext()
        with scope:
            model=HFLM(pretrained=lab.model,tokenizer=lab.tokenizer,batch_size=batch,max_length=2048,
                enable_thinking=False,chat_template_args={'enable_thinking':False})
            protocol=dict(model=str(a.model.resolve()),variant=name,group=a.group,batch_size=batch,max_new_tokens=512,
                intervention=depth[name] if name in depth else dict(manifest_sha256=sha((a.standard if name=='standard-seed11' else a.programs/name/'delta')/'manifest.json')) if name!='base' else None,
                parameters=sum(v.numel() for v in lab.model.parameters()),harness=importlib.metadata.version('lm_eval'),
                tasks=tasks,source_sha256={n:hashlib.sha256(b).hexdigest() for n,b in SOURCES.items()},
                model_source_files=base['files'],note='Same preinitialized-model HFLM path for every condition; fresh response cache; full partitions only')
            write_json(out/'protocol.json',protocol);started=time.perf_counter()
            result=lm_eval.simple_evaluate(model=model,tasks=tasks,num_fewshot=0,task_manager=TaskManager(include_path=str(alias)),
                batch_size=batch,log_samples=True,use_cache=str(out/'responses'),bootstrap_iters=1000,
                apply_chat_template=True,gen_kwargs=dict(max_gen_toks=512,do_sample=False),
                random_seed=20261106,numpy_random_seed=20261106,torch_random_seed=20261106,fewshot_random_seed=20261106)
            samples=result.pop('samples');counts={}
            for task,rows in samples.items():
                counts[task]=len({r['doc_id'] for r in rows})
                expected=1319 if 'gsm8k' in task else 1172 if 'arc_challenge' in task else 10042
                if counts[task]!=expected:raise ValueError('Incomplete public partition')
                with open(out/(task+'.samples.jsonl'),'w') as stream:
                    for row in rows:stream.write(json.dumps(row,default=str)+'\n')
            write_json(out/'results.json',result)
            write_json(out/'complete.json',dict(status='full_programming_effect_complete',samples=counts,
                elapsed_seconds=time.perf_counter()-started,variant=name,parameters=protocol['parameters'],
                note='Parameter storage/depth and full task scores. In-process retained native layers prevent treating CUDA allocation as standalone compressed memory.'))
            print(json.dumps(dict(event='full_effect_complete',variant=name,group=a.group,scores=result['results'])),flush=True)
        del model
        torch.cuda.empty_cache()


if __name__=='__main__':main()
