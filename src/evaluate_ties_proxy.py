"""CPU FP32 behavioral proxy panel, distinct from full BF16 generation effects."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--guard',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh proxy output required')
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    protocol=json.loads((a.study/'protocol.json').read_text());analysis=json.loads((a.study/'analysis.json').read_text())
    if sha(a.study/'dense_updates.safetensors')!=analysis['dense_update_sha256']:raise ValueError('Changed merge')
    source_meta=json.loads((Path(protocol['sources'][0]['path'])/'manifest.json').read_text())
    for name,digest in source_meta['base_files'].items():
        if sha(a.model/name)!=digest:raise ValueError('Different base files')
    view=json.loads((a.view/'manifest.json').read_text())
    if sha(a.view/'development.jsonl')!=view['files']['development.jsonl']:raise ValueError('Changed development data')
    rows=[json.loads(x) for x in (a.view/'development.jsonl').read_text().split('\n') if x.strip()][:64]
    if len(rows)!=64 or any(r['source_split']!='train' or r['view_role']!='development' for r in rows):raise ValueError('Declared training-derived development64 required')
    if a.guard.name!='validation.jsonl':raise ValueError('ARC validation only')
    guards=sorted([json.loads(x) for x in a.guard.read_text().split('\n') if x.strip()],key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:32]
    if len(guards)!=32:raise ValueError('Guard32 required')
    a.output.mkdir(parents=True);code=a.output/'code';code.mkdir()
    for name in ['evaluate_ties_proxy.py','search_metrics.py']:
        shutil.copy2(Path(__file__).parent/name,code/name)
    # Source updates and loader are frozen by the compiled merge study.
    sys.path.insert(0,str((a.study/'code').resolve()));sys.path.insert(0,str(code.resolve()))
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from delta_programming import load_effective
    from search_metrics import encode_pair,choice_metrics
    torch.set_num_threads(4)
    deltas=[]
    for source in protocol['sources']:
        path=Path(source['path'])
        if sha(path/'manifest.json')!=source['manifest_sha256']:raise ValueError('Source metadata changed')
        meta,pairs=load_effective(path)
        if meta['base_id']!=protocol['base_id']:raise ValueError('Different source base')
        deltas.append({name:(B@A).float() for name,(A,B) in pairs.items()})
    mean={name:torch.stack([d[name] for d in deltas]).mean(0) for name in deltas[0]};del deltas,pairs
    ties=load_file(str(a.study/'dense_updates.safetensors'));ratio=analysis['ties_to_mean_norm']
    if not ratio or ratio<=0:raise ValueError('Nonzero update norms required')
    conditions={'base':(None,0.),'ordinary_mean':(mean,1.),'ties':(ties,1.),
        'ties_mean_norm':(ties,1/ratio),'mean_ties_norm':(mean,ratio)}
    order=sorted(conditions,key=lambda name:hashlib.sha256(('seed401:'+name).encode()).hexdigest())
    spec=dict(dtype='CPU FP32 parameter updates and execution; not native BF16 benchmark equivalence',
        conditions=order,scales={k:v[1] for k,v in conditions.items()},merge_sha256=analysis['dense_update_sha256'],
        source_study_protocol_sha256=sha(a.study/'protocol.json'),development_ids=[r['id'] for r in rows],
        guard_ids=[r['id'] for r in guards],development_sha256=sha(a.view/'development.jsonl'),guard_sha256=sha(a.guard),
        local_source_sha256={f.name:sha(f) for f in code.glob('*.py')},scope='Teacher-forced proxies only, no generation accuracy or speed claim')
    (a.output/'protocol.json').write_text(json.dumps(spec,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').eval().requires_grad_(False)
    modules=dict(model.named_modules());original={name:modules[name].weight.detach().clone() for name in ties}
    def prefix(prompt):return tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    @torch.inference_mode()
    def score(context,target):
        prefix_ids,target_ids=encode_pair(tokenizer,context,target)
        if not target_ids:raise ValueError('Empty target')
        logits=model(torch.tensor([prefix_ids+target_ids]),use_cache=False).logits[0,len(prefix_ids)-1:-1]
        return float(F.cross_entropy(logits,torch.tensor(target_ids),reduction='sum'))
    records=[]
    for name in order:
        update,scale=conditions[name];started=time.perf_counter()
        with torch.no_grad():
            for key,value in original.items():modules[key].weight.copy_(value if update is None else value+scale*update[key])
        domain=[];guard=[]
        try:
            for row in rows:
                nll=score(prefix(row['prompt']),row['answer']+tokenizer.eos_token)
                domain.append(dict(id=row['id'],nll=nll,answer_bytes=len(row['answer'].encode())))
            for row in guards:
                texts=row['choices']['text'];context=prefix('Question: '+row['question']+'\nAnswer:')
                scores=[-score(context,text) for text in texts]
                guard.append(dict(id=row['id'],**choice_metrics(scores,texts,row['choices']['label'].index(row['answerKey']))))
        finally:
            with torch.no_grad():
                for key,value in original.items():modules[key].weight.copy_(value)
        record=dict(condition=name,domain_bits_per_answer_byte=sum(r['nll'] for r in domain)/(math.log(2)*sum(r['answer_bytes'] for r in domain)),
            guard_norm_ce=sum(r['norm_ce'] for r in guard)/len(guard),guard_acc_norm=sum(r['acc_norm'] for r in guard)/len(guard),
            elapsed_seconds=time.perf_counter()-started,domain=domain,guard=guard)
        with (a.output/'results.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ['domain','guard']}),flush=True);records.append(record)
    (a.output/'complete.json').write_text(json.dumps(dict(status='cpu_fp32_proxy_panel_complete',conditions=len(records),scope=spec['scope']),indent=2))


if __name__=='__main__':main()
