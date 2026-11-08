"""Real-model programming observations: layer influence, ablation, delta rank.

Calibration/response data come only from official training-derived views.
Public test inference is a separate phase, never a source of these statistics.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from safetensors.torch import save_file
from weightlab import Lab,write_json
from delta_bank import DeltaBank,fingerprint,sha
from delta_programming import load_effective,weighted_truncate,emit,describe,norm,residual
from depth_programming import block_influence,choose_layers,depth_variant
from prepare_training_views import read_jsonl
from search_metrics import losses


NAMES=['programming_probe.py','delta_programming.py','depth_programming.py','delta_bank.py',
    'analyze_delta.py','weightlab.py','prepare_training_views.py','finqa_views.py','search_metrics.py']
SOURCES={name:(Path(__file__).parent/name).read_bytes() for name in NAMES}


def correlation(first,second):
    def ranks(values):
        order=sorted(range(len(values)),key=lambda i:values[i]);result=[0.]*len(values);i=0
        while i<len(order):
            j=i+1
            while j<len(order) and values[order[j]]==values[order[i]]:j+=1
            for k in range(i,j):result[order[k]]=(i+j-1)/2
            i=j
        return torch.tensor(result,dtype=torch.float64)
    a,b=ranks(first),ranks(second);a-=a.mean();b-=b.mean();den=a.norm()*b.norm()
    return float(a@b/den) if den else None


@torch.inference_mode()
def collect(lab,rows,soup):
    influence=[torch.zeros((),device='cuda') for _ in lab.layers];handles=[]
    for i,layer in enumerate(lab.layers):
        def hook(module,args,output,index=i):
            value=output[0] if isinstance(output,(tuple,list)) else output
            influence[index].add_(block_influence(args[0],value).mean())
        handles.append(layer.register_forward_hook(hook))
    def forward(row):
        ids=lab.tokenizer.encode(lab.prefix(row),add_special_tokens=False)+lab.tokenizer.encode(row['answer'],add_special_tokens=False)+[lab.tokenizer.eos_token_id]
        lab.model(torch.tensor([ids],device='cuda'),use_cache=False)
    try:
        for row in rows:forward(row)
    finally:
        for handle in handles:handle.remove()
    bi=[float(value/len(rows)) for value in influence]
    bank=DeltaBank(lab.model,fingerprint(lab.manifest()['files']));bank.register('source',soup)
    meta,data=bank.roles['source'];moments={};counts={};handles=[]
    for name,spec in meta['targets'].items():
        A=data[name+'.A'].to('cuda');moments[name]=torch.zeros(spec['rank'],spec['rank'],dtype=torch.float64,device='cuda');counts[name]=0
        def hook(module,args,key=name,a=A):
            x=args[0].detach().reshape(-1,a.shape[1]).float();z=(x@a.T).double()
            moments[key].add_(z.T@z);counts[key]+=len(x)
        handles.append(lab.model.get_submodule(name).register_forward_pre_hook(hook))
    try:
        with bank.activate('source'):
            for row in rows:forward(row)
    finally:
        for handle in handles:handle.remove()
    return bi,{name:(value/counts[name]).cpu() for name,value in moments.items()},counts


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--programs',type=Path,required=True);p.add_argument('--view',type=Path,required=True)
    p.add_argument('--guard',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration-n',type=int,default=128);p.add_argument('--response-n',type=int,default=64)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads((a.view/'manifest.json').read_text())
    for file in ['calibration.jsonl','development.jsonl']:
        if sha(a.view/file)!=manifest['files'][file]:raise ValueError('Changed source view')
    rows=read_jsonl(a.view/'calibration.jsonl')[:a.calibration_n];domain=read_jsonl(a.view/'development.jsonl')[:a.response_n]
    if len(rows)!=a.calibration_n or len(domain)!=a.response_n:raise ValueError('Incomplete observation data')
    if any(r['source_split']!='train' for r in rows+domain):raise ValueError('No heldout construction data allowed')
    if any(r['view_role']!='calibration' for r in rows) or any(r['view_role']!='development' for r in domain):raise ValueError('Wrong data roles')
    if a.guard.name!='validation.jsonl':raise ValueError('Guard must be ARC validation')
    guards=sorted(read_jsonl(a.guard),key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:32]
    a.output.mkdir(parents=True);(a.output/'code').mkdir()
    for name,data in SOURCES.items():(a.output/'code'/name).write_bytes(data)
    write_json(a.output/'protocol.json',dict(model=str(a.model.resolve()),calibration_ids=[r['id'] for r in rows],
        development_ids=[r['id'] for r in domain],guard_ids=[r['id'] for r in guards],view_sha256=manifest['files'],
        guard_sha256=sha(a.guard),code_sha256={n:hashlib.sha256(b).hexdigest() for n,b in SOURCES.items()},
        bi='ShortGPT equation2; mean of per-example mean token cosine, variable-length domain calibration adaptation',
        moments='uncentered latent second moment under the full uniform-soup variant; no gradients',
        response='exact one-layer deletion versus native source on development64/ARCvalidation32; not generation accuracy'))
    lab=Lab(a.model);lab.model.requires_grad_(False);base=lab.manifest();write_json(a.output/'base_manifest.json',base)
    bi,moments,counts=collect(lab,rows,a.programs/'soup/delta')
    save_file({name:value.contiguous() for name,value in moments.items()},str(a.output/'latent_moments.safetensors'))
    source,pairs=load_effective(a.programs/'soup/delta');compressed={};evidence={}
    for name,(A,B) in pairs.items():
        aa,bb,record=weighted_truncate(A,B,moments[name],8);compressed[name]=(aa,bb)
        record.update(relative_weight_error=residual(A,B,aa.float(),bb.float())/max(norm(A,B),1e-30),tokens=counts[name])
        evidence[name]=record
    emit(a.programs/'activation-r8/delta',base,compressed,'latent activation-weighted update SVD',dict(
        source=str(a.programs/'soup/delta'),moment_file_sha256=sha(a.output/'latent_moments.safetensors'),modules=evidence,
        scope='Adapted local SVD-LLM objective; compress delta only, no sequential recovery or full-base SVD'))
    write_json(a.programs/'activation-r8/geometry.json',describe(a.programs/'activation-r8/delta'))
    selectors={}
    for label,mode,fraction in [('bi10','bi',.1),('bi20','bi',.2),('random10','random',.1),('tail10','tail',.1)]:
        kept=choose_layers(bi,max(1,round(len(bi)*fraction)),mode)
        selectors[label]=dict(kept=kept,removed=[i for i in range(len(bi)) if i not in kept],requested_fraction=fraction,
            actual_layer_fraction=1-len(kept)/len(bi),policy=mode,base_id=fingerprint(base['files']))
    write_json(a.output/'layer_programs.json',dict(block_influence=bi,programs=selectors))
    baseline,guard=losses(lab,domain,guards);responses=[]
    for layer in range(len(bi)):
        with depth_variant(lab.model,[i for i in range(len(bi)) if i!=layer]):risk,g=losses(lab,domain,guards)
        row=dict(layer=layer,normalized_depth=layer/max(1,len(bi)-1),block_influence=bi[layer],
            domain_risk=risk,domain_risk_delta=risk-baseline,guard_norm_ce=sum(g['norm_ce'])/len(guards),
            guard_ce_delta=(sum(g['norm_ce'])-sum(guard['norm_ce']))/len(guards))
        responses.append(row)
        with open(a.output/'layer_responses.jsonl','a') as stream:stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    write_json(a.output/'complete.json',dict(status='programming_observations_complete',layers=len(bi),baseline_domain_risk=baseline,
        bi_vs_domain_damage_spearman=correlation(bi,[r['domain_risk_delta'] for r in responses]),
        bi_vs_guard_damage_spearman=correlation(bi,[r['guard_ce_delta'] for r in responses]),
        source_parameters=sum(p.numel() for p in lab.model.parameters()),
        scope='Real model calibration and causal deletion on training-derived development data; not official generation/quality confirmation. Per-layer correlations are descriptive, layers share inputs and are not independent samples.'))


if __name__=='__main__':main()
