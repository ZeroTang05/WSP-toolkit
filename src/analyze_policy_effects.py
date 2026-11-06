"""Full single-seed GSM policy panel: paired effects and proxy disagreement."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from analyze_model_trends import checked_scores,bootstrap_effect

STATES={'base':'19a714d92e0d239c121c','standard_lora':'cd4214c6b8a6fec102a7',
    'fixed':'49319db339a522243f38','random':'b729daaacf7588140f3e',
    'greedy':'c3f399cbb7e7b292944f','bo':'d4bd365ee3bd53423041'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);scores={};records=[]
    for method,state in STATES.items():
        path=a.root/'Qwen3-0.6B'/state/'gsm8k'
        values,hashes,settings,files=checked_scores(path,'gsm8k','Qwen3-0.6B')
        scores[method]=(values,hashes,settings,files)
        if method!='base' and (hashes,settings,files)!=scores['base'][1:]:raise ValueError('Changed matched evaluation')
        protocol=json.loads((path/'protocol.json').read_text())
        record=dict(method=method,state=state,correct=int(values.sum()),n=len(values),
            input_sha256={n:hashlib.sha256((path/n).read_bytes()).hexdigest()
                for n in ['results.json','protocol.json','idea4_gsm8k_cot_zeroshot_v1.samples.jsonl']})
        if protocol.get('plan'):
            plan=Path(protocol['plan']['file'])
            if hashlib.sha256(plan.read_bytes()).hexdigest()!=protocol['plan']['sha256']:raise ValueError('Changed plan')
            record['plan']=json.loads(plan.read_text())
        records.append(record)
        if method in ['greedy','bo']:
            out=a.output/method;out.mkdir(exist_ok=True)
            for f in path.iterdir():
                if f.suffix in ['.json','.jsonl']:shutil.copy2(f,out/f.name)
    comparisons=[]
    for method in ['greedy','bo']:
        for reference in ['base','standard_lora','fixed']:
            delta=scores[method][0]-scores[reference][0]
            comparisons.append(dict(method=method,reference=reference,improved=int((delta>0).sum()),
                regressed=int((delta<0).sum()),**bootstrap_effect(delta[None,:],resamples=5000)))
    result=dict(model='Qwen3-0.6B',seed=11,records=records,comparisons=comparisons,
        scope='Same frozen runtime/full official test. Single source seed, item-only intervals; descriptive comparisons, no multiplicity-adjusted claims. Greedy is single-axis probing, not full coordinate descent.')
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
