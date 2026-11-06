"""Freeze and run collaboration evidence before any new role extraction."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

FILES=['run_hotpot_collaboration.py','hotpot_collaboration.py','analyze_hotpot_collaboration.py','weightlab.py']
MODES=['direct','collaborate','self_refine','independent_vote','no_communication']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--confirm-if-positive',action='store_true');p.add_argument('--preparation-pid',type=int)
    a=p.parse_args();root=a.root.resolve()
    view=root/'benchmarks/views/hotpot-collaboration-v1'
    dest=root/'benchmarks/hotpot-collaboration-v1'/a.model.name;dest.mkdir(parents=True,exist_ok=True)
    lock=open(dest/'runner.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources={n:(root/'src'/n).read_bytes() for n in FILES}
    signature=hashlib.sha256(b''.join(n.encode()+b for n,b in sorted(sources.items()))).hexdigest()
    frozen=root/'research/frozen'/('hotpot-collaboration-'+signature)/'code';frozen.mkdir(parents=True,exist_ok=True)
    for name,data in sources.items():(frozen/name).write_bytes(data)
    while not (view/'manifest.json').exists():
        proc=Path('/proc')/str(a.preparation_pid)
        if a.preparation_pid is None or not (proc/'cmdline').exists():raise FileNotFoundError('Preparation terminated before view completion')
        command=(proc/'cmdline').read_bytes()
        if b'prepare_hotpot_collaboration.py' not in command:raise RuntimeError('Preparation PID no longer matches expected command')
        print(json.dumps(dict(event='waiting_for_live_preparation',pid=a.preparation_pid,frozen_source=signature)),flush=True)
        time.sleep(15)
    spec=dict(version='hotpot-collaboration-v1',model=str(a.model.resolve()),source_sha256=signature,
        modes=MODES,seeds=[11,22,33],budgets=[384,768],development=256,official_dev=7405,
        demonstrations='Same first two disjoint training-calibration examples in every call and condition; full candidate documents and labeled output.',
        view_manifest_sha256=hashlib.sha256((view/'manifest.json').read_bytes()).hexdigest(),
        confirm_if_positive=a.confirm_if_positive,selection='Development-only highest collaboration minus strongest single/refine/vote EM; tie favors lower budget.',
        hypothesis='Collaboration versus development-selected strong baseline at development-selected budget; official lower paired seed/item CI >0 required.',
        scope='No role training. Equal output-token caps only; separately report actual cost. Official dev is confirmation, never named test.')
    path=dest/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Changed frozen specification')
    path.write_text(json.dumps(spec,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def run(partition,budget,seed,mode):
        out=dest/partition/f'budget{budget}'/f'seed{seed}'/mode
        if (out/'complete.json').exists():return
        log=dest/f'{partition}_{budget}_{seed}_{mode}.log'
        print(json.dumps(dict(event='start',partition=partition,budget=budget,seed=seed,mode=mode)),flush=True)
        with open(log,'a') as stream:
            subprocess.run([str(root/'.venv/bin/python'),'-u',str(frozen/'hotpot_collaboration.py'),
                '--model',str(a.model),'--view',str(view),'--partition',partition,'--mode',mode,'--budget',str(budget),
                '--seed',str(seed),'--output',str(out)],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(event='complete',partition=partition,budget=budget,seed=seed,mode=mode)),flush=True)
    summaries=[]
    import sys
    sys.path.insert(0,str(frozen));from analyze_hotpot_collaboration import analyze
    for budget in spec['budgets']:
        for seed in spec['seeds']:
            for mode in MODES:run('development',budget,seed,mode)
        result=analyze(dest,'development',budget);(dest/f'development_{budget}_analysis.json').write_text(json.dumps(result,indent=2));summaries.append(result)
    candidates=[]
    for result in summaries:
        scores={r['mode']:r['answer_em'] for r in result['summaries']}
        baseline=max(['direct','self_refine','independent_vote'],key=lambda m:scores[m])
        candidates.append(dict(budget=result['budget'],baseline=baseline,development_gain=scores['collaborate']-scores[baseline]))
    selected=max(candidates,key=lambda c:(c['development_gain'],-c['budget']))
    (dest/'selected_confirmation.json').write_text(json.dumps(dict(selected=selected,candidates=candidates,rule=spec['selection']),indent=2))
    if selected['development_gain']<=0 or not a.confirm_if_positive:
        (dest/'development_complete.json').write_text(json.dumps(dict(status='development_no_positive_gain' if selected['development_gain']<=0 else 'development_complete_confirmation_not_requested',selected=selected),indent=2));return
    confirmation_modes=['collaborate',selected['baseline'],'no_communication']
    for seed in spec['seeds']:
        for mode in confirmation_modes:run('official_dev',selected['budget'],seed,mode)
    result=analyze(dest,'official_dev',selected['budget'],modes=confirmation_modes)
    primary=next(r for r in result['collaboration_effects'] if r['reference']==selected['baseline'])
    result['primary_reference']=selected['baseline'];result['collaboration_gate_passed']=primary['seed_item_bootstrap95'][0]>0
    (dest/'confirmation_analysis.json').write_text(json.dumps(result,indent=2))
    (dest/'matrix_complete.json').write_text(json.dumps(dict(status='collaboration_confirmation_complete',gate_passed=result['collaboration_gate_passed'],
        note='Role extraction may be designed from positive evidence; it is not implemented or proven by this gate.'),indent=2))


if __name__=='__main__':main()
