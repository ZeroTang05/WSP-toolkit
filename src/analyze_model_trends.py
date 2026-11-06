"""Matched-item, seed-aware public benchmark effects and model contrasts.

Training seeds are paired within a model/method comparison and resampled
independently across different models. Test items remain paired across models.
The small number of model families is descriptive, not a population sample.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from analyze_public import read_scores

TASKS={
    'gsm8k':('idea4_gsm8k_cot_zeroshot_v1','flexible-extract','exact_match',1319,'gsm8k',32),
    'arc':('idea4_arc_challenge_v1','none','acc_norm',1172,'guards',8),
    'hellaswag':('idea4_hellaswag_v1','none','acc_norm',10042,'guards',8)}
SEEDS={11,22,33}
MODELS={'Qwen3-0.6B':('qwen3',596049920),'Qwen3-1.7B':('qwen3',1720574976),
    'SmolLM2-360M-Instruct':('smollm2',361821120),'SmolLM2-1.7B-Instruct':('smollm2',1711376384)}


def bootstrap_effect(left,right=None,resamples=2000,seed=20261106):
    """Rows=training seeds, columns=identical ordered public test items.

    For a model contrast, independently resample seed rows on each side and
    use the same sampled item columns. A single available seed yields item-only
    uncertainty; callers must not interpret it as training-seed robustness.
    """
    left=np.asarray(left,dtype=float)
    right=None if right is None else np.asarray(right,dtype=float)
    if left.ndim!=2 or not left.shape[0] or not left.shape[1] or not np.isfinite(left).all():
        raise ValueError('Expected finite seed-by-item effects')
    if right is not None and (right.ndim!=2 or right.shape[1]!=left.shape[1] or not right.shape[0] or not np.isfinite(right).all()):
        raise ValueError('Mismatched item columns')
    if resamples<100:raise ValueError('Too few bootstrap draws')
    rng=np.random.default_rng(seed);draws=[]
    for _ in range(resamples):
        effect=left[rng.integers(len(left),size=len(left))].mean(0)
        if right is not None:effect=effect-right[rng.integers(len(right),size=len(right))].mean(0)
        items=rng.integers(left.shape[1],size=left.shape[1]);draws.append(float(effect[items].mean()))
    return dict(estimate=float(left.mean()-(right.mean() if right is not None else 0)),
        bootstrap95=np.quantile(draws,[.025,.975]).tolist(),resamples=resamples,seed=seed,
        item_pairing='same item resample on both sides',
        seed_pairing='independent across models' if right is not None else 'resampled paired seed effect rows',
        uncertainty='seed-and-item empirical bootstrap' if len(left)>1 and (right is None or len(right)>1)
            else 'at least one side has one seed; does not establish training-seed robustness')


def checked_scores(path,task,model):
    name,filter_name,metric,expected,_,batch=TASKS[task]
    protocol=json.loads((path/'protocol.json').read_text())
    complete=json.loads((path/'complete.json').read_text())
    fingerprint=json.loads((path/'model_fingerprint.json').read_text())
    if Path(protocol['model']).name!=model or fingerprint['source']!=protocol['model']:
        raise ValueError('Model index/recorded source mismatch')
    if complete['status']!='full_complete' or protocol['smoke_limit'] is not None:
        raise ValueError('Smoke/partial results are not public confirmation')
    if protocol['batch_size']!=batch:raise ValueError('Different evaluation batch')
    if name not in protocol['tasks']:raise ValueError('Wrong task')
    runners=list(path.glob('runner_source_*.py'))
    if len(runners)!=1 or hashlib.sha256(runners[0].read_bytes()).hexdigest()!=complete['code_sha256']:
        raise ValueError('Changed benchmark runner snapshot')
    helpers=path/('helpers_'+complete['code_sha256'])
    for helper,expected_sha in protocol['implementation_files'].items():
        if hashlib.sha256((helpers/helper).read_bytes()).hexdigest()!=expected_sha:
            raise ValueError('Changed benchmark helper snapshot')
    scores=read_scores(path,name,filter_name,metric)
    if set(scores)!=set(range(expected)):raise ValueError('Incomplete official document IDs')
    settings={k:protocol[k] for k in ['batch_size','max_new_tokens','chat_template','enable_thinking',
        'num_fewshot','evaluation_seed','harness_version','task_files','implementation_files']}
    settings['runtime']={k:complete[k] for k in ['torch','transformers','gpu','code_sha256']}
    # MC group lists two YAMLs; all states in this analysis must use that same
    # group protocol. Comparing artifacts with changed helper implementations
    # requires an explicit separate study, even if their labels match.
    values=np.array([scores[i]['score'] for i in range(expected)])
    hashes=tuple(scores[i]['doc_hash'] for i in range(expected))
    return values,hashes,settings,fingerprint['files']


def analyze(index_path,capacity_index=None,tasks=('gsm8k','arc','hellaswag'),resamples=2000):
    entries=json.loads(index_path.read_text());cache={};references={};summaries=[];matrices={};bindings={}
    for row in entries:
        if row['model'] not in MODELS:raise ValueError('Unknown model '+row['model'])
        if row['method']=='base':
            if row['model'] in references:raise ValueError('Duplicate model baseline')
            references[row['model']]=row
    all_entries=[dict(**r,capacity='C1') for r in entries if r['method']!='base']
    if capacity_index:
        for row in json.loads(capacity_index.read_text()):
            all_entries.append(dict(**row,method='standard_lora',capacity='C2'))
    def read(path,task,model):
        key=(str(path),task,model)
        if key not in cache:cache[key]=checked_scores(Path(path),task,model)
        values,hashes,settings,files=cache[key]
        if model in bindings and files!=bindings[model]:raise ValueError('Different recorded base weights/tokenizer')
        bindings[model]=files
        return values,hashes,settings
    grouped={}
    for row in all_entries:
        key=(row['capacity'],row['model'],row['method'])
        if row['seed'] not in SEEDS:raise ValueError('Unexpected training seed')
        bucket=grouped.setdefault(key,{})
        if row['seed'] in bucket:raise ValueError('Duplicate method/seed entry')
        bucket[row['seed']]=row
    for (capacity,model,method),seeds in sorted(grouped.items()):
        if model not in references:raise ValueError('Missing same-model base')
        for task in tasks:
            group=TASKS[task][4];base,hashes,settings=read(references[model][group],task,model)
            effects=[];accuracies={};paths=[]
            for seed,row in sorted(seeds.items()):
                values,other_hashes,other_settings=read(row[group],task,model)
                if hashes!=other_hashes:raise ValueError('Method/base document identity mismatch')
                if settings!=other_settings:raise ValueError('Method/base execution protocol mismatch')
                effects.append(values-base);accuracies[str(seed)]=float(values.mean());paths.append(row[group])
            effects=np.asarray(effects)
            summary=dict(capacity=capacity,model=model,family=MODELS[model][0],parameters=MODELS[model][1],
                method=method,task=task,n=len(base),seeds=sorted(seeds),missing_seeds=sorted(SEEDS-set(seeds)),
                base_accuracy=float(base.mean()),seed_accuracies=accuracies,mean_accuracy=float(np.mean(list(accuracies.values()))),
                effect_vs_base=bootstrap_effect(effects,resamples=resamples),paths=paths,
                reused_identical_C1_condition=all(r.get('reused_C1_identical_condition',False) for r in seeds.values()))
            summaries.append(summary)
            matrices[(capacity,method,task,model)]=(effects,hashes,settings,set(seeds))
    contrasts=[];capacity_contrasts=[]
    for capacity,method,task in sorted({key[:3] for key in matrices}):
        models=sorted(key[3] for key in matrices if key[:3]==(capacity,method,task))
        for first,second in itertools.combinations(models,2):
            left,lh,ls,lseeds=matrices[(capacity,method,task,first)]
            right,rh,rs,rseeds=matrices[(capacity,method,task,second)]
            # Do not promote single-seed observations to a family/scale result.
            if lseeds!=SEEDS or rseeds!=SEEDS:continue
            if lh!=rh or ls!=rs:raise ValueError('Across-model benchmark protocol mismatch')
            ratio=max(MODELS[first][1],MODELS[second][1])/min(MODELS[first][1],MODELS[second][1])
            relation=('same_family_different_scale' if MODELS[first][0]==MODELS[second][0] else
                'different_family_near_matched_scale' if ratio<=1.1 else 'different_family_and_scale')
            contrasts.append(dict(capacity=capacity,method=method,task=task,model_a=first,model_b=second,
                relation=relation,parameter_ratio=ratio,gain_a_minus_gain_b=bootstrap_effect(left,right,resamples=resamples),
                mean_gain_same_sign=bool(np.sign(left.mean())==np.sign(right.mean())),
                seed_gains_a=left.mean(1).tolist(),seed_gains_b=right.mean(1).tolist()))
    for model in MODELS:
        for task in tasks:
            keys=[(capacity,'standard_lora',task,model) for capacity in ['C1','C2']]
            if any(key not in matrices for key in keys):continue
            left,lh,ls,lseeds=matrices[keys[0]];right,rh,rs,rseeds=matrices[keys[1]]
            if lseeds!=SEEDS or rseeds!=SEEDS:continue
            if lh!=rh or ls!=rs:raise ValueError('Capacity-control protocol mismatch')
            capacity_contrasts.append(dict(model=model,task=task,
                c2_minus_c1=bootstrap_effect(right-left,resamples=resamples),
                seed_pairing='same-seed training data sequence, paired within this model',
                reused_identical_condition=bool(np.array_equal(left,right))))
    expected=[(capacity,model,method,task) for capacity,methods in [('C1',['standard_lora','fixed','random','greedy','bo','llm'])]+([('C2',['standard_lora'])] if capacity_index else [])
        for model in MODELS for method in methods for task in tasks]
    present={(r['capacity'],r['model'],r['method'],r['task']) for r in summaries}
    missing_conditions=[dict(capacity=c,model=m,method=p,task=t) for c,m,p,t in expected if (c,m,p,t) not in present]
    return dict(status='available_model_trend_analysis',index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        capacity_index_sha256=hashlib.sha256(capacity_index.read_bytes()).hexdigest() if capacity_index else None,
        summaries=summaries,contrasts=contrasts,capacity_contrasts=capacity_contrasts,
        missing_conditions=missing_conditions,expected_condition_task_count=len(expected),
        complete_three_seed_conditions=sum(not r['missing_seeds'] for r in summaries),
        effect_units='fraction correct; multiply by 100 for absolute percentage points',
        scope='Descriptive finite-model contrasts, not causal family/architecture effects or equivalence tests. Only complete three-seed pairs enter model contrasts. CIs are unadjusted empirical seed/item bootstrap; three seeds weakly estimate seed variability. Shared/reused states are not independent repetitions. Training-data/pretraining contamination and generalization beyond benchmarks remain unmeasured.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True)
    p.add_argument('--capacity-index',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',choices=list(TASKS),default=list(TASKS));p.add_argument('--resamples',type=int,default=2000)
    a=p.parse_args();result=analyze(a.index,a.capacity_index,a.tasks,a.resamples)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(summaries=len(result['summaries']),contrasts=len(result['contrasts']),scope=result['scope']),indent=2))
