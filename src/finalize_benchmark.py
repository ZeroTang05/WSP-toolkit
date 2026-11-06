"""Validate and finalize an existing harness output; never regenerate predictions.

Used after the initial wrapper confused per-filter log records with unique
questions. Original metrics, response cache and raw samples remain unchanged.
"""
import argparse
import json
from pathlib import Path
import math


def finalize(directory):
    results=json.loads((directory/'results.json').read_text())
    counts={};raw_counts={};checks={}
    expected={'idea4_gsm8k_cot_zeroshot_v1':1319,'idea4_arc_challenge_v1':1172,'idea4_hellaswag_v1':10042}
    for task,metrics in results['results'].items():
        with open(directory/(task+'.samples.jsonl')) as stream:rows=[json.loads(line) for line in stream]
        ids={row['doc_id'] for row in rows}
        if ids!=set(range(expected[task])):raise AssertionError('Missing or duplicated benchmark question IDs')
        counts[task]=len(ids);raw_counts[task]=len(rows)
        filters={row['filter'] for row in rows}
        recomputed={}
        for filter_name in filters:
            group=[r for r in rows if r['filter']==filter_name]
            if len(group)!=expected[task] or len({r['doc_id'] for r in group})!=expected[task]:
                raise AssertionError('Incomplete per-filter results')
            for key,value in metrics.items():
                if key.endswith(','+filter_name) and '_stderr,' not in key:
                    metric=key.split(',')[0]
                    actual=sum(float(r[metric]) for r in group)/len(group)
                    if not math.isclose(actual,value,rel_tol=1e-10,abs_tol=1e-12):raise AssertionError('Metric mismatch')
                    recomputed[key]=actual
        if task=='idea4_gsm8k_cot_zeroshot_v1':
            cfg=results['configs'][task]
            raw=Path(cfg['dataset_kwargs']['data_files']['test'])
            official=[json.loads(line) for line in open(raw)]
            if len(official)!=expected[task]:raise AssertionError('Wrong official dataset size')
            if any(r['doc']!=official[r['doc_id']] for r in rows):raise AssertionError('Logged question/label differs from official input')
        checks[task]=dict(filters=sorted(filters),metrics_recomputed=recomputed)
    source_path=directory/'execution_provenance.json'
    source=json.loads(source_path.read_text()) if source_path.exists() else {}
    complete=dict(status='full_complete',samples=counts,raw_sample_records=raw_counts,
        finalized_from_existing_raw_predictions=True,validation_checks=checks,
        code_sha256=source.get('executed_runner_sha256'),
        note='Corrected unique-question validation after initial per-filter record-count assertion; no new inference or metric change.')
    if (directory/'complete.json').exists():raise FileExistsError('Completion evidence already exists')
    (directory/'complete.json').write_text(json.dumps(complete,indent=2))
    print(json.dumps(complete,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('directory',type=Path)
    finalize(p.parse_args().directory)
