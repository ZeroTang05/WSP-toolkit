"""Summarize frozen pilot results without feeding test outcomes back into search."""
import argparse
import json
import random
from pathlib import Path


def mean(xs):
    return sum(xs)/len(xs)


def paired_ci(differences):
    rng = random.Random(991)
    samples = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(3000))
    return [samples[75], samples[2924]]


def ranks(values):
    result = []
    for value in values:
        lower = sum(v < value for v in values)
        equal = sum(v == value for v in values)
        result.append(lower+(equal-1)/2)
    return result


def correlation(a, b):
    a, b = ranks(a), ranks(b)
    ma, mb = mean(a), mean(b)
    numerator = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    denom = (sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))**0.5
    return numerator/denom if denom else None


def summarize(run):
    baseline = json.loads((run/'original_final.json').read_text())
    results = []
    for path in sorted(run.glob('*_final.json')):
        value = json.loads(path.read_text())
        diffs = [int(r['exact_match'])-int(b['exact_match'])
                 for r,b in zip(value['generation']['rows'], baseline['generation']['rows'])]
        assert [r['id'] for r in value['generation']['rows']] == [r['id'] for r in baseline['generation']['rows']]
        item = dict(name=path.stem.removesuffix('_final'), model=value['model'],
            exact_match=value['generation']['exact_match'], exact_match_delta=mean(diffs),
            paired_bootstrap_delta_ci95=paired_ci(diffs), answer_nll=value['test']['answer_nll'],
            guard_nll=value['guard_test']['token_nll'], parameters=value['profile']['parameters'],
            forward_median_ms=value['profile']['median_ms'],
            peak_allocated_bytes=value['profile']['peak_allocated_bytes'])
        item['parameter_reduction'] = 1-item['parameters']/baseline['profile']['parameters']
        item['forward_latency_reduction'] = 1-item['forward_median_ms']/baseline['profile']['median_ms']
        item['peak_memory_reduction'] = 1-item['peak_allocated_bytes']/baseline['profile']['peak_allocated_bytes']
        patch_path = Path(value['model'])/'weightlab_patch.json'
        if patch_path.exists() and (run/'lora40_final.json').exists():
            patch = json.loads(patch_path.read_text())
            if Path(patch.get('base_model','')).name=='lora40':
                specialist = json.loads((run/'lora40_final.json').read_text())
                specialist_diffs = [int(r['exact_match'])-int(b['exact_match']) for r,b in
                    zip(value['generation']['rows'],specialist['generation']['rows'])]
                item['specialist_exact_match_delta'] = mean(specialist_diffs)
                item['specialist_delta_ci95'] = paired_ci(specialist_diffs)
        results.append(item)
    result = dict(results=results, note='Pilot only; synthetic arithmetic and tiny prose guard. No benchmark/general ability claim.')
    if (run/'sensitivity.json').exists():
        probe = json.loads((run/'probe.json').read_text())
        outcomes = json.loads((run/'sensitivity.json').read_text())['outcomes']
        assert len(outcomes) == len(probe['ablations'])
        target = [r['nll_delta'] for r in outcomes]
        result['diagnostic'] = dict(n_layers=len(outcomes), intervention='down projection scale=0.8',
            causal_ablation_signed_spearman=correlation([r['nll_delta'] for r in probe['ablations']], target),
            activation_importance_absolute_spearman=correlation([r['importance_mean'] for r in probe['statistics']], [abs(v) for v in target]),
            weight_norm_absolute_spearman=correlation([r['output_column_norm_mean'] for r in probe['statistics']], [abs(v) for v in target]),
            causal_ablation_absolute_spearman=correlation([abs(r['nll_delta']) for r in probe['ablations']], [abs(v) for v in target]),
            caveat='Different search subsets for prediction and outcome, but same model and task family. Descriptive only; not a matched-cost optimizer comparison.')
    (run/'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    summarize(parser.parse_args().run)
