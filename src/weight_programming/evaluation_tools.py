"""Development set pairwise evaluation and preference score difference are performed via shared forwards."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import numpy as np

from .forward_service import paired_comparison
from .grading import extract


class BoxedExactGrader:
    """Use reproducible answer text matching in MCP worker threads."""

    identity = {"name": "boxed_exact", "version": 1}

    def __call__(self, item, text, finish_reason):
        prediction, boxes = extract(text)
        gold = item.get("answer")
        correct = None if gold is None else prediction is not None and prediction.strip() == gold.strip()
        return {"correct": correct, "prediction": prediction, "boxed": prediction is not None, "boxes": boxes}


def _cluster_interval(original, edited, seed, draws):
    """Resample by question to retain the correlation of repeated generation of the same question."""
    reference = {(row["id"], row["sample"]): row["correct"] for row in original}
    by_case = defaultdict(list)
    for row in edited:
        by_case[row["id"]].append(int(row["correct"]) - int(reference[(row["id"], row["sample"])]))
    effects = np.asarray([mean(by_case[key]) for key in sorted(by_case)])
    if effects.size < 2:
        return None
    sample = np.random.default_rng(seed).choice(effects, size=(draws, effects.size), replace=True)
    return np.quantile(sample.mean(1), [0.025, 0.975]).tolist()


def aligned_completion(backend, prefix, text):
    """Returns the answer tokens after the prompt word boundary for use in scoring and gradient attribution."""
    full = backend.encode(text)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError("Use a prompt boundary that aligns with complete tokens")
    return full[len(prefix):]


class EvaluationTools:
    """Generate reproducible reports from authorized data and specified model versions."""

    def __init__(self, forward, versions, artifacts, run_forward):
        self.forward = forward
        self.versions = versions
        self.artifacts = artifacts
        self.run_forward = run_forward

    def evaluate_variants(self, baseline_variant_id, candidate_variant_ids, dataset_id, case_ids,
                          max_new_tokens, samples=1, metric="boxed_exact", bootstrap_draws=1000,
                          *, comparison=None):
        """Evaluate baselines and candidates on the same problem and generation seeds."""
        practice = self.forward.practice
        if practice.name != dataset_id or practice.role != "development" or self.forward.grader is None:
            raise ValueError("An explicitly registered development dataset is required")
        if metric != self.forward.grader.identity["name"]:
            raise ValueError("Requested metric differs from the registered grader")
        if any(not isinstance(practice.rows[key].get("answer"), str) or not practice.rows[key]["answer"].strip()
               for key in case_ids):
            raise ValueError("Evaluation cases require a canonical answer")
        variants = [baseline_variant_id, *candidate_variant_ids]
        if len(set(variants)) != len(variants):
            raise ValueError("Baseline and candidates must be distinct")
        candidate_service = comparison or self
        if (candidate_service.forward.practice.identity != practice.identity
                or candidate_service.forward.policy != self.forward.policy
                or candidate_service.forward.grader is None
                or candidate_service.forward.grader.identity != self.forward.grader.identity):
            raise ValueError("Comparison requires the same registered data, generation policy and metric")
        conditions = [(baseline_variant_id, self)] + [(variant, candidate_service) for variant in candidate_variant_ids]
        for variant, service in conditions:
            service.versions.variant(variant)
        summaries, records = [], {}
        for variant, service in conditions:
            original = variant == service.versions.base["variant_id"]
            label = "original" if original else "edited"
            request = {"type": "generate", "model": label, "problems": case_ids, "n": samples,
                       "max_tokens": max_new_tokens}
            snapshot = None if original else service.versions.snapshot(variant)
            result = service.forward.request(request, variant_snapshot=snapshot)
            path = Path(result["artifact_dir"]) / result["generation_files"][label]
            with path.open(encoding="utf-8") as stream:
                records[variant] = [json.loads(line) for line in stream]
            artifacts = service.artifacts.publish_forward(result, {})
            summary = result["summaries"][label]
            seconds = summary["generation_seconds"]
            summaries.append({"variant_id": variant, "run_id": f"request_{result['attempt']:04d}",
                              "workspace": "primary" if service is self else "comparison",
                              "predictions_artifact_id": artifacts[label]["artifact_id"], **summary,
                              "wall_seconds": result["elapsed_seconds"],
                              "output_tokens_per_second": summary["output_tokens"] / seconds if seconds else None})
        baseline = records[baseline_variant_id]
        comparisons = []
        for candidate in candidate_variant_ids:
            rows = records[candidate]
            compared = paired_comparison(baseline, rows, require_same_prompt_format=candidate_service is self)
            interval = _cluster_interval(baseline, rows, self.forward.policy.seed, bootstrap_draws)
            comparisons.append({"candidate_variant_id": candidate, **compared, "paired_bootstrap_ci95": interval})
        return {"dataset_id": dataset_id, "dataset_sha256": practice.identity["sha256"], "split": practice.role,
                "metric": metric, "case_ids": case_ids, "samples_per_case": samples,
                "baseline_variant_id": baseline_variant_id, "summaries": summaries, "comparisons": comparisons,
                "same_rendered_prompt_required": candidate_service is self}

    def score_preferences(self, variant_id, reference_variant_id, pairs, aggregation="sum"):
        """Compare answer log probabilities and explicitly check tokenization boundaries."""
        backend = self.forward.backend
        texts, starts = [], []
        for pair in pairs:
            prompt = pair["prompt"]
            prefix = backend.encode(prompt)
            if pair["chosen"] == pair["rejected"]:
                raise ValueError("Chosen and rejected answers must differ")
            for side in ("chosen", "rejected"):
                text = prompt + pair[side]
                suffix = aligned_completion(backend, prefix, text)
                texts.append(text)
                starts.append((len(prefix), len(prefix) + len(suffix)))
        if len({pair["id"] for pair in pairs}) != len(pairs):
            raise ValueError("Preference pair IDs must be unique")

        def scores(model_id):
            run = self.run_forward(model_id, "teacher_forced", texts=texts, token_logprobs=True)
            data, _ = self.artifacts.array(run["artifacts"]["token_logprobs"]["artifact_id"])
            values = [data[index, start:end].astype(np.float64) for index, (start, end) in enumerate(starts)]
            return run["run_id"], [float(value.sum() if aggregation == "sum" else value.mean()) for value in values]

        run_id, candidate = scores(variant_id)
        reference_run, reference = (run_id, candidate) if variant_id == reference_variant_id else scores(reference_variant_id)
        records = []
        for i, pair in enumerate(pairs):
            index = 2 * i
            margin = candidate[index] - candidate[index + 1]
            reference_margin = reference[index] - reference[index + 1]
            records.append({"pair_id": pair["id"], "chosen_tokens": starts[index][1] - starts[index][0],
                            "rejected_tokens": starts[index + 1][1] - starts[index + 1][0],
                            "chosen_score": candidate[index], "rejected_score": candidate[index + 1],
                            "margin": margin, "reference_margin": reference_margin,
                            "relative_margin": margin - reference_margin})
        return {"variant_id": variant_id, "reference_variant_id": reference_variant_id, "aggregation": aggregation,
                "run_id": run_id, "reference_run_id": reference_run, "pairs": len(records),
                "mean_relative_margin": mean(row["relative_margin"] for row in records),
                "report_artifact_id": self.artifacts.save_records("preferences", records), "preview": records[:5]}
