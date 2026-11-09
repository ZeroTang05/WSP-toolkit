"""通过共用登记表提供激活替换、消融与站点扫描。"""

from statistics import mean
from time import perf_counter
import math

import numpy as np
import torch

from .patching import paired_interventions, prepare_pair
from .tool_registry import ToolSpec, object_schema


_STRING = {"type": "string", "minLength": 1}
_INDEX = {"type": "integer", "minimum": 0}
_POSITIONS = {"type": "array", "items": _INDEX, "minItems": 1, "uniqueItems": True}
_PAIR = object_schema({"clean_text": _STRING, "corrupt_text": _STRING, "positions": _POSITIONS,
                       "target_token": _INDEX, "alternative_token": _INDEX, "query_position": _INDEX},
                      "clean_text", "corrupt_text", "positions", "target_token", "alternative_token")
_METRIC = {"enum": ["logit_difference", "target_probability", "target_log_probability"]}


class InterventionTools:
    """沿用同一候选版本、预算账本和产物服务。"""

    def __init__(self, forward, versions, artifacts, sites):
        self.forward = forward
        self.versions = versions
        self.artifacts = artifacts
        self.sites = sites

    def specs(self):
        """登记可以显式授权的两个工具。"""
        site_ids = {"type": "array", "items": _STRING, "uniqueItems": True}
        cases = {"type": "array", "items": _PAIR, "minItems": 1}
        common = {"variant_id": _STRING, "site_id": _STRING, "pair": _PAIR,
                  "channels": _POSITIONS, "metric": _METRIC}
        usual = object_schema({**common, "method": {"enum": ["replace", "ablation"]}}, "variant_id", "site_id", "pair")
        directed = object_schema({**common, "method": {"const": "direction"}, "strength": {"type": "number"},
                                  "direction": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 32768},
                                  "direction_artifact_id": _STRING}, "variant_id", "site_id", "pair", "method", "strength")
        directed["oneOf"] = [{"required": ["direction"]}, {"required": ["direction_artifact_id"]}]
        return (
            ToolSpec("run_intervention", "Measure activation replacement, zero ablation or direction injection.",
                     {"type": "object", "oneOf": [usual, directed]},
                     self.run_intervention, permission="intervene",
                     output_schema={"type": "object", "required": ["run_id", "artifact_id", "effects"]}),
            ToolSpec("localize_behavior", "Scan observation sites using paired inputs and rank their measured effects.",
                     object_schema({"variant_id": _STRING, "site_ids": {**site_ids, "minItems": 1},
                                    "pairs": cases, "review_pairs": cases, "control_site_ids": site_ids,
                                    "metric": _METRIC, "method": {"const": "exact_scan"},
                                    "forward_budget": {"type": "integer", "minimum": 3, "maximum": 256}},
                                   "variant_id", "site_ids", "pairs"),
                     self.localize_behavior, permission="intervene",
                     output_schema={"type": "object", "required": ["run_id", "artifact_id", "ranked_sites"]}),
        )

    def _site(self, site_id, channels=None):
        site = self.sites.get(site_id)
        kinds = {"hidden": "residual", "mlp": "mlp_input"}
        if site is None or site["kind"] not in kinds:
            raise ValueError(f"Site does not support activation intervention: {site_id}")
        if channels is not None and any(type(i) is not int or not 0 <= i < site["width"] for i in channels):
            raise ValueError(f"Channel outside observation site: {site_id}")
        return {"kind": kinds[site["kind"]], "layer": site["index"], "channels": channels}

    def _run(self, variant_id, ids, pairs, metric, intervention, forward_budget, *, channels=None,
             discovery_pairs=None, request_type, shift=None, direction_evidence=None):
        backend, policy = self.forward.backend, self.forward.policy
        specs = [self._site(site_id, channels) for site_id in ids]
        pairs_prepared = [prepare_pair(backend, policy, pair["clean_text"], pair["corrupt_text"],
                                       pair["target_token"], pair["alternative_token"], pair.get("query_position"))
                          for pair in pairs]
        calls_per_pair = len(ids) + 2
        tokens = sum(len(prepared[0]) for prepared in pairs_prepared) * calls_per_pair
        if len(pairs) * calls_per_pair > forward_budget or tokens > policy.max_total_text_tokens:
            raise ValueError("Intervention exceeds the declared forward or token budget")
        for pair, (clean_ids, _, _) in zip(pairs, pairs_prepared):
            if any(type(position) is not int or not 0 <= position < len(clean_ids) for position in pair["positions"]):
                raise ValueError("Intervention position is outside the paired inputs")
            if len(clean_ids) * backend.vocab_size * 4 > min(policy.max_return_bytes, 512_000_000):
                raise ValueError("Intervention score tensor exceeds the memory budget")
        snapshot = self.versions.snapshot(variant_id)
        request = {"type": request_type, "metric": metric, "method": intervention, "site_ids": ids,
                   "pairs": pairs, "channels": channels, "discovery_pairs": discovery_pairs}
        if direction_evidence is not None:
            request["direction"] = direction_evidence
        ledger = self.forward.ledger
        attempt = ledger.begin(request, variant_id)
        run_id = f"request_{attempt:04d}"
        start = perf_counter()
        forwards = 0

        def read(*arguments):
            nonlocal forwards
            forwards += 1
            return backend._read(*arguments)

        try:
            results = []
            for pair, prepared in zip(pairs, pairs_prepared):
                sites = [dict(spec, positions=pair["positions"], method=intervention, shift=shift) for spec in specs]
                results.append(paired_interventions(backend, snapshot, prepared, sites,
                                                    pair["target_token"], pair["alternative_token"], read))
            effects = np.asarray([[item["effects"][metric]["patched_minus_corrupt"] for item in row]
                                  for row in results], dtype=np.float32)
            artifact_id = self.artifacts.save_array("intervention_effects", effects, ["pair", "site"], run_id=run_id,
                                                    metric=metric, site_ids=ids, variant_id=variant_id)
            usage = {"elapsed_seconds": perf_counter() - start, "executed_model_forwards": forwards,
                     "input_tokens": tokens, "artifact_id": artifact_id}
            ledger.finish(attempt, True, usage)
        except Exception as error:
            ledger.finish(attempt, False, {"error_type": type(error).__name__, "error": str(error)[:500],
                                           "attempted_model_forwards": forwards})
            raise
        return {"run_id": run_id, "variant_id": variant_id, "metric": metric,
                "artifact_id": artifact_id, "usage": usage}, results

    def _direction(self, site_id, channels, strength, direction, artifact_id):
        """把用户方向变成选定通道上的单位方向，再按强度缩放。"""
        self._site(site_id, channels)
        if strength is None or not math.isfinite(strength) or (direction is None) == (artifact_id is None):
            raise ValueError("Direction injection needs one vector and a finite strength")
        if artifact_id is not None:
            array, metadata = self.artifacts.array(artifact_id)
            if metadata["axes"] != ["channel"] or metadata.get("site_id", site_id) != site_id:
                raise ValueError("Direction artifact must have the selected site's channel axis")
            direction = array.copy()
        vector = torch.as_tensor(direction, dtype=torch.float64)
        if vector.shape != (self.sites[site_id]["width"],) or not torch.isfinite(vector).all():
            raise ValueError("Direction must contain one finite value per channel")
        selected = slice(None) if channels is None else channels
        norm = float(vector[selected].norm())
        if not 0 < norm < math.inf:
            raise ValueError("Direction must have a finite nonzero norm on the selected channels")
        shift = torch.zeros_like(vector, dtype=torch.float32)
        shift[selected] = ((vector[selected] / norm) * strength).float()
        if not torch.isfinite(shift).all():
            raise ValueError("Scaled direction cannot be represented at runtime precision")
        evidence = {"strength": strength, "input_norm": norm, "artifact_id": artifact_id}
        if artifact_id is None:
            evidence["values"] = vector.tolist()
        return shift, evidence

    def run_intervention(self, variant_id, site_id, pair, method="replace", metric="logit_difference", channels=None,
                         direction=None, direction_artifact_id=None, strength=None):
        """返回一次干预的三个条件与差值。"""
        shift = evidence = None
        if method == "direction":
            shift, evidence = self._direction(site_id, channels, strength, direction, direction_artifact_id)
        result, rows = self._run(variant_id, [site_id], [pair], metric, method, 3, channels=channels,
                                 request_type="run_intervention", shift=shift, direction_evidence=evidence)
        if evidence is not None:
            result["direction"] = {key: value for key, value in evidence.items() if key != "values"}
        return {**result, "site_id": site_id, "method": method, **rows[0][0]}

    def localize_behavior(self, variant_id, site_ids, pairs, metric="logit_difference", method="exact_scan",
                          review_pairs=None, control_site_ids=(), forward_budget=32):
        """按发现集平均效应排序，同时报告对照和复核效果。"""
        review_pairs = review_pairs or []
        ids = [*site_ids, *control_site_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate and control sites must be distinct")
        summary, rows = self._run(variant_id, ids, [*pairs, *review_pairs], metric, "replace", forward_budget,
                                  discovery_pairs=len(pairs), request_type="localize_behavior")
        discovery, review = rows[:len(pairs)], rows[len(pairs):]
        ranked = []
        for index, site_id in enumerate(ids):
            effects = [row[index]["effects"][metric]["patched_minus_corrupt"] for row in discovery]
            review_effects = [row[index]["effects"][metric]["patched_minus_corrupt"] for row in review]
            ranked.append({"site_id": site_id, "mean_effect": mean(effects), "pairs": len(effects),
                           "positive_pairs": sum(value > 0 for value in effects),
                           "review_mean_effect": mean(review_effects) if review_effects else None})
        return {**summary, "method": method, "site_ids": ids, "discovery_pairs": len(pairs),
                "review_pairs": len(review_pairs), "controls": ranked[len(site_ids):],
                "ranked_sites": sorted(ranked[:len(site_ids)], key=lambda item: (-item["mean_effect"], item["site_id"]))}
