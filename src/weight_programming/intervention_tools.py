"""通过共用登记表提供激活替换、消融与站点扫描。"""

from importlib.util import find_spec
import math
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from .attribution_patching import atp_star_scores
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

    def __init__(self, forward, versions, artifacts, sites, *, allow_gradients=False):
        self.forward = forward
        self.versions = versions
        self.artifacts = artifacts
        self.sites = sites
        self.has_leace = find_spec("concept_erasure") is not None
        self.has_atp_star = (allow_gradients and forward.backend.identity["model_type"] in ("llama", "qwen3")
                             and forward.backend.num_layers >= 2)

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
        methods = [usual, directed]
        if self.has_leace:
            methods.append(object_schema({**{key: value for key, value in common.items() if key != "channels"},
                "method": {"const": "concept_erasure"}, "eraser_artifact_id": _STRING,
                "control_seed": {**_INDEX, "maximum": 2**32 - 1}},
                "variant_id", "site_id", "pair", "method", "eraser_artifact_id"))
        scan_fields = {"variant_id": _STRING, "site_ids": {**site_ids, "minItems": 1}, "pairs": cases,
                       "review_pairs": cases, "control_site_ids": site_ids, "metric": _METRIC,
                       "forward_budget": {"type": "integer", "minimum": 3, "maximum": 256}}
        scans = [object_schema({**scan_fields, "method": {"const": "exact_scan"}}, "variant_id", "site_ids", "pairs")]
        if self.has_atp_star:
            scans.append(object_schema({**scan_fields, "method": {"const": "atp_star"},
                "verify_top_k": {"type": "integer", "minimum": 1, "maximum": 32},
                "backward_budget": {"type": "integer", "minimum": 1, "maximum": 512}},
                "variant_id", "site_ids", "pairs", "method"))
        return (
            ToolSpec("run_intervention", "Measure a temporary change to activations at a selected site.",
                     {"type": "object", "oneOf": methods},
                     self.run_intervention, permission="intervene",
                     output_schema={"type": "object", "required": ["run_id", "artifact_id", "effects"]}),
            ToolSpec("localize_behavior", "Rank sites with exact interventions or an AtP* gradient prefilter.",
                     {"type": "object", "oneOf": scans},
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

    def _prepare_pairs(self, pairs):
        """共用配对分词、位置和输出内存校验。"""
        backend, policy = self.forward.backend, self.forward.policy
        prepared = []
        for pair in pairs:
            record = prepare_pair(backend, policy, pair["clean_text"], pair["corrupt_text"],
                                  pair["target_token"], pair["alternative_token"], pair.get("query_position"))
            length = len(record[0])
            if any(type(position) is not int or not 0 <= position < length for position in pair["positions"]):
                raise ValueError("Intervention position is outside the paired inputs")
            if length * backend.vocab_size * 4 > min(policy.max_return_bytes, 512_000_000):
                raise ValueError("Intervention score tensor exceeds the memory budget")
            prepared.append(record)
        return prepared

    def _run(self, variant_id, ids, pairs, metric, intervention, forward_budget, *, channels=None,
             discovery_pairs=None, request_type, shift=None, transform=None, evidence=None):
        backend, policy = self.forward.backend, self.forward.policy
        specs = [self._site(site_id, channels) for site_id in ids]
        pairs_prepared = self._prepare_pairs(pairs)
        calls_per_pair = len(ids) + 2
        tokens = sum(len(prepared[0]) for prepared in pairs_prepared) * calls_per_pair
        if len(pairs) * calls_per_pair > forward_budget or tokens > policy.max_total_text_tokens:
            raise ValueError("Intervention exceeds the declared forward or token budget")
        snapshot = self.versions.snapshot(variant_id)
        request = {"type": request_type, "metric": metric, "method": intervention, "site_ids": ids,
                   "pairs": pairs, "channels": channels, "discovery_pairs": discovery_pairs}
        if evidence is not None:
            request[intervention] = evidence
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
                sites = [dict(spec, positions=pair["positions"], method=intervention, shift=shift, transform=transform)
                         for spec in specs]
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

    def _eraser(self, variant_id, site_id, artifact_id, control_seed):
        """只在拟合时登记的版本和站点应用变换，可用同幅度随机方向做对照。"""
        from concept_erasure import LeaceEraser

        self._site(site_id)
        factors, info = self.artifacts.array(artifact_id)
        if (info.get("method") != "leace" or info["axes"] != ["channel", "factor"]
                or info.get("site_id") != site_id or info.get("variant_id") != variant_id):
            raise ValueError("Use a fitted LEACE artifact from the selected model version and site")
        rank = factors.shape[1] // 2
        packed = torch.from_numpy(factors.copy()).to(self.forward.backend.device)
        eraser = LeaceEraser(packed[:, :rank], packed[:, rank:2 * rank].T, packed[:, -1])
        if control_seed is None:
            return eraser

        def matched_random(values):
            original = values.to(packed.dtype)
            norm = (eraser(original) - original).norm(dim=-1, keepdim=True)
            generator = torch.Generator().manual_seed(control_seed)
            random = torch.randn(original.shape, dtype=packed.dtype, generator=generator).to(packed.device)
            return (original + torch.nn.functional.normalize(random) * norm).to(values.dtype)

        return matched_random

    def run_intervention(self, variant_id, site_id, pair, method="replace", metric="logit_difference", channels=None,
                         direction=None, direction_artifact_id=None, strength=None, eraser_artifact_id=None,
                         control_seed=None):
        """返回一次干预的三个条件与差值。"""
        shift = transform = evidence = None
        if method == "direction":
            shift, evidence = self._direction(site_id, channels, strength, direction, direction_artifact_id)
        if method == "concept_erasure":
            transform = self._eraser(variant_id, site_id, eraser_artifact_id, control_seed)
            evidence = {"eraser_artifact_id": eraser_artifact_id, "control_seed": control_seed,
                        "mode": "leace" if control_seed is None else "matched_random_shift"}
        result, rows = self._run(variant_id, [site_id], [pair], metric, method, 3, channels=channels,
                                 request_type="run_intervention", shift=shift, transform=transform, evidence=evidence)
        if evidence is not None:
            result[method] = {key: value for key, value in evidence.items() if key != "values"}
        return {**result, "site_id": site_id, "method": method, **rows[0][0]}

    def _prefilter_atp(self, variant_id, site_ids, pairs, prepared, metric):
        """记录估计值及实际前向和反向次数，供精确干预选点。"""
        backend = self.forward.backend
        sites = [self._site(site) for site in site_ids]
        snapshot = self.versions.snapshot(variant_id)
        ledger = self.forward.ledger
        request = {"type": "localize_behavior", "method": "atp_star", "metric": metric,
                   "site_ids": site_ids, "pairs": pairs, "grad_drop_layers": list(range(backend.num_layers))}
        attempt = ledger.begin(request, variant_id)
        run_id = f"request_{attempt:04d}"
        started = perf_counter()
        usage = {"executed_model_forwards": 0, "executed_model_backwards": 0}
        try:
            values = np.stack([atp_star_scores(backend, snapshot, sites, record, pair, metric, usage)
                               for record, pair in zip(prepared, pairs)])
            if not np.isfinite(values).all():
                raise ValueError("AtP* produced nonfinite gradient estimates")
            artifact_id = self.artifacts.save_array("atp_star_estimates", values, ["pair", "site", "estimator"],
                run_id=run_id, variant_id=variant_id, site_ids=site_ids, metric=metric,
                estimator_names=["plain_atp", "grad_drop_magnitude"], method="atp_star",
                evidence_type="gradient_estimate")
            usage.update(elapsed_seconds=perf_counter() - started, artifact_id=artifact_id,
                         input_tokens=2 * sum(len(row[0]) for row in prepared))
            ledger.finish(attempt, True, usage)
        except Exception as error:
            ledger.finish(attempt, False, {**usage, "error_type": type(error).__name__, "error": str(error)[:500]})
            raise
        averages = values.mean(axis=0)
        ranking = [{"site_id": site, "mean_plain_effect": float(averages[index, 0]),
                    "mean_grad_drop_magnitude": float(averages[index, 1])} for index, site in enumerate(site_ids)]
        ranking.sort(key=lambda row: (-row["mean_grad_drop_magnitude"], row["site_id"]))
        return {"run_id": run_id, "artifact_id": artifact_id, "ranking": ranking, "usage": usage,
                "correction": "grad_drop", "site_kinds": ["residual", "mlp_input"]}

    def localize_behavior(self, variant_id, site_ids, pairs, metric="logit_difference", method="exact_scan",
                          review_pairs=None, control_site_ids=(), forward_budget=32, verify_top_k=4, backward_budget=128):
        """按发现集平均效应排序，同时报告对照和复核效果。"""
        review_pairs = review_pairs or []
        ids = [*site_ids, *control_site_ids]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate and control sites must be distinct")
        prefilter = None
        if method == "atp_star":
            if not self.has_atp_star:
                raise ValueError("AtP* requires gradient access and a supported Llama or Qwen3 decoder")
            prepared = self._prepare_pairs([*pairs, *review_pairs])
            top = min(len(site_ids), verify_top_k)
            exact_calls = top + len(control_site_ids) + 2
            forwards = 2 * len(pairs) + exact_calls * len(prepared)
            tokens = 2 * sum(len(row[0]) for row in prepared[:len(pairs)]) + exact_calls * sum(len(row[0]) for row in prepared)
            if (len(site_ids) > 128 or max(len(row[0]) for row in prepared[:len(pairs)]) > 256
                    or forwards > forward_budget or tokens > self.forward.policy.max_total_text_tokens
                    or (self.forward.backend.num_layers + 1) * len(pairs) > backward_budget):
                raise ValueError("AtP* exceeds the candidate, gradient sequence, forward or backward budget")
            for site in control_site_ids:
                self._site(site)
            prefilter = self._prefilter_atp(variant_id, site_ids, pairs, prepared[:len(pairs)], metric)
            site_ids = [row["site_id"] for row in prefilter["ranking"][:top]]
            ids = [*site_ids, *control_site_ids]
            forward_budget -= prefilter["usage"]["executed_model_forwards"]
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
        result = {**summary, "method": method, "site_ids": ids, "discovery_pairs": len(pairs),
                  "review_pairs": len(review_pairs), "controls": ranked[len(site_ids):],
                  "ranked_sites": sorted(ranked[:len(site_ids)], key=lambda item: (-item["mean_effect"], item["site_id"]))}
        if prefilter is not None:
            result["prefilter"] = prefilter
            exact = result["usage"]
            estimate = prefilter["usage"]
            result["usage"] = {**exact, **{field: exact[field] + estimate[field]
                               for field in ("elapsed_seconds", "input_tokens", "executed_model_forwards")},
                               "executed_model_backwards": estimate["executed_model_backwards"]}
        return result
