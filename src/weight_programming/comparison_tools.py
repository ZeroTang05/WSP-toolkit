"""复用权重差分、共同输入的表示和候选评测来比较模型。"""

import hashlib
import json

import numpy as np

from .representation_tools import activation_rows


def _gram(array):
    """按样本对齐后计算居中的线性核。"""
    if not np.isfinite(array).all():
        raise ValueError("Representation contains nonfinite values")
    centered = array.astype(np.float64) - array.mean(axis=0, dtype=np.float64)
    gram = centered @ centered.T
    norm = np.linalg.norm(gram)
    return gram / norm if norm else np.full(gram.shape, np.nan)


class ComparisonTools:
    """宿主可登记第二个检查点，跨基座比较使用各自的执行器。"""

    def __init__(self, primary):
        self.primary = primary

    def compare_models(self, left_variant_id, right_variant_id, views, right_workspace="primary",
                       texts=None, left_sites=None, right_sites=None, tensors=None, evaluation=None, crosscoder=None):
        """独立返回所选视图及其证据。"""
        left = self.primary
        right = left.workspace_tools(right_workspace)
        left.versions.variant(left_variant_id)
        right.versions.variant(right_variant_id)
        result = {"left_variant_id": left_variant_id, "right_variant_id": right_variant_id,
                  "right_workspace": right_workspace, "views": views}
        if "weights" in views:
            if left is not right:
                raise ValueError("Direct weight comparison requires a shared registered weight space")
            result["weights"] = left.delta.analyze_weight_delta(left_variant_id, right_variant_id, tensors=tensors)
        if "representations" in views:
            result["representations"] = self._representations(left, right, left_variant_id, right_variant_id,
                                                                texts, left_sites, right_sites, right_workspace)
        if "behavior" in views:
            if evaluation is None:
                raise ValueError("Behavior comparison requires registered development evaluation settings")
            result["behavior"] = left.evaluation.evaluate_variants(left_variant_id, [right_variant_id], **evaluation,
                                                                     comparison=right.evaluation if right is not left else None)
        if "crosscoder" in views:
            if crosscoder is None:
                raise ValueError("Crosscoder comparison requires a registered dictionary and two snapshots")
            options = dict(crosscoder)
            artifact_id = options.pop("left_artifact_id")
            result["crosscoder"] = left.features.extract_features(artifact_id=artifact_id, method="crosscoder",
                                                                    right_workspace=right_workspace,
                                                                    expected_variants=[left_variant_id, right_variant_id],
                                                                    **options)
        return result

    @staticmethod
    def _representation_rows(tools, variant_id, texts, sites):
        mode = "teacher_forced" if tools.is_language_model else "predict"
        run = tools.model_forward(variant_id, mode, texts=texts, observe=sites)
        keys = {entry.get("site_id"): entry["artifact_id"] for entry in run["artifacts"].values()}
        representations = []
        for site in sites:
            array, _, _, _ = activation_rows(tools.artifacts, keys[site])
            representations.append(_gram(array))
        return run["run_id"], representations

    def _representations(self, left, right, left_id, right_id, texts, left_sites, right_sites, right_workspace):
        if texts is None or len(texts) < 2:
            raise ValueError("Representation comparison requires at least two shared inputs")
        left_sites = left_sites or ["block.0.residual_out"]
        right_sites = right_sites or ["block.0.residual_out"]
        for tools, sites in ((left, left_sites), (right, right_sites)):
            if any(tools._sites.get(site, {}).get("kind") not in ("hidden", "mlp", "mlp_out") for site in sites):
                raise ValueError("Representation comparison requires residual or MLP activation sites")
        left_run, a = self._representation_rows(left, left_id, texts, left_sites)
        if left is right and left_id == right_id and left_sites == right_sites:
            right_run, b = left_run, a
        else:
            right_run, b = self._representation_rows(right, right_id, texts, right_sites)
        cka = np.asarray([[np.clip((x * y).sum(), 0, 1) for y in b] for x in a])
        artifact_id = left.artifacts.save_array("linear_cka", cka, ["left_site", "right_site"],
                                                left_sites=left_sites, right_sites=right_sites)
        inputs_sha256 = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
        return {"method": "linear_cka", "alignment": "same_text_last_valid_token", "samples": len(texts),
                "inputs_sha256": inputs_sha256, "left_sites": left_sites, "right_sites": right_sites,
                "left_run_id": left_run, "right_run_id": right_run, "right_workspace": right_workspace,
                "artifact_id": artifact_id, "undefined_cells": int(np.isnan(cka).sum())}
