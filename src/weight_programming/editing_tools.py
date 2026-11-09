"""把已有的方向投影和更新合并算法接到统一补丁。"""

import math

import torch

from .directions import project_output_direction
from .merging import dare_drop, ties_merge
from .reconstruction import input_hessian, sparsegpt_reconstruct


def _norm(tensors):
    return math.sqrt(sum(float(value.double().square().sum()) for value in tensors.values()))


def _save(versions, parent_variant_id, replacements, method, source_ids=(), patch_ids=()):
    edits = [{"tensor": name, "op": "replace"} for name in replacements]
    return versions.create_patch(parent_variant_id, edits, method,
                                 lambda edit: replacements[edit["tensor"]], source_ids, patch_ids)


def project_weight(versions, parent_variant_id, tensor, direction, columns, strength=1.0, source_ids=()):
    """保留未选列，并把所选列沿指定方向的分量削弱。"""
    weight = versions.read_native(parent_variant_id, tensor)
    vector = torch.as_tensor(direction, dtype=torch.float32)
    replacement = project_output_direction(weight, vector, columns, strength)
    result = _save(versions, parent_variant_id, {tensor: replacement}, "output_direction_projection", source_ids)
    realized = replacement.to(weight.dtype).double() - weight.double()
    return {**result, "columns": columns, "strength": strength, "realized_norm": float(realized.norm())}


def merge_updates(versions, parent_variant_id, patch_ids, method, coefficients=None,
                  density=None, target_norm=None, drop_probability=None, seed=None, dare_merge=None,
                  adapter_deltas=(), adapter_ids=()):
    """将同一父版本上的补丁转成有效更新，再做加权或 TIES 合并。"""
    patches = [versions.patch_tensors(parent_variant_id, patch_id) for patch_id in patch_ids]
    if len(patches) + len(adapter_deltas) < 2:
        raise ValueError("Provide at least two source patches or registered adapters")
    names = sorted({versions.groups[name][0] for patch in [*patches, *adapter_deltas] for name in patch})
    parent_native = versions.read_tensors(parent_variant_id, names)
    parent = {name: value.float() for name, value in parent_native.items()}
    deltas = [{name: patch[name].float() - weight if name in patch else torch.zeros_like(weight)
               for name, weight in parent.items()} for patch in patches]
    deltas.extend({name: delta[name].float() if name in delta else torch.zeros_like(weight)
                   for name, weight in parent.items()} for delta in adapter_deltas)
    merge_rule = method
    dare_evidence = None
    if method == "dare":
        if drop_probability is None or density is not None:
            raise ValueError("DARE requires a drop probability and uses its own random mask")
        merge_rule = "task_arithmetic" if dare_merge is None else dare_merge
        deltas, dare_evidence = dare_drop(deltas, drop_probability, 11 if seed is None else seed)
    elif any(value is not None for value in (drop_probability, seed, dare_merge)):
        raise ValueError("Drop probability, seed and DARE merge rule require method=dare")
    if merge_rule == "ties":
        if coefficients is not None:
            raise ValueError("Coefficients are only used by task arithmetic")
        default_density = 1.0 if method == "dare" else 0.2
        keep_density = default_density if density is None else density
        merged, evidence = ties_merge(deltas, [parent_variant_id] * len(deltas),
                                      density=keep_density)
    elif merge_rule == "task_arithmetic":
        if density is not None:
            raise ValueError("Density is only used by TIES")
        if coefficients is None:
            coefficients = [1.0 / len(deltas)] * len(deltas)
        if len(coefficients) != len(deltas) or not all(math.isfinite(value) for value in coefficients):
            raise ValueError("Provide one finite coefficient per patch")
        merged = {name: sum(coefficient * delta[name] for coefficient, delta in zip(coefficients, deltas))
                  for name in names}
        evidence = {"coefficients": coefficients, "source_norms": [_norm(delta) for delta in deltas]}
    else:
        raise ValueError(f"Unknown merge rule: {merge_rule}")
    if dare_evidence is not None:
        evidence = {**evidence, "dare": {**dare_evidence, "merge_rule": merge_rule}}
    original_norm = _norm(merged)
    if target_norm is not None:
        if not math.isfinite(target_norm) or target_norm <= 0 or original_norm == 0:
            raise ValueError("Norm matching requires a finite positive target and nonzero update")
        merged = {name: value * (target_norm / original_norm) for name, value in merged.items()}
    replacements = {name: weight + merged[name] for name, weight in parent.items()}
    patch = _save(versions, parent_variant_id, replacements, method,
                  source_ids=[f"updates/{adapter_id}" for adapter_id in adapter_ids], patch_ids=patch_ids)
    native = {name: value.to(parent_native[name].dtype).float() - parent[name]
              for name, value in replacements.items()}
    return {**patch, "input_patch_ids": patch_ids, "input_adapter_ids": list(adapter_ids), "merge_evidence": evidence,
            "merged_norm": original_norm, "target_norm": target_norm, "realized_norm": _norm(native)}


def prune_and_compensate(versions, artifacts, parent_variant_id, tensor, inputs, input_artifact_id,
                         method, sparsity=None, mask_artifact_id=None, relative_damping=0.01):
    """复用 Wanda 或 SparseGPT 算法，保存删除掩码与候选补丁。"""
    weight = versions.read_native(parent_variant_id, tensor)
    if (weight.ndim != 2 or inputs.ndim != 2 or inputs.shape[1] != weight.shape[1]
            or not inputs.numel() or not torch.isfinite(inputs).all()):
        raise ValueError("Calibration rows must match the linear weight input coordinates")
    sources = [input_artifact_id]
    if method == "wanda":
        from paper_methods import wanda_mask

        if sparsity is None or mask_artifact_id is not None:
            raise ValueError("Wanda requires sparsity and computes its own mask")
        mask = wanda_mask(weight.float(), inputs.float().square().sum(0), sparsity)
        replacement = weight.masked_fill(mask, 0)
        details = {"selection": "wanda_per_output_row"}
    else:
        if weight.shape[1] ** 2 * 8 > 256_000_000:
            raise ValueError("The SparseGPT input covariance exceeds the 256 MB matrix budget")
        if method == "sparsegpt_fixed_mask":
            if sparsity is not None or mask_artifact_id is None:
                raise ValueError("Fixed-mask reconstruction requires mask_artifact_id only")
            array, info = artifacts.array(mask_artifact_id)
            if info.get("tensor") != tensor or info.get("variant_id") != parent_variant_id:
                raise ValueError("Mask belongs to different weight coordinates or a different variant")
            options = {"prune_mask": torch.from_numpy(array.copy())}
            sources.append(mask_artifact_id)
        elif method == "sparsegpt_adaptive":
            if sparsity is None or mask_artifact_id is not None:
                raise ValueError("Adaptive reconstruction requires sparsity and computes its own mask")
            options = {"sparsity": sparsity}
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        replacement, mask, details = sparsegpt_reconstruct(weight, input_hessian(inputs),
                                                            relative_damping=relative_damping, **options)
    native = replacement.to(weight.dtype)
    layer_error = inputs.double() @ (native.double() - weight.double()).T
    mask_id = artifacts.save_array("prune_mask", mask.numpy(), ["output", "input"],
                                   tensor=tensor, variant_id=parent_variant_id)
    patch = _save(versions, parent_variant_id, {tensor: replacement}, method, [*sources, mask_id])
    return {**patch, "mask_artifact_id": mask_id, "calibration_tokens": inputs.shape[0],
            "realized_zero_fraction": float(native.eq(0).double().mean()),
            "output_mean_squared_error": float(layer_error.square().mean()), "pruning_evidence": details}
