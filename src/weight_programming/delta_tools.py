"""在已登记的权重坐标上分析与压缩模型更新。"""

import math
from collections import defaultdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import torch

from .editing_tools import _save
from .patches import _digest
from .tensor_store import sha


class DeltaTools:
    """复用候选分支和已有的低秩更新代数。"""

    def __init__(self, versions, artifacts, input_rows):
        self.versions = versions
        self.artifacts = artifacts
        self.input_rows = input_rows
        self.updates = versions.root / "updates"

    def _adapter(self, adapter_id, parent_variant_id):
        from delta_programming import load_effective

        if not re.fullmatch(r"[A-Za-z0-9_-]+", adapter_id):
            raise ValueError("Invalid registered update ID")
        folder = self.updates / adapter_id
        if folder.is_symlink():
            raise ValueError("Registered updates must belong to the workspace")
        meta, factors = load_effective(folder)
        if meta["kind"] != "functional_lora" or meta["version"] != 1 or not factors:
            raise ValueError("Expected a WSP functional LoRA update")
        provenance = meta["provenance"]
        detail_parent = provenance.get("details", {}).get("parent_variant_id", self.versions.base["variant_id"])
        expected_parent = provenance.get("parent_variant_id", detail_parent)
        if expected_parent != parent_variant_id:
            raise ValueError("Update and requested parent variant differ")
        if meta["base_id"] != _digest(meta["base_files"]):
            raise ValueError("Invalid update base fingerprint")
        # 旧适配器可能记录更宽的文件集，逐项验证并要求包含全部权重分片。
        source = self.versions.store
        weight_files = {source._source_path(name).name for name in source.names}
        if not weight_files.issubset(meta["base_files"]):
            raise ValueError("Update fingerprint omits checkpoint weights")
        for filename, expected in meta["base_files"].items():
            if Path(filename).name != filename:
                raise ValueError(f"Invalid update base file: {filename}")
            recorded = source._manifest["source_files"].get(filename)
            if (recorded if recorded is not None else sha(source.model_dir / filename)) != expected:
                raise ValueError(f"Update base file does not match: {filename}")
        result = {}
        for module, (a, b) in factors.items():
            name = module + ".weight"
            shape = source.tensor_info(name)["shape"]
            if (a.ndim != 2 or b.ndim != 2 or list(shape) != [b.shape[0], a.shape[1]]
                    or a.shape[0] != b.shape[1] or not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                raise ValueError(f"Update factors do not match: {name}")
            result[name] = (a, b)
        return result

    def _tensors(self, ids, names):
        if names is not None:
            return list(names)
        return sorted({name for variant_id in ids for name in self.versions.tensor_versions(variant_id)})

    @torch.no_grad()
    def analyze_weight_delta(self, base_variant_id, variant_id=None, adapter_id=None, tensors=None,
                             other_variant_id=None, direction_rank=0, input_artifacts=None):
        """返回原精度差分、更新方向及可选的局部输出变化。"""
        self.versions.variant(base_variant_id)
        if (variant_id is None) == (adapter_id is None):
            raise ValueError("Choose exactly one candidate variant or registered update")
        factors = self._adapter(adapter_id, base_variant_id) if adapter_id is not None else None
        if variant_id is not None:
            self.versions.variant(variant_id)
        if other_variant_id is not None:
            self.versions.variant(other_variant_id)
        ids = [item for item in (base_variant_id, variant_id, other_variant_id) if item is not None]
        names = list(tensors) if tensors is not None else sorted(set(factors or ()) | set(self._tensors(ids, None)))
        input_artifacts = input_artifacts or {}
        if set(input_artifacts) - set(names):
            raise ValueError("Input artifacts must correspond to selected tensors")
        records = []
        layers = defaultdict(lambda: {"tensors": 0, "changed_elements": 0, "delta_norm_squared": 0.0})
        left = right = dot = 0.0
        for name in names:
            base = self.versions.read_native(base_variant_id, name)
            if factors is None:
                candidate = self.versions.read_native(variant_id, name)
                effective = None
            elif name in factors:
                a, b = factors[name]
                effective = b @ a
                candidate = (base.double() + effective).to(base.dtype)
            else:
                candidate, effective = base, None
            delta = candidate.double() - base.double()
            squared = float(delta.square().sum())
            left += squared
            base_norm = float(base.double().norm())
            row = {"tensor": name, "shape": list(base.shape), "dtype": str(base.dtype),
                   "changed": int(delta.ne(0).sum()), "delta_norm": math.sqrt(squared),
                   "relative_norm": math.sqrt(squared) / base_norm if base_norm else None}
            match = re.search(r"(?:^|\.)layers\.([0-9]+)\.", name)
            if match is not None:
                layer = int(match.group(1))
                row["layer"] = layer
                layers[layer]["tensors"] += 1
                layers[layer]["changed_elements"] += row["changed"]
                layers[layer]["delta_norm_squared"] += squared
            if delta.ndim == 2:
                row["changed_columns"] = int(delta.ne(0).any(0).sum())
            if effective is not None:
                row["theoretical_update_norm"] = float(effective.norm())
                row["native_rounding_error_norm"] = float((effective - delta).norm())
            if other_variant_id is not None:
                other = self.versions.read_native(other_variant_id, name).double() - base.double()
                norm2, pair_dot = float(other.square().sum()), float((delta * other).sum())
                right += norm2
                dot += pair_dot
                row["signed_delta_cosine"] = pair_dot / math.sqrt(squared * norm2) if squared and norm2 else None
            if direction_rank and squared and delta.ndim == 2:
                u, s, vh = self._svd(delta)
                count = min(direction_rank, s.numel())
                row["singular_values"] = s[:count].tolist()
                row["output_directions"] = self.artifacts.save_array("directions", u[:, :count].T.numpy(),
                                                                     ["direction", "output"], tensor=name)
                row["input_directions"] = self.artifacts.save_array("directions", vh[:count].numpy(),
                                                                    ["direction", "input"], tensor=name)
            if name in input_artifacts:
                inputs = self.input_rows(base_variant_id, name, input_artifacts[name]).double()
                response = inputs @ delta.T
                row["output_change_rms"] = float(response.square().mean().sqrt())
            records.append(row)
        records.sort(key=lambda row: (-row["delta_norm"], row["tensor"]))
        layer_summaries = [{"layer": layer, "tensors": data["tensors"], "changed_elements": data["changed_elements"],
                            "delta_norm": math.sqrt(data["delta_norm_squared"])} for layer, data in sorted(layers.items())]
        return {"base_variant_id": base_variant_id, "variant_id": variant_id, "adapter_id": adapter_id,
                "other_variant_id": other_variant_id, "tensor_count": len(records),
                "changed_tensors": sum(row["changed"] > 0 for row in records), "delta_norm": math.sqrt(left),
                "signed_delta_cosine": dot / math.sqrt(left * right) if left and right else None,
                "report_artifact_id": self.artifacts.save_records("weight_delta", records),
                "layer_summaries": layer_summaries, "top_tensors": records[:5]}

    @staticmethod
    def _svd(delta):
        if delta.ndim != 2 or delta.numel() > 4_000_000:
            raise ValueError("Exact dense SVD requires a selected 2-D tensor of at most 4M values")
        return torch.linalg.svd(delta, full_matrices=False)

    def publish_update(self, pairs, method, provenance, native_adapter=None, method_provenance=None):
        """统一发布低秩因子，训练时可同时保留原生 PEFT 文件。"""
        from delta_programming import emit

        source = self.versions.store
        base = {"source": str(source.model_dir), "files": source._manifest["source_files"]}
        self.updates.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.updates) as temporary:
            bundle = Path(temporary) / "update"
            manifest = emit(bundle, base, {name.removesuffix(".weight"): pair for name, pair in pairs.items()},
                            method, provenance, method_provenance=method_provenance)
            if native_adapter is not None:
                native_adapter.save_pretrained(bundle / "peft", safe_serialization=True, save_embedding_layers=False)
            update_id = _digest(manifest)
            with self.versions._publish_lock():
                destination = self.updates / update_id
                if not destination.exists():
                    bundle.rename(destination)
        return update_id

    @torch.no_grad()
    def compress_update(self, parent_variant_id, rank, method="truncated_svd", patch_id=None, adapter_id=None,
                        input_artifacts=None, materialize=False):
        """压缩已有更新，记录误差并按需生成密集补丁。"""
        from delta_programming import norm, residual, truncate, weighted_truncate

        self.versions.variant(parent_variant_id)
        if (patch_id is None) == (adapter_id is None):
            raise ValueError("Choose exactly one source patch or registered update")
        if adapter_id is not None:
            factors = self._adapter(adapter_id, parent_variant_id)
        else:
            replacements = self.versions.patch_tensors(parent_variant_id, patch_id)
            factors = {}
            for name, tensor in replacements.items():
                if name != self.versions.groups[name][0]:
                    continue
                if not name.endswith(".weight") or tensor.ndim != 2:
                    raise ValueError(f"Patch contains a tensor outside low-rank linear updates: {name}")
                delta = tensor.double() - self.versions.read_native(parent_variant_id, name).double()
                u, s, vh = self._svd(delta)
                factors[name] = (vh, u * s)
        input_artifacts = input_artifacts or {}
        if method == "activation_weighted_svd" and set(input_artifacts) != set(factors):
            raise ValueError("Activation-weighted compression requires one input artifact per target")
        if method == "truncated_svd" and input_artifacts:
            raise ValueError("Input artifacts are used only by activation-weighted compression")
        compressed, details = {}, {}
        for name, (a, b) in factors.items():
            if method == "truncated_svd":
                small_a, small_b, metrics = truncate(a, b, rank)
                metrics["leading_singular_values"] = metrics.pop("singular_values")[:rank]
            else:
                inputs = self.input_rows(parent_variant_id, name, input_artifacts[name]).double()
                latent = inputs @ a.T
                small_a, small_b, metrics = weighted_truncate(a, b, latent.T @ latent / len(latent), rank)
            small_a, small_b = small_a.float(), small_b.float()
            compressed[name] = (small_a, small_b)
            details[name] = {**metrics, "rank": rank, "source_update_norm": norm(a, b),
                             "frobenius_error": residual(a, b, small_a, small_b),
                             "stored_parameters": small_a.numel() + small_b.numel()}
        provenance = {"parent_variant_id": parent_variant_id, "source_patch_id": patch_id,
                      "source_adapter_id": adapter_id, "input_artifacts": input_artifacts, "metrics": details}
        update_id = self.publish_update(compressed, method, provenance)
        patch = None
        if materialize:
            replacements = {name: self.versions.read_native(parent_variant_id, name).float() + b @ a
                            for name, (a, b) in compressed.items()}
            patch = _save(self.versions, parent_variant_id, replacements, method,
                          list(input_artifacts.values()), [patch_id] if patch_id else [])
        return {"adapter_id": update_id, "update_path": f"updates/{update_id}", "method": method,
                "parent_variant_id": parent_variant_id, "patch_id": patch["patch_id"] if patch else None,
                "stored_parameters": sum(row["stored_parameters"] for row in details.values()),
                "report_artifact_id": self.artifacts.save_records("compression", ({"tensor": name, **row}
                                                                                 for name, row in details.items()))}
