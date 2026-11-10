"""Analyze and compress model updates on registered weight coordinates."""

import json
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
    """Reuse candidate branches and existing low-rank update algebra."""

    def __init__(self, versions, artifacts, input_rows, backend):
        self.versions = versions
        self.artifacts = artifacts
        self.input_rows = input_rows
        self.backend = backend
        self.updates = versions.root / "updates"
        modules = {id(module): name for name, module in backend.model.named_modules()}
        self.layer_prefixes = [(modules[id(layer)] + ".", index) for index, layer in enumerate(backend.layers)]

    @torch.no_grad()
    def import_peft_adapter(self, parent_variant_id, adapter_dir):
        """Import native linear LoRA from the host, preserving PEFT scaling and transpose semantics."""
        from peft import LoraConfig, PeftConfig, PeftModel
        from peft.tuners.lora.layer import Linear, LoraLayer
        from peft.utils import get_peft_model_state_dict, load_peft_weights

        from .isolated_model import load_variant

        folder = Path(adapter_dir).expanduser().resolve(strict=True)
        config = PeftConfig.from_pretrained(folder, local_files_only=True)
        if not isinstance(config, LoraConfig) or (config.bias != "none" or config.modules_to_save
                or config.layer_replication or config.target_parameters or config.trainable_token_indices
                or config.lora_bias):
            raise ValueError("Native import requires linear LoRA without extra base-weight or bias changes")
        # Independent instances load the specified parent version, and the import does not change the model being used.
        base = load_variant(self.backend, self.versions, parent_variant_id, dtype=torch.float32)
        adapter = PeftModel.from_pretrained(base, folder, config=config, local_files_only=True).eval()
        saved = load_peft_weights(folder, device="cpu", local_files_only=True)
        loaded = get_peft_model_state_dict(adapter, save_embedding_layers=False)
        if set(saved) != set(loaded):
            raise ValueError("Native adapter contains missing or unsupported state entries")
        for key, value in saved.items():
            if not torch.equal(value.float().cpu(), loaded[key].float().cpu()):
                raise ValueError(f"Native adapter state did not load exactly: {key}")
        pairs, details = {}, {}
        for name, layer in adapter.get_base_model().named_modules():
            if not isinstance(layer, LoraLayer):
                continue
            if not isinstance(layer, Linear) or layer.lora_variant:
                raise ValueError(f"Native adapter type has no verified linear-factor conversion: {name}")
            a = layer.lora_A["default"].weight.detach().float().cpu()
            b = layer.lora_B["default"].weight.detach().float().cpu() * layer.scaling["default"]
            if layer.fan_in_fan_out:
                a, b = b.T, a.T
            delta = layer.get_delta_weight("default").float().cpu()
            torch.testing.assert_close(b @ a, delta, rtol=1e-5, atol=1e-7)
            tensor = name + ".weight"
            if list(delta.shape) != self.versions.store.tensor_info(tensor)["shape"]:
                raise ValueError(f"Native adapter target does not match the checkpoint: {tensor}")
            pairs[tensor] = (a.contiguous(), b.contiguous())
            details[tensor] = {"scale": float(layer.scaling["default"]),
                               "fan_in_fan_out": bool(layer.fan_in_fan_out), "rank": a.shape[0]}
        if not pairs:
            raise ValueError("Native adapter has no supported linear LoRA targets")
        provenance = {"parent_variant_id": parent_variant_id, "source": str(folder),
                      "source_files": {path.name: sha(path) for path in sorted(folder.iterdir()) if path.is_file()},
                      "peft_config": json.loads((folder / "adapter_config.json").read_text(encoding="utf-8")),
                      "targets": details}
        adapter_id = self.publish_update(pairs, "native_peft_import", provenance,
            method_provenance={"papers": ["https://arxiv.org/abs/2106.09685"],
                               "implementation_grade": "PEFT native linear LoRA conversion"})
        return {"adapter_id": adapter_id, "parent_variant_id": parent_variant_id,
                "tensor_count": len(pairs), "targets": details}

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
        # Older adapters may record a broader file set; verify each file and require all weight shards.
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
                             other_variant_id=None, direction_rank=0, input_artifacts=None,
                             direction_solver="exact", direction_seed=11, direction_iterations=4):
        """Return the difference at original precision, update direction, and optional local output changes."""
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
            layer = next((index for prefix, index in self.layer_prefixes if name.startswith(prefix)), None)
            if layer is not None:
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
                if direction_solver == "exact":
                    u, s, vh = self._svd(delta)
                elif direction_solver == "randomized":
                    if delta.numel() > 32_000_000:
                        raise ValueError("Randomized direction analysis accepts at most 32M selected matrix values")
                    # Stochastic low-rank decomposition using PyTorch, fixed seed and limited number of subspace iterations.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(direction_seed)
                        u, s, v = torch.svd_lowrank(delta, q=min(direction_rank + 8, min(delta.shape)),
                                                   niter=direction_iterations)
                    vh = v.T
                else:
                    raise ValueError("Choose exact or randomized direction decomposition")
                count = min(direction_rank, s.numel())
                row["singular_values"] = s[:count].tolist()
                row["direction_decomposition"] = {"method": direction_solver, "rank": count,
                    "seed": direction_seed if direction_solver == "randomized" else None,
                    "iterations": direction_iterations if direction_solver == "randomized" else None,
                    "residual_frobenius_norm": math.sqrt(max(0.0, squared - float(s[:count].square().sum())))}
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
        """Low-rank factors are published uniformly, and native PEFT files can be retained during training."""
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
        """Compress an existing update, record its error, and optionally generate a dense patch."""
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
        if isinstance(rank, dict) and set(rank) != set(factors):
            raise ValueError("Per-tensor rank budgets must cover exactly the selected update tensors")
        compressed, details = {}, {}
        for name, (a, b) in factors.items():
            retained_rank = rank[name] if isinstance(rank, dict) else rank
            if method == "truncated_svd":
                small_a, small_b, metrics = truncate(a, b, retained_rank)
                metrics["leading_singular_values"] = metrics.pop("singular_values")[:retained_rank]
            else:
                inputs = self.input_rows(parent_variant_id, name, input_artifacts[name]).double()
                latent = inputs @ a.T
                small_a, small_b, metrics = weighted_truncate(a, b, latent.T @ latent / len(latent), retained_rank)
            small_a, small_b = small_a.float(), small_b.float()
            compressed[name] = (small_a, small_b)
            details[name] = {**metrics, "rank": retained_rank, "source_update_norm": norm(a, b),
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
