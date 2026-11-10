"""Release structural or quantized checkpoints and isolate the upstream compression runtime environment."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from safetensors import safe_open
from safetensors.torch import save_file

from .patches import _digest
from .tensor_store import sha
from .worker_runtime import run_model_worker


def publish_checkpoint(versions, output, record):
    """Publish the complete checkpoint and manifest to the shared export directory."""
    root = versions.root / "exports" / "structures"
    export_id = _digest(record)
    (output / "structure_manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    with versions._publish_lock():
        destination = root / export_id
        if not destination.exists():
            output.rename(destination)
    checkpoint = f"exports/structures/{export_id}"
    return {**record, "export_id": export_id, "checkpoint_path": checkpoint,
            "manifest_path": f"{checkpoint}/structure_manifest.json"}


class ExternalCompression:
    """Pass the parent model and calibration tokens to a standalone Python specified by the host."""

    def __init__(self, tools, quantization_python=None, slicegpt_python=None):
        self.tools = tools
        self.quantization_python = self._register(quantization_python, ("llama", "qwen3"), "LLM Compressor")
        self.slicegpt_python = self._register(slicegpt_python, ("llama",), "SliceGPT")

    def _register(self, python, architectures, name):
        """Register only host-provided environments compatible with the model."""
        if python is None:
            return None
        if not self.tools.allow_edits or self.tools.forward.backend.identity["model_type"] not in architectures:
            raise ValueError(f"{name} requires edit permission and one of these architectures: {architectures}")
        path = Path(python).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"Provide an executable Python from the {name} environment")
        return path

    @contextmanager
    def _stage(self, variant_id, method, calibration_texts, python, worker, **options):
        """Both backends share calibration, parent version overrides, and staging directories."""
        if python is None:
            raise ValueError(f"Register a Python environment for {method}")
        tools = self.tools
        backend, policy = tools.forward.backend, tools.forward.policy
        ids = [backend.encode(text) for text in calibration_texts]
        tokens = sum(map(len, ids))
        if (max(map(len, ids)) > min(2048, backend.context_length, policy.max_text_tokens)
                or tokens > min(32768, policy.max_total_text_tokens)):
            raise ValueError("Calibration exceeds the declared token budget")
        snapshot = tools.versions.snapshot(variant_id)
        base = tools.versions.store.model_dir
        root = tools.versions.root / "exports" / "structures"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f"{method}-", dir=root) as temporary:
            directory = Path(temporary)
            if snapshot["values"]:
                save_file(snapshot["values"], directory / "parent.safetensors")
            request = {"checkpoint": str(base), "method": method, "calibration_ids": ids, **options}
            (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
            report = run_model_worker(python, worker, directory, backend.device)
            original_bytes = sum(path.stat().st_size for path in base.glob("*.safetensors"))
            calibration = {"calibration_tokens": tokens, "calibration_samples": len(ids), "calibration_sha256": _digest(ids)}
            yield directory / "checkpoint", report, original_bytes, calibration

    def quantize(self, variant_id, method, calibration_texts, group_size):
        """Verify that the saved tensor is packed to 4 bits, then register the export manifest."""
        with self._stage(variant_id, method, calibration_texts, self.quantization_python, "quantization_worker.py",
                         group_size=group_size) as (output, report, original_size, calibration):
            config = json.loads((output / "config.json").read_text(encoding="utf-8"))["quantization_config"]
            groups = [group["weights"] for group in config["config_groups"].values()]
            if (config["quant_method"] != "compressed-tensors" or config["format"] != "pack-quantized"
                    or not groups or any(item["num_bits"] != 4 or item["group_size"] != group_size
                                         or item["symmetric"] != report["symmetric"] for item in groups)):
                raise ValueError("Export has no matching compressed-tensors 4-bit configuration")
            packed, zero_points = set(), set()
            files = sorted(output.glob("*.safetensors"))
            for path in files:
                with safe_open(path, framework="pt", device="cpu") as tensors:
                    packed.update(name.removesuffix(".weight_packed") for name in tensors.keys()
                                  if name.endswith(".weight_packed") and tensors.get_slice(name).get_dtype() == "I32")
                    zero_points.update(name.removesuffix(".weight_zero_point") for name in tensors.keys()
                                       if name.endswith(".weight_zero_point"))
            if not packed or len(packed) != report["expected_quantized_linears"]:
                raise ValueError("Expected one packed integer weight tensor per targeted linear layer")
            if not report["symmetric"] and zero_points != packed:
                raise ValueError("Asymmetric quantization requires a zero-point tensor for every packed weight")
            size = sum(path.stat().st_size for path in files)
            if not 0 < size < original_size:
                raise ValueError("Packed weights must occupy fewer bytes than the source checkpoint")
            record = {"variant_id": variant_id, "method": method, "artifact_format": "compressed-tensors",
                      "original_parameters": report["parameters"], "parameters": report["parameters"],
                      "layer_count": report["layer_count"], "original_checkpoint_bytes": original_size,
                      "checkpoint_bytes": size, "details": {"bits": 4, "group_size": group_size,
                          "packed_tensors": len(packed), "zero_point_tensors": len(zero_points),
                          "symmetric": report["symmetric"], **calibration,
                          "weight_files_sha256": {path.name: sha(path) for path in files},
                          "runtime": report["runtime"]}}
            return publish_checkpoint(self.tools.versions, output, record)

    def slice(self, variant_id, calibration_texts, target_hidden_size):
        """Verify the new hidden width and weight files before publishing an artifact that SliceGPT can reload."""
        tools = self.tools
        width = tools.forward.backend.hidden_size
        if not 0 < target_hidden_size < width or target_hidden_size % 8:
            raise ValueError("SliceGPT target width must be a positive multiple of 8 below the original width")
        with self._stage(variant_id, "slicegpt", calibration_texts, self.slicegpt_python, "slicegpt_worker.py",
                         target_hidden_size=target_hidden_size) as (output, report, original_size, calibration):
            weights = output / "model.safetensors"
            with safe_open(weights, framework="pt", device="cpu") as tensors:
                dimensions = tensors.get_slice("model.embed_tokens.weight").get_shape()
            if dimensions[1] != target_hidden_size or report["parameters"] >= report["original_parameters"]:
                raise ValueError("SliceGPT must reach the target width and reduce total parameters")
            tools.forward.backend.tokenizer.save_pretrained(output)
            record = {"variant_id": variant_id, "method": "slicegpt", "artifact_format": "slicegpt",
                      "original_parameters": report["original_parameters"], "parameters": report["parameters"],
                      "layer_count": report["layer_count"], "original_checkpoint_bytes": original_size,
                      "checkpoint_bytes": weights.stat().st_size, "details": {"hidden_size": target_hidden_size,
                          "original_hidden_size": width, **calibration, "runtime": report["runtime"],
                          "weight_files_sha256": {weights.name: sha(weights)},
                          "rotation_source_sha256": report["rotation_source_sha256"]}}
            return publish_checkpoint(tools.versions, output, record)
