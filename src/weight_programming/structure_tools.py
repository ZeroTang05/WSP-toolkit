"""复用深度和前馈通道剪枝，导出可独立重载的模型。"""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from .isolated_model import load_variant, response_logprobs, supervised_ids
from .compression_exports import ExternalCompression, publish_checkpoint


class StructureTools:
    """用观测选择结构，再在独立模型上实际删层或删通道。"""

    def __init__(self, tools, *, quantization_python=None, slicegpt_python=None):
        self.tools = tools
        self.supported = tools.forward.backend.identity["model_type"] in ("llama", "qwen3")
        self.external = ExternalCompression(tools, quantization_python, slicegpt_python)

    def _influence(self, variant_id, texts, drop_layers):
        from depth_programming import block_influence, choose_layers

        tools = self.tools
        layers = tools.forward.backend.num_layers
        sites = ["embedding.out", *(f"block.{i}.residual_out" for i in range(layers))]
        if not 0 < drop_layers < layers:
            raise ValueError("Choose a positive drop count below the model's layer count")
        run = tools.model_forward(variant_id, "teacher_forced", texts=texts, observe=sites)
        refs = run["artifacts"]
        valid, _ = tools.artifacts.array(refs["valid_tokens"]["artifact_id"])
        mask = torch.from_numpy(valid.copy())
        counts = mask.sum(1)
        scores = []
        before, _ = tools.artifacts.array(refs["hidden_0"]["artifact_id"])
        for layer in range(1, layers + 1):
            after, _ = tools.artifacts.array(refs[f"hidden_{layer}"]["artifact_id"])
            changes = block_influence(torch.from_numpy(before.copy()), torch.from_numpy(after.copy()))
            scores.append(float(((changes * mask).sum(1) / counts).mean()))
            before = after
        return choose_layers(scores, drop_layers), {"calibration_run_id": run["run_id"], "block_influence": scores}

    def _taylor(self, model, case_ids, fraction):
        from paper_methods import taylor_mlp_scores
        from weightlab import Lab, cast_floating_parameters

        tools = self.tools
        rows = [tools.training_data.rows[key] for key in case_ids]
        backend = tools.forward.backend
        policy = tools.forward.policy
        pairs = [supervised_ids(backend, row["problem"], row["answer"], policy.max_text_tokens) for row in rows]
        tokens = sum(len(prompt) + len(target) for prompt, target in pairs)
        if tokens > policy.max_total_text_tokens:
            raise ValueError("Taylor calibration exceeds the token budget")
        native_dtype = next(model.parameters()).dtype
        cast_floating_parameters(model, torch.float32)
        layers = model.model.layers
        for layer in layers:
            layer.mlp.requires_grad_(True)
        with torch.enable_grad():
            for prompt, target in pairs:
                (-response_logprobs(model, prompt, target).mean() / len(pairs)).backward()
        scores = taylor_mlp_scores(layers)
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        lab = SimpleNamespace(layers=layers, model=model)
        kept = Lab.masks(lab, scores, fraction)
        Lab.structural_prune(lab, kept)
        cast_floating_parameters(model, native_dtype)
        return {"calibration_dataset_sha256": tools.training_data.identity["sha256"], "case_ids": case_ids,
                "calibration_tokens": tokens, "original_channel_count": scores[0].numel(),
                "intermediate_size": len(kept[0]), "kept_channels": [indices.tolist() for indices in kept]}

    def compress_model(self, variant_id, method, calibration_texts=None, drop_layers=None,
                       calibration_case_ids=None, fraction=None, group_size=128, target_hidden_size=None):
        """输出改变张量形状或包含打包整数权重的检查点。"""
        from depth_programming import depth_variant

        tools = self.tools
        backend = tools.forward.backend
        if not tools.allow_edits:
            raise PermissionError("Structural export requires edit permission")
        if not self.supported:
            raise ValueError("Structural export requires a supported Llama or Qwen3 decoder")
        if method in ("gptq", "awq"):
            return self.external.quantize(variant_id, method, calibration_texts, group_size)
        if method == "slicegpt":
            return self.external.slice(variant_id, calibration_texts, target_hidden_size)
        if method == "block_influence":
            if not calibration_texts or drop_layers is None or calibration_case_ids is not None or fraction is not None:
                raise ValueError("Block influence requires calibration texts and a drop layer count")
            kept, details = self._influence(variant_id, calibration_texts, drop_layers)
        elif method == "taylor_mlp":
            if not calibration_case_ids or fraction is None or calibration_texts is not None or drop_layers is not None:
                raise ValueError("Taylor MLP requires training case IDs and a channel drop fraction")
            if not tools.allow_gradients or tools.training_data is None:
                raise PermissionError("Taylor selection requires gradient permission and registered training data")
        else:
            raise ValueError(f"Unknown structural method: {method}")
        model = load_variant(backend, tools.versions, variant_id)
        original = sum(p.numel() for p in model.parameters())
        if method == "block_influence":
            details["kept_layers"] = kept
            surgery = depth_variant(model, kept)
        else:
            details = self._taylor(model, calibration_case_ids, fraction)
            surgery = nullcontext()
        root = tools.versions.root / "exports" / "structures"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as temporary:
            output = Path(temporary) / "checkpoint"
            with surgery:
                model.save_pretrained(output, safe_serialization=True)
                backend.tokenizer.save_pretrained(output)
                compressed = sum(p.numel() for p in model.parameters())
                layers = model.config.num_hidden_layers
            record = {"variant_id": variant_id, "method": method, "details": details,
                      "original_parameters": original, "parameters": compressed, "layer_count": layers,
                      "checkpoint_bytes": sum(path.stat().st_size for path in output.glob("*.safetensors"))}
            result = publish_checkpoint(tools.versions, output, record)
        preview = {key: value for key, value in details.items() if key != "kept_channels"}
        return {**result, "details": preview}
