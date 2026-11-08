"""用宿主登记的稀疏字典分析已保存的模型激活。"""

import json
import math
import re

import numpy as np
from safetensors.torch import load_file
import torch

from .tensor_store import sha


_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
_MAX_DICTIONARY_BYTES = 512 * 1024**2
_MAX_LATENT_VALUES = 16 * 1024**2


class FeatureTools:
    """复用激活快照和产物服务，按版本、站点和词元坐标对齐。"""

    def __init__(self, tools):
        self.tools = tools
        self.root = tools.versions.root / "features"

    def _manifest(self, encoder_id):
        """读取由宿主放在固定工作目录中的字典清单。"""
        if not re.fullmatch(_ID_PATTERN, encoder_id):
            raise ValueError("Invalid registered encoder ID")
        folder = self.root / encoder_id
        path = folder / "manifest.json"
        if folder.resolve().parent != self.root.resolve() or path.resolve().parent != folder:
            raise ValueError("Feature dictionary must be within its registered directory")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Feature dictionary manifest must be an object")
        count = {"sae": 1, "crosscoder": 2}.get(record.get("method"))
        sources = record.get("sources", [])
        if (record.get("format_version") != 1 or count is None or not isinstance(sources, list) or len(sources) != count
                or record.get("activation") not in ("relu", "topk", "jumprelu")):
            raise ValueError("Unsupported feature dictionary format, method or activation")
        for source in sources:
            if (not isinstance(source, dict) or not all(isinstance(source.get(key), str) and source[key]
                                                       for key in ("variant_id", "site_id"))
                    or type(source.get("center_on_decoder_bias", False)) is not bool):
                raise ValueError("Each dictionary source needs a model variant and activation site")
        return folder, record

    def available(self, variant_id):
        """仅列出和所选模型版本相关的字典入口。"""
        found = []
        for path in sorted(self.root.glob("*/manifest.json")):
            _, record = self._manifest(path.parent.name)
            if any(source["variant_id"] == variant_id for source in record["sources"]):
                found.append({"encoder_id": path.parent.name, "method": record["method"],
                              "activation": record["activation"], "sources": record["sources"]})
        return found

    @staticmethod
    def _snapshot(tools, artifact_id, source, sample_indices, token_position):
        """从快照中取有效词元，并验证字典针对的是同一模型与站点。"""
        store = tools.artifacts
        array, info = store.array(artifact_id)
        if info["axes"] != ["sample", "token", "channel"] or info.get("site_id") != source["site_id"]:
            raise ValueError("Dictionary requires a matching sample/token/channel snapshot and site")
        run_id = info["run_id"]
        run = tools.get_run(run_id)
        if run["status"] != "succeeded" or run["variant_id"] != source["variant_id"]:
            raise ValueError("Dictionary was registered for a different model variant")
        valid, _ = store.array(f"{run_id}/valid_tokens")
        token_ids, _ = store.array(f"{run_id}/input_ids")
        selected = list(range(len(valid))) if sample_indices is None else sample_indices
        if not selected or any(not 0 <= sample < len(valid) for sample in selected):
            raise ValueError("Select valid sample indices in the snapshot")
        if token_position is None:
            local_rows, positions = np.nonzero(valid[selected])
            samples = np.asarray(selected)[local_rows]
        else:
            if token_position >= valid.shape[1] or not valid[selected, token_position].all():
                raise ValueError("The selected token position must be valid in every selected sample")
            samples = np.asarray(selected)
            positions = np.full(len(selected), token_position)
        if not 0 < len(samples) <= 2048:
            raise ValueError("Select between one and 2048 valid tokens")
        sequences = [token_ids[sample, valid[sample]].tolist() for sample in selected]
        values = torch.from_numpy(array[samples, positions]).float()
        if not torch.isfinite(values).all():
            raise ValueError("Feature analysis requires finite activations")
        coordinates = list(zip(samples.tolist(), positions.tolist(), token_ids[samples, positions].tolist()))
        return values, coordinates, sequences

    @staticmethod
    def _weights(folder, record, widths):
        """验证 Safetensors 摘要、坐标轴大小及稀疏函数的参数。"""
        path = folder / "weights.safetensors"
        if path.resolve().parent != folder or path.stat().st_size > _MAX_DICTIONARY_BYTES:
            raise ValueError("Dictionary weights must be local and at most 512 MiB")
        if sha(path) != record["weights_sha256"]:
            raise ValueError("Feature dictionary weights differ from the registered digest")
        data = load_file(path, device="cpu")
        bias = data["b_enc"]
        feature_count = bias.numel()
        expected = {"b_enc": (feature_count,)}
        for index, width in enumerate(widths):
            expected.update({f"W_enc.{index}": (width, feature_count),
                             f"W_dec.{index}": (feature_count, width), f"b_dec.{index}": (width,)})
        activation = record["activation"]
        if activation == "jumprelu":
            expected["threshold"] = (feature_count,)
        if (not 0 < feature_count <= 131072 or set(data) != set(expected)
                or any(not value.is_floating_point() or tuple(value.shape) != expected[key]
                       for key, value in data.items())):
            raise ValueError("Dictionary tensors do not match the declared coordinates")
        if not all(torch.isfinite(value).all() for value in data.values()):
            raise ValueError("Dictionary weights must be finite")
        if activation == "topk" and (type(record.get("k")) is not int or not 1 <= record["k"] <= feature_count):
            raise ValueError("TopK dictionary requires a valid k")
        if activation == "jumprelu" and (data["threshold"] < 0).any():
            raise ValueError("JumpReLU thresholds must be nonnegative")
        return {key: value.float() for key, value in data.items()}

    @torch.inference_mode()
    def extract_features(self, encoder_id, artifact_id, method="sae", sample_indices=None, token_position=None,
                         feature_ids=None, top_features=10, top_examples=3, right_artifact_id=None,
                         right_workspace="primary", expected_variants=None):
        """生成特征响应、最强触发位置和逐来源重建误差。"""
        folder, record = self._manifest(encoder_id)
        if record["method"] != method or (method == "crosscoder") != (right_artifact_id is not None):
            raise ValueError("The registered method determines whether one or two snapshots are required")
        sources = record["sources"]
        if expected_variants is not None and [source["variant_id"] for source in sources] != expected_variants:
            raise ValueError("Crosscoder does not match the requested left and right variants")
        snapshot_ids = [artifact_id] if right_artifact_id is None else [artifact_id, right_artifact_id]
        contexts = [self.tools] if method == "sae" else [self.tools, self.tools.workspace_tools(right_workspace)]
        snapshots = [self._snapshot(tools, ref, source, sample_indices, token_position)
                     for tools, ref, source in zip(contexts, snapshot_ids, sources)]
        coordinates, sequences = snapshots[0][1:]
        tokenizer = self.tools.forward.backend.tokenizer
        if method == "crosscoder":
            other_tokenizer = contexts[1].forward.backend.tokenizer
            if ((coordinates, sequences) != snapshots[1][1:]
                    or tokenizer.batch_decode(sequences) != other_tokenizer.batch_decode(sequences)):
                raise ValueError("Crosscoder snapshots must have aligned input token sequences")
        inputs = [snapshot[0] for snapshot in snapshots]
        weights = self._weights(folder, record, [value.shape[1] for value in inputs])
        width = weights["b_enc"].numel()
        if len(coordinates) * width > _MAX_LATENT_VALUES:
            raise ValueError("Feature analysis is too large; select fewer samples or one token position")
        if feature_ids is not None and any(not 0 <= index < width for index in feature_ids):
            raise ValueError("Feature ID outside the registered dictionary")
        scales = [source.get("input_scale", 1.0) for source in sources]
        if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
            raise ValueError("Dictionary input scale must be positive and finite")

        # 每个来源先在自己的坐标编码，再合并成一份特征响应。
        blocks = []
        residual_ss = [0.0] * len(inputs)
        for start in range(0, len(coordinates), 64):
            end = start + 64
            pre = weights["b_enc"]
            for side, (values, scale, source) in enumerate(zip(inputs, scales, sources)):
                rows = values[start:end] * scale
                if source.get("center_on_decoder_bias", False):
                    rows = rows - weights[f"b_dec.{side}"]
                pre = pre + rows @ weights[f"W_enc.{side}"]
            active = pre.relu()
            if record["activation"] == "topk":
                strengths, indices = active.topk(record["k"], dim=1)
                active = torch.zeros_like(active).scatter_(1, indices, strengths)
            elif record["activation"] == "jumprelu":
                active = active * (pre > weights["threshold"])
            blocks.append(active)
            for side, (values, scale) in enumerate(zip(inputs, scales)):
                reconstruction = (active @ weights[f"W_dec.{side}"] + weights[f"b_dec.{side}"]) / scale
                residual_ss[side] += float((reconstruction - values[start:end]).double().square().sum())
        activations = torch.cat(blocks)
        if not torch.isfinite(activations).all() or not all(math.isfinite(error) for error in residual_ss):
            raise ValueError("Dictionary computation produced nonfinite values")
        maximum = activations.amax(dim=0)
        ids = feature_ids if feature_ids is not None else torch.argsort(maximum, descending=True, stable=True)[
            :min(top_features, width)].tolist()
        selected = activations[:, ids]
        rows = [{"sample": sample, "position": position, "token_id": token,
                 "token": tokenizer.decode([token])} for sample, position, token in coordinates]
        features = []
        for column, feature in enumerate(ids):
            strengths = selected[:, column]
            hits = int(strengths.count_nonzero())
            ordered = torch.argsort(strengths, descending=True, stable=True)[:min(hits, top_examples)].tolist()
            norms = [float(weights[f"W_dec.{side}"][feature].norm()) for side in range(len(inputs))]
            norm_sum = sum(norms)
            features.append({"feature_id": feature, "active_rows": hits, "max_activation": float(maximum[feature]),
                             "decoder_norms": norms, "relative_decoder_norms": [norm / norm_sum for norm in norms]
                             if norm_sum else None,
                             "top_examples": [{**rows[index], "activation": float(strengths[index])} for index in ordered]})
        errors = []
        for source, values, error in zip(sources, inputs, residual_ss):
            centered = values.double() - values.double().mean(dim=0)
            variation = float(centered.square().sum())
            errors.append({"variant_id": source["variant_id"], "site_id": source["site_id"],
                           "mse": error / values.numel(), "fraction_variance_unexplained": error / variation if variation else None})
        artifacts = self.tools.artifacts
        workspaces = ["primary", right_workspace][:len(inputs)]
        result_id = artifacts.save_array("feature_activations", selected.numpy(), ["token_row", "feature"],
                                         metric="feature activation", feature_ids=ids, source_artifact_ids=snapshot_ids,
                                         source_workspaces=workspaces, dictionary_sha256=record["weights_sha256"],
                                         encoder_id=encoder_id, method=method)
        return {"encoder_id": encoder_id, "method": method, "feature_count": width, "token_rows": len(rows),
                "mean_active_features": float(activations.count_nonzero()) / len(rows), "feature_ids": ids,
                "source_artifact_ids": snapshot_ids, "source_workspaces": workspaces,
                "dictionary_sha256": record["weights_sha256"], "reconstruction": errors, "artifact_id": result_id,
                "rows_artifact_id": artifacts.save_records("feature_token_rows", rows),
                "report_artifact_id": artifacts.save_records("feature_responses", features), "preview": features[:5]}
