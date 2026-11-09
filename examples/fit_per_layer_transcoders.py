"""用真实前馈输入和输出拟合一组小型 Circuit Tracer 字典。"""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from itertools import islice
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder
from transformers import AutoModelForCausalLM, AutoTokenizer


def fit(model_path, dataset_path, destination, variant_id):
    """为示例保存一层一份的稀疏预测器和模型绑定清单。"""
    torch.manual_seed(14)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    layers, width = model.model.layers, model.config.hidden_size
    inputs, targets = [[] for _ in layers], [[] for _ in layers]
    with dataset_path.open(encoding="utf-8") as stream:
        texts = [json.loads(line)["problem"] for line in islice(stream, 32)]

    def capture(index):
        def hook(module, args, value):
            inputs[index].append(args[0].reshape(-1, width).clone())
            targets[index].append(value.reshape(-1, width).clone())
        return hook

    with torch.no_grad(), ExitStack() as hooks:
        for index, layer in enumerate(layers):
            handle = layer.mlp.register_forward_hook(capture(index))
            hooks.callback(handle.remove)
        for text in texts:
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :128]
            model(ids, use_cache=False, logits_to_keep=1)

    destination.mkdir(parents=True)
    digests = {}
    for index, (rows, outputs) in enumerate(zip(inputs, targets)):
        x, y = torch.cat(rows), torch.cat(outputs)
        transcoder = SingleLayerTranscoder(width, 64, F.relu, index, device=torch.device("cpu"), dtype=torch.float32)
        with torch.no_grad():
            transcoder.W_enc.normal_(std=width ** -0.5)
            transcoder.W_dec.normal_(std=0.01)
            transcoder.b_dec.copy_(y.mean(0))
        scale = (y - transcoder.b_dec.detach()).square().mean().clamp_min(1e-12)
        optimizer = torch.optim.Adam(transcoder.parameters(), lr=0.003)
        # 重建真实 MLP 输出，同时鼓励只有少量特征发出信号。
        for _ in range(60):
            pick = torch.randint(len(x), (min(128, len(x)),))
            features = transcoder.encode(x[pick])
            predicted = transcoder.decode(features)
            loss = (predicted - y[pick]).square().mean() / scale + 0.03 * features.abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        path = destination / f"layer_{index}.safetensors"
        transcoder.to_safetensors(str(path))
        digests[path.name] = sha256(path.read_bytes()).hexdigest()
    manifest = {"format_version": 1, "model_kind": "transcoder_set", "variant_id": variant_id,
                "feature_input_hook": "mlp.hook_in", "feature_output_hook": "mlp.hook_out", "weights_sha256": digests,
                "provenance": {"calibration_texts_sha256": sha256(json.dumps(texts).encode()).hexdigest(),
                               "samples": len(texts), "features_per_layer": 64, "seed": 14, "steps": 60}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant-id", required=True)
    arguments = parser.parse_args()
    fit(arguments.model, arguments.dataset, arguments.output, arguments.variant_id)
