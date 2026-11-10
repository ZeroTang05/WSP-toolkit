"""Validate classification scores, encoder snapshots, probes, and weight patches with real SST-2 samples."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


async def main(model, dataset, comparison_model=None):
    """Use the same tools on a real classifier and compare each result with direct Transformers inference."""
    groups = {0: [], 1: []}
    with dataset.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if len(groups[row["label"]]) < 8:
                groups[row["label"]].append(row)
            if all(len(values) == 8 for values in groups.values()):
                break
    if any(len(values) < 8 for values in groups.values()):
        raise ValueError("Provide at least eight real examples of each SST-2 label")
    rows = [row for pair in zip(groups[0], groups[1]) for row in pair]
    texts, labels = [row["sentence"] for row in rows], [str(row["label"]) for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    direct = AutoModelForSequenceClassification.from_pretrained(model, local_files_only=True,
                                dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        expected = torch.cat([direct(**tokenizer(text, return_tensors="pt")).logits for text in texts]).numpy()
        first = direct(**tokenizer(texts[0], return_tensors="pt"), output_hidden_states=True, output_attentions=True)
    with TemporaryDirectory(prefix="wsp-classification-") as workspace:
        args = ["-m", "weight_programming.mcp_server", "--model", str(model), "--workspace", workspace, "--allow-edits"]
        if comparison_model is not None:
            args.extend(["--comparison-model", str(comparison_model)])
        server = StdioServerParameters(command=sys.executable, args=args)
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    result = await session.call_tool(name, arguments)
                    if result.is_error:
                        raise RuntimeError(result.structured_content or result.content)
                    return result.structured_content

                info = await call("describe_model", {})
                variant = info["variant_id"]
                assert info["modes"] == ["predict"] and info["lens_methods"] == []
                available = {tool.name for tool in (await session.list_tools()).tools}
                assert "probe_representation" in available and "scan_lens" not in available
                final_site = f"block.{info['layer_count'] - 1}.residual_out"
                run = await call("model_forward", {"mode": "predict", "variant_id": variant, "texts": texts,
                    "observe": ["embedding.out", final_site, "block.0.mlp.down_proj.input"]})
                scores = await call("read_artifact", {"artifact_id": run["artifacts"]["class_logits"]["artifact_id"]})
                np.testing.assert_allclose(scores["values"], expected, rtol=1e-6, atol=1e-6)
                snapshot = run["artifacts"][f"hidden_{info['layer_count']}"]["artifact_id"]
                cell = await call("read_artifact", {"artifact_id": snapshot, "region": [[0, 1], [0, 1], [0, 8]]})
                np.testing.assert_allclose(cell["values"][0][0], first.hidden_states[-1][0, 0, :8], atol=1e-6)
                probe = await call("probe_representation", {"artifact_id": snapshot, "labels": labels,
                    "train_indices": list(range(8)), "evaluation_indices": list(range(8, 16)), "token_position": 0,
                    "input_baseline_artifact_id": run["artifacts"]["hidden_0"]["artifact_id"]})
                figure = await call("render_evidence", {"artifact_id": snapshot, "view": "activation_heatmap"})
                assert (Path(workspace) / figure["plot_path"]).is_file()
                observed = await call("model_forward", {"mode": "predict", "variant_id": variant, "texts": texts[:1],
                    "observe": [f"block.{i}.attention.probs" for i in range(info["layer_count"])]})
                length = len(tokenizer.encode(texts[0]))
                attention = await call("read_artifact", {"artifact_id": observed["artifacts"]["attention_0"]["artifact_id"],
                    "region": [[0, 1], [0, 1], [0, length], [0, length]]})
                np.testing.assert_allclose(attention["values"][0][0], first.attentions[0][0, 0], atol=1e-6)
                rollout = await call("trace_circuit", {"run_id": observed["run_id"], "method": "attention_rollout"})
                assert rollout["through_layer"] == info["layer_count"] - 1
                patch = await call("create_patch", {"parent_variant_id": variant, "method": "classifier_bias_example",
                    "edits": [{"tensor": "classifier.bias", "op": "add", "values": [0.125, -0.125]}]})
                candidate = await call("apply_patch", {"parent_variant_id": variant, "patch_id": patch["patch_id"]})
                changed = await call("model_forward", {"mode": "predict", "variant_id": candidate["variant_id"],
                    "texts": texts[:1]})
                actual = await call("read_artifact", {"artifact_id": changed["artifacts"]["class_logits"]["artifact_id"]})
                np.testing.assert_allclose(actual["values"], expected[:1] + [0.125, -0.125], atol=1e-6)
                cka = await call("compare_models", {"left_variant_id": variant, "right_variant_id": candidate["variant_id"],
                    "views": ["weights", "representations"], "texts": texts[:4], "left_sites": [final_site],
                    "right_sites": [final_site], "tensors": ["classifier.bias"]})
                matrix = await call("read_artifact", {"artifact_id": cka["representations"]["artifact_id"]})
                np.testing.assert_allclose(matrix["values"], [[1.0]], atol=1e-12)
                cross_architecture = None
                if comparison_model is not None:
                    compared = await call("compare_models", {"left_variant_id": variant,
                        "right_variant_id": info["comparison_variant_id"], "right_workspace": "comparison",
                        "views": ["representations"], "texts": texts[:8], "left_sites": [final_site],
                        "right_sites": ["block.0.residual_out"]})
                    shared = await call("read_artifact", {"artifact_id": compared["representations"]["artifact_id"]})
                    cross_architecture = shared["values"][0][0]
                    assert 0 <= cross_architecture <= 1
                print(json.dumps({"model_type": info["model_type"], "labels": info["label_names"], "samples": len(texts),
                    "direct_logits_max_error": float(np.max(np.abs(np.asarray(scores["values"]) - expected))),
                    "probe_accuracy": probe["accuracy"], "control_accuracy": probe["control_accuracy"],
                    "attention_layers": info["layer_count"], "weight_patch_verified": True,
                    "cross_architecture_cka": cross_architecture}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path, help="SST-2 training JSONL with sentence and label")
    parser.add_argument("--comparison-model", type=Path, help="Optional Qwen3 or Llama checkpoint for cross-architecture CKA")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.dataset, args.comparison_model))
