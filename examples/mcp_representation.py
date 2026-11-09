"""用真实 GSM8K 样本演示模型层级读出和线性探针。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoTokenizer


async def main(model, dataset, attribution):
    with Path(dataset).open(encoding="utf-8") as stream:
        rows = [json.loads(next(stream)) for _ in range(6)]
    labels = ["even" if int(row["answer"]) % 2 == 0 else "odd" for row in rows]
    with TemporaryDirectory(prefix="wsp-representation-") as workspace:
        args = ["-m", "weight_programming.mcp_server", "--model", model, "--workspace", workspace]
        if attribution:
            args.append("--allow-gradients")
        server = StdioServerParameters(command=sys.executable, args=args)
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                model_info = await call("describe_model", {})
                last = model_info["layer_count"]
                final_site = f"block.{last - 1}.residual_out"
                run = await call("model_forward", {"variant_id": model_info["variant_id"], "mode": "teacher_forced",
                                                   "texts": [row["problem"] for row in rows], "topk": 3,
                                                   "observe": ["embedding.out", "block.0.residual_out", final_site]})
                lens = await call("scan_lens", {"run_id": run["run_id"], "site_ids": ["block.0.residual_out", final_site],
                                                "sample_indices": [0], "topk": 3})
                final = lens["preview"][1]
                position = final["position"]
                scored = await call("read_artifact", {"artifact_id": run["artifacts"]["topk_ids"]["artifact_id"],
                                                     "region": [[0, 1], [position, position + 1], [0, 3]]})
                assert [token["token_id"] for token in final["top_tokens"]] == scored["values"][0][0]
                probe = await call("probe_representation", {"artifact_id": run["artifacts"][f"hidden_{last}"]["artifact_id"],
                                                           "labels": labels, "train_indices": [0, 1, 2, 3],
                                                           "evaluation_indices": [4, 5],
                                                           "input_baseline_artifact_id": run["artifacts"]["hidden_0"]["artifact_id"]})
                assert probe["train_samples"] == 4 and probe["evaluation_samples"] == 2
                assert probe["input_baseline_accuracy"] is not None
                summary = {"matched_final_token_ids": scored["values"][0][0], "train_samples": 4,
                           "evaluation_samples": 2, "control_accuracy": probe["control_accuracy"]}
                if attribution:
                    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
                    paris, = tokenizer.encode(" Paris", add_special_tokens=False)
                    rome, = tokenizer.encode(" Rome", add_special_tokens=False)
                    report = await call("attribute_output", {"variant_id": model_info["variant_id"],
                                                             "text": "The capital of France is",
                                                             "baseline_text": "The capital of Italy is",
                                                             "target_token": paris, "alternative_token": rome, "steps": 24})
                    signed = await call("read_artifact", {"artifact_id": report["artifact_id"]})
                    assert abs(sum(signed["values"][0]) - report["attribution_sum"]) < 1e-6
                    assert abs(report["integration_error"]) < 0.1 * abs(report["score_change"])
                    image = await call("render_evidence", {"artifact_id": report["artifact_id"], "view": "attribution_heatmap"})
                    assert (Path(workspace) / image["plot_path"]).is_file()
                    summary["integration_error"] = report["integration_error"]
                    text, baseline = "The capital of France is", "The capital of Italy is"
                    chosen, rejected = " Paris, France", " Rome"
                    variant = model_info["variant_id"]
                    preferences = await call("score_preferences", {"variant_id": variant, "reference_variant_id": variant,
                        "pairs": [{"id": name, "prompt": prompt, "chosen": chosen, "rejected": rejected}
                                  for name, prompt in (("input", text), ("baseline", baseline))]})
                    attributed = await call("attribute_output", {"method": "contrastive_sequence", "variant_id": variant,
                        "text": text, "baseline_text": baseline, "chosen": chosen, "rejected": rejected, "steps": 24})
                    input_margin, baseline_margin = [row["margin"] for row in preferences["preview"]]
                    assert abs(attributed["input_score"] - input_margin) < 0.05
                    assert abs(attributed["baseline_score"] - baseline_margin) < 0.05
                    assert abs(attributed["integration_error"]) < 0.1 * abs(attributed["score_change"])
                    summary["answer_attribution"] = {"score_change": attributed["score_change"],
                                                     "integration_error": attributed["integration_error"]}
                print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--attribution", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.dataset, args.attribution))
