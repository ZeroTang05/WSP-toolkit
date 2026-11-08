"""用真实 GSM8K 开发样本演示配对评测和偏好分差。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from weight_programming import open_inspection
from weight_programming.forward_service import PracticeSet


async def main(model, dataset, cross_checkpoint):
    with Path(dataset).open(encoding="utf-8") as stream:
        observed_cases = [json.loads(next(stream)) for _ in range(3)]
        cases = observed_cases[:2]
    with TemporaryDirectory(prefix="wsp-evaluation-") as workspace:
        server = StdioServerParameters(command=sys.executable,
                                       args=["-m", "weight_programming.mcp_server", "--model", model,
                                             "--workspace", workspace, "--allow-edits", "--development-jsonl",
                                             dataset, "--dataset-id", "gsm8k-train-for-tool-check"])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                base = (await call("describe_model", {}))["variant_id"]
                listed = await call("list_tensors", {"variant_id": base, "layer": 0, "limit": 100})
                tensor = next(row["name"] for row in listed["tensors"] if row["name"].endswith("mlp.down_proj.weight"))
                patch = await call("create_patch", {"parent_variant_id": base, "method": "evaluation_check",
                                                    "edits": [{"tensor": tensor, "op": "add", "region": [[0, 1], [0, 1]],
                                                               "values": [[0.5]]}]})
                variant = (await call("apply_patch", {"parent_variant_id": base, "patch_id": patch["patch_id"]}))["variant_id"]
                report = await call("evaluate_variants", {"baseline_variant_id": base, "candidate_variant_ids": [variant],
                                                          "dataset_id": "gsm8k-train-for-tool-check", "max_new_tokens": 8,
                                                          "case_ids": [case["id"] for case in cases]})
                assert report["comparisons"][0]["graded_problems"] == 2
                assert all(row["graded_samples"] == 2 for row in report["summaries"])
                rows = await call("read_artifact", {"artifact_id": report["summaries"][0]["predictions_artifact_id"]})
                assert len(rows["records"]) == 2
                pair = {"id": cases[0]["id"], "prompt": cases[0]["problem"] + "\nAnswer:",
                        "chosen": " " + cases[0]["answer"], "rejected": " " + cases[1]["answer"]}
                preferences = await call("score_preferences", {"variant_id": variant, "reference_variant_id": base,
                                                                "pairs": [pair]})
                row = preferences["preview"][0]
                assert row["relative_margin"] == row["chosen_score"] - row["rejected_score"] - row["reference_margin"]
                texts = [case["problem"] for case in observed_cases]
                comparison = await call("compare_models", {"left_variant_id": base, "right_variant_id": variant,
                                                           "views": ["weights", "representations"], "texts": texts})
                assert comparison["weights"]["changed_tensors"] == 1
                evidence = comparison["representations"]["artifact_id"]
                cka = await call("read_artifact", {"artifact_id": evidence})
                assert 0 <= cka["values"][0][0] <= 1
                if any(tool.name == "render_evidence" for tool in (await session.list_tools()).tools):
                    image = await call("render_evidence", {"artifact_id": evidence, "view": "similarity_heatmap"})
                    assert (Path(workspace) / image["plot_path"]).is_file()
                result = {"paired_cases": 2, "ci95": report["comparisons"][0]["paired_bootstrap_ci95"],
                          "relative_preference_margin": row["relative_margin"], "same_base_cka": cka["values"][0][0]}
                if cross_checkpoint:
                    saved = await call("export_model", {"variant_id": variant})
        if cross_checkpoint:
            folder = Path(workspace)
            practice = PracticeSet(cases, role="development", name="cross-checkpoint-gsm8k")
            peer = open_inspection(folder / saved["checkpoint_path"], folder / "peer", practice=practice)
            primary = open_inspection(model, folder / "primary", comparison=peer, practice=practice)
            cross = primary.registry.call("compare_models", {"left_variant_id": primary.variant_id,
                                                            "right_variant_id": peer.variant_id,
                                                            "right_workspace": "comparison",
                                                            "views": ["representations", "behavior"], "texts": texts,
                                                            "evaluation": {"dataset_id": practice.name, "case_ids": list(practice.rows),
                                                                           "max_new_tokens": 8}})
            values = primary.registry.call("read_artifact", {"artifact_id": cross["representations"]["artifact_id"]})
            assert values["values"] == cka["values"]
            result["cross_checkpoint_cka"] = values["values"][0][0]
            assert cross["behavior"]["comparisons"][0]["graded_problems"] == 2
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--cross-checkpoint", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.dataset, args.cross_checkpoint))
