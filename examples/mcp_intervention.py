"""Measure activation replacement, ablation, and candidate layer scanning with real models."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoTokenizer


async def main(model, render, atp_star):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    clean, corrupt = "The capital of France is", "The capital of Italy is"
    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(corrupt, add_special_tokens=False)
    assert len(clean_ids) == len(corrupt_ids)
    target, = tokenizer.encode(" Paris", add_special_tokens=False)
    alternative, = tokenizer.encode(" Rome", add_special_tokens=False)
    pair = {"clean_text": clean, "corrupt_text": corrupt,
            "positions": [i for i, (a, b) in enumerate(zip(clean_ids, corrupt_ids)) if a != b],
            "target_token": target, "alternative_token": alternative}
    with TemporaryDirectory(prefix="wsp-intervention-") as workspace:
        args = ["-m", "weight_programming.mcp_server", "--model", model, "--workspace", workspace, "--allow-interventions"]
        if atp_star:
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
                variant = model_info["variant_id"]
                request = {"variant_id": variant, "site_id": "embedding.out", "pair": pair}
                replaced = await call("run_intervention", request)
                assert replaced["patched"] == replaced["clean"]
                ablated = await call("run_intervention", {**request, "method": "ablation"})
                assert math.isfinite(ablated["patched"]["logit_difference"])
                snapshots = await call("model_forward", {"variant_id": variant, "mode": "teacher_forced",
                                                         "texts": [clean, corrupt], "observe": ["embedding.out"]})
                position, = pair["positions"]
                embeddings = snapshots["artifacts"]["hidden_0"]
                width = embeddings["shape"][-1]
                aligned = []
                for sample in range(2):
                    region = [[sample, sample + 1], [position, position + 1], [0, width]]
                    row = await call("read_artifact", {"artifact_id": embeddings["artifact_id"], "region": region})
                    aligned.append(row["values"][0][0])
                direction = [positive - negative for positive, negative in zip(*aligned)]
                strength = math.sqrt(sum(value * value for value in direction))
                steered = await call("run_intervention", {**request, "method": "direction",
                                                          "direction": direction, "strength": strength})
                assert abs(steered["patched"]["logit_difference"] - replaced["clean"]["logit_difference"]) < 1e-5
                scan = await call("localize_behavior", {"variant_id": variant, "pairs": [pair],
                                                        "site_ids": ["embedding.out", "block.0.residual_out"]})
                assert scan["usage"]["executed_model_forwards"] == 4
                atp = None
                if atp_star:
                    assert "atp_star" in model_info["localization_methods"]
                    review = {**pair, "clean_text": "The capital of Germany is", "corrupt_text": "The capital of Spain is",
                              "target_token": tokenizer.encode(" Berlin", add_special_tokens=False)[0],
                              "alternative_token": tokenizer.encode(" Madrid", add_special_tokens=False)[0]}
                    common = {"variant_id": variant, "pairs": [pair], "review_pairs": [review]}
                    candidates = ["embedding.out", *[f"block.{layer}.mlp.down_proj.input" for layer in range(7)]]
                    report = await call("localize_behavior", {**common, "site_ids": candidates, "method": "atp_star",
                                        "verify_top_k": 2, "forward_budget": 10})
                    estimates = await call("read_artifact", {"artifact_id": report["prefilter"]["artifact_id"]})
                    assert estimates["shape"] == [1, len(candidates), 2]
                    assert estimates["estimator_names"] == ["plain_atp", "grad_drop_magnitude"]
                    assert report["usage"]["executed_model_backwards"] == model_info["layer_count"] + 1
                    verified = [row["site_id"] for row in report["prefilter"]["ranking"][:2]]
                    exact = await call("localize_behavior", {**common, "site_ids": verified})
                    assert report["ranked_sites"] == exact["ranked_sites"]
                    atp = {"candidates": len(candidates), "verified": verified,
                           "review_effects": [row["review_mean_effect"] for row in report["ranked_sites"]],
                           "forward_calls": report["usage"]["executed_model_forwards"],
                           "backward_calls": report["usage"]["executed_model_backwards"]}
                values = await call("read_artifact", {"artifact_id": scan["artifact_id"]})
                assert values["shape"] == [1, 2]
                if render:
                    plot = await call("render_evidence", {"artifact_id": scan["artifact_id"],
                                                         "view": "patching_heatmap"})
                    assert plot["shape"] == [1, 2]
                    assert plot["top_cells"][0]["value"] == max(values["values"][0])
                    image = Path(workspace) / plot["plot_path"]
                    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                    observed = await call("model_forward", {"variant_id": variant, "mode": "teacher_forced",
                                                            "texts": [clean],
                                                            "observe": ["block.0.residual_out", "block.0.attention.probs"]})
                    for key, view in [("hidden_1", "activation_heatmap"), ("attention_0", "attention_heatmap")]:
                        figure = await call("render_evidence", {"artifact_id": observed["artifacts"][key]["artifact_id"],
                                                               "view": view})
                        assert figure["bytes"] > 0
                print(json.dumps({"replacement": replaced["effects"]["logit_difference"],
                                  "ablation": ablated["effects"]["logit_difference"],
                                  "direction": steered["effects"]["logit_difference"],
                                  "scan_forwards": 4, "ranked_sites": scan["ranked_sites"],
                                  "rendered_heatmaps": render, "atp_star": atp}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--render", action="store_true", help="Also render the saved activation and effect arrays")
    parser.add_argument("--atp-star", action="store_true", help="Run GradDrop and verify the shortlisted sites")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.render, args.atp_star))
