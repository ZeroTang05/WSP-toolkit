"""Compare path replacement, ACDC edge deletion, and EAP gradient estimation on real models."""

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


async def main(model):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)

    def pair(clean, corrupt, target, alternative):
        target_id, = tokenizer.encode(" " + target, add_special_tokens=False)
        alternative_id, = tokenizer.encode(" " + alternative, add_special_tokens=False)
        return {"clean_text": f"The capital of {clean} is", "corrupt_text": f"The capital of {corrupt} is",
                "target_token": target_id, "alternative_token": alternative_id}

    discovery = pair("France", "Italy", "Paris", "Rome")
    review = pair("Germany", "Spain", "Berlin", "Madrid")
    with TemporaryDirectory(prefix="wsp-circuits-") as workspace:
        server = StdioServerParameters(command=sys.executable, args=["-m", "weight_programming.mcp_server",
            "--model", model, "--workspace", workspace, "--allow-interventions", "--allow-gradients"])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                model_info = await call("describe_model", {})
                last = f"block.{model_info['layer_count'] - 1}.mlp"
                path = ["embedding.out", "block.0.attention", "block.0.mlp", "output"]
                assert set(path) <= set(model_info["circuit_nodes"])
                common = {"variant_id": model_info["variant_id"], "pairs": [discovery]}
                path_result = await call("trace_circuit", {**common, "method": "path_patching", "path": path,
                                                           "review_pairs": [review]})
                assert path_result["usage"]["executed_model_forwards"] == 6
                assert path_result["mean_effect"] != 0 and math.isfinite(path_result["review_mean_effect"])

                direct = {"source": last, "target": "output"}
                edges = [direct, {"source": "embedding.out", "target": "block.0.attention"},
                         {"source": "block.0.attention", "target": "block.0.mlp"},
                         {"source": "block.0.mlp", "target": "output"}]
                estimate = await call("trace_circuit", {**common, "method": "eap", "edges": edges})
                assert estimate["usage"]["executed_model_forwards"] == 2 and estimate["usage"]["executed_model_backwards"] == 1
                edge_score = next(row["score"] for row in estimate["top_edges"] if row["source"] == last)
                strength = 0.01
                small_change = await call("trace_circuit", {**common, "method": "path_patching", "path": [last, "output"],
                                                           "strength": strength})
                local_slope = small_change["mean_effect"] / strength
                assert math.isclose(edge_score, local_slope, rel_tol=0.03, abs_tol=0.03), (edge_score, local_slope)

                prune = await call("trace_circuit", {**common, "method": "acdc", "edges": edges,
                                                      "threshold": 0.01, "review_pairs": [review]})
                assert len(prune["kept_edges"]) + prune["removed_edges"] == len(edges)
                records = await call("read_artifact", {"artifact_id": prune["artifact_id"]})
                checked, = [row for row in records["records"] if row["stage"] == "review"]
                assert all(math.isfinite(checked[key]) for key in ("retained_score", "removed_score", "random_retained_score"))
                graph = await call("render_evidence", {"artifact_id": estimate["graph_artifact_id"], "view": "graph"})
                assert graph["evidence_type"] == "gradient_estimate"
                assert (Path(workspace) / graph["plot_path"]).is_file()
                circuit = await call("render_evidence", {"artifact_id": prune["graph_artifact_id"], "view": "graph"})
                assert circuit["evidence_type"] == "intervention_effect" and circuit["edges_rendered"] == len(edges)
                print(json.dumps({"path_effect": path_result["mean_effect"], "path_review": path_result["review_mean_effect"],
                    "eap_edge": edge_score, "exact_local_slope": local_slope, "acdc_kept": len(prune["kept_edges"]),
                    "acdc_removed": prune["removed_edges"], "acdc_review": checked}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    asyncio.run(main(parser.parse_args().model))
