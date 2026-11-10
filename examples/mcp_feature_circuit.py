"""Fit small dictionaries with real models and generate feature attribution plots from MCP."""

import argparse
import asyncio
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from transformers import AutoTokenizer


async def main(model, dataset, tracer_python):
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    target, = tokenizer.encode(" Paris", add_special_tokens=False)
    training = Path(__file__).with_name("fit_per_layer_transcoders.py")
    transcoder_id = "smollm2-local-fit"
    with TemporaryDirectory(prefix="wsp-feature-circuit-") as workspace:
        server = StdioServerParameters(command=sys.executable, args=["-m", "weight_programming.mcp_server",
            "--model", model, "--workspace", workspace, "--allow-gradients", "--circuit-tracer-python", tracer_python])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                info = await call("describe_model", {})
                bundle = Path(workspace) / "transcoders" / transcoder_id
                subprocess.run([tracer_python, str(training), "--model", model, "--dataset", str(dataset),
                                "--output", str(bundle), "--variant-id", info["variant_id"]], check=True)
                info = await call("describe_model", {})
                assert transcoder_id in info["feature_transcoders"]
                result = await call("trace_circuit", {"method": "feature_attribution_graph", "variant_id": info["variant_id"],
                    "transcoder_id": transcoder_id, "text": "The capital of France is", "target_tokens": [target],
                    "max_feature_nodes": 32, "node_threshold": 0.98})
                graph, = (await call("read_artifact", {"artifact_id": result["graph_artifact_id"]}))["records"]
                assert {"feature", "logit", "reconstruction_error"} <= {node["kind"] for node in graph["nodes"]}
                indices = {node["id"]: node["matrix_index"] for node in graph["nodes"]}
                edge = graph["edges"][0]
                row, column = indices[edge["target"]], indices[edge["source"]]
                readback = await call("read_artifact", {"artifact_id": result["artifact_id"],
                                                        "region": [[row, row + 1], [column, column + 1]]})
                assert readback["values"] == [[edge["score"]]]
                figure = await call("render_evidence", {"artifact_id": result["graph_artifact_id"], "view": "graph"})
                assert figure["evidence_type"] == "replacement_model_attribution"
                assert (Path(workspace) / figure["plot_path"]).is_file()
                scores = result["fidelity"]
                assert math.isfinite(scores["kl_original_to_replacement"])
                report, = (await call("read_artifact", {"artifact_id": result["report_artifact_id"]}))["records"]
                print(json.dumps({"nodes": result["node_count"], "features": report["selected_features"],
                    "edges_shown": result["edges_shown"], "kl_original_to_replacement": scores["kl_original_to_replacement"],
                    "targets": scores["targets"], "graph_scores": report["graph_scores"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--circuit-tracer-python", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.dataset, args.circuit_tracer_python))
