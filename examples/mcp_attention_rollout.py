"""用真实检查点演示复用注意力快照和热力图。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import numpy as np


async def main(model):
    """比较首层计算与原始注意力，并查看全层传播图。"""
    with TemporaryDirectory(prefix="wsp-rollout-") as workspace:
        server = StdioServerParameters(command=sys.executable,
                                       args=["-m", "weight_programming.mcp_server", "--model", model,
                                             "--workspace", workspace])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                description = await call("describe_model", {})
                layers = description["trace_circuit_methods"]["attention_rollout"]["through_layers"]
                run = await call("model_forward", {"variant_id": description["variant_id"], "mode": "teacher_forced",
                                                   "texts": ["The capital of France is", "The river flows into the sea."],
                                                   "observe": [f"block.{i}.attention.probs" for i in layers]})
                request = {"run_id": run["run_id"], "method": "attention_rollout"}
                first = await call("trace_circuit", {**request, "through_layer": 0})
                n = len(first["token_ids"])
                heads = run["artifacts"]["attention_0"]["shape"][1]
                raw = await call("read_artifact", {"artifact_id": first["source_artifact_ids"][0],
                                                   "region": [[0, 1], [0, heads], [0, n], [0, n]]})
                first_values = await call("read_artifact", {"artifact_id": first["artifact_id"]})
                expected = (np.asarray(raw["values"])[0].mean(0) + np.eye(n)) / 2
                np.testing.assert_allclose(first_values["values"], expected, atol=1e-7)

                result = await call("trace_circuit", {**request, "through_layer": layers[-1]})
                values = await call("read_artifact", {"artifact_id": result["artifact_id"]})
                np.testing.assert_allclose(np.sum(values["values"], axis=1), 1, atol=1e-12)
                assert len(result["source_artifact_ids"]) == len(layers)
                figure = await call("render_evidence", {"artifact_id": result["artifact_id"], "view": "attention_heatmap"})
                assert (Path(workspace) / figure["plot_path"]).is_file()
                query = result["token_positions"][-1]
                flow_request = {**request, "method": "attention_flow", "query_position": query}
                first_flow = await call("trace_circuit", {**flow_request, "through_layer": 0})
                np.testing.assert_allclose(first_flow["raw_flow"], expected[-1], atol=1e-7)
                flow = await call("trace_circuit", {**flow_request, "through_layer": layers[-1]})
                flow_values = await call("read_artifact", {"artifact_id": flow["artifact_id"]})
                np.testing.assert_allclose(flow_values["values"][0], np.asarray(flow["raw_flow"]) / sum(flow["raw_flow"]))
                flow_figure = await call("render_evidence", {"artifact_id": flow["artifact_id"], "view": "attention_heatmap"})
                assert flow_figure["top_cells"][0]["token_positions"]["query_token"] == query
                print(json.dumps({"model_type": description["model_type"], "observed_layers": len(layers),
                                  "rollout_shape": figure["shape"], "flow_shape": flow_figure["shape"],
                                  "evidence_type": result["evidence_type"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    asyncio.run(main(parser.parse_args().model))
