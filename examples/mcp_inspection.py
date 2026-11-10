"""Demonstrate Python and MCP read-only tools with a real local model."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from weight_programming import open_inspection


async def inspect_tools(call):
    """Each tool is simply executed once, retaining weights and activation slices for comparison."""
    model = await call("describe_model", {})
    site = "block.0.residual_out"
    assert any(item["site_id"] == site for item in model["sites"])
    variant_id = model["variant_id"]
    run = await call("model_forward", {"variant_id": variant_id, "mode": "teacher_forced",
                                      "texts": ["The capital of France is"], "observe": [site]})
    activation = run["artifacts"]["hidden_1"]
    sliced = await call("read_artifact", {"artifact_id": activation["artifact_id"],
                                         "region": [[0, 1], [0, 1], [0, 4]]})
    assert sliced["site_id"] == site and np.asarray(sliced["values"]).shape == (1, 1, 4)
    listed = await call("list_tensors", {"variant_id": variant_id, "layer": 0, "limit": 2})
    tensor = next(item for item in listed["tensors"] if len(item["shape"]) == 2)
    weight = await call("read_tensor", {"variant_id": variant_id, "name": tensor["name"],
                                       "region": [[0, 1], [0, 4]]})
    assert weight["dtype"] == tensor["dtype"] and np.asarray(weight["values"]).shape == (1, 4)
    status = await call("get_run", {"run_id": run["run_id"]})
    assert status["status"] == "succeeded" and status["artifacts"]["hidden_1"] == activation["artifact_id"]
    generation = await call("model_forward", {"variant_id": variant_id, "mode": "generate",
                                              "prompts": ["What is the capital of France?"],
                                              "observe": [site], "max_new_tokens": 3})
    page = await call("read_artifact", {"artifact_id": generation["artifacts"]["original"]["artifact_id"],
                                       "limit": 1})
    sample = page["records"][0]
    replay = await call("read_artifact", {"artifact_id": sample["activations"]["hidden_1"]["prompt_last"],
                                         "region": [[0, 4]]})
    assert replay["site_id"] == site and len(replay["values"]) == 4
    return {"model_type": model["model_type"], "layer_count": model["layer_count"],
            "activation_shape": activation["shape"], "activation_slice": sliced["values"],
            "weight_dtype": weight["dtype"], "weight_slice": weight["values"],
            "generated_tokens": sample["generated_token_ids"], "replayed_slice": replay["values"]}


async def main(model):
    with TemporaryDirectory(prefix="wsp-inspection-") as workspace:
        folder = Path(workspace)
        tools = open_inspection(model, folder / "python")

        async def python_call(name, arguments):
            return tools.registry.call(name, arguments)

        direct = await inspect_tools(python_call)
        server = StdioServerParameters(command=sys.executable,
                                       args=["-m", "weight_programming.mcp_server", "--model", model,
                                             "--workspace", str(folder / "mcp")])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                discovered = {tool.name for tool in (await session.list_tools()).tools}
                assert discovered == {spec.name for spec in tools.registry.list_tools()}

                async def mcp_call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                remote = await inspect_tools(mcp_call)
        assert direct == remote
        print(json.dumps({"tools": sorted(discovered), **direct}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.model))
