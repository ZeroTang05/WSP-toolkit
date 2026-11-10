"""Validating persistent Python and patch proposals on Linux with real checkpoints."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(model):
    """Save a weight slice in two Python cells and submit a replayable patch."""
    with TemporaryDirectory(prefix="wsp-python-") as workspace:
        server = StdioServerParameters(command=sys.executable,
                                       args=["-m", "weight_programming.mcp_server", "--model", model,
                                             "--workspace", workspace, "--allow-python", "--allow-edits"])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as client:
                await client.initialize()

                async def call(name, arguments):
                    result = await client.call_tool(name, arguments)
                    if result.is_error:
                        raise RuntimeError(result.structured_content or result.content)
                    return result.structured_content

                model_info = await call("describe_model", {})
                base = model_info["variant_id"]
                assert model_info["python_execution"]
                listed = await call("list_tensors", {"variant_id": base, "layer": 0})
                target = next(item["name"] for item in listed["tensors"] if item["name"].endswith("mlp.down_proj.weight"))
                region = [[0, 1], [0, 1]]
                common = {"session_id": "analysis", "variant_id": base}
                first = await call("execute_python", {**common,
                                                       "code": f"weight = load({target!r}, {region!r}); float(weight[0, 0])"})
                assert first["status"] == "succeeded", first.get("output")
                original = await call("read_tensor", {"variant_id": base, "name": target, "region": region})
                assert float(first["output"]) == original["values"][0][0]

                observed = await call("model_forward", {"variant_id": base, "mode": "teacher_forced",
                                                        "texts": ["The capital of France is"], "observe": ["embedding.out"]})
                second = await call("execute_python", {**common,
                    "code": "activation = read_array('activation'); print(activation.shape); save_array('proposal', weight.float() + 0.5)",
                    "input_artifacts": {"activation": observed["artifacts"]["hidden_0"]["artifact_id"]},
                    "outputs": {"proposal": ["row", "column"]},
                    "edits": [{"tensor": target, "op": "replace", "region": region, "value": "proposal"}]})
                assert second["status"] == "succeeded", second.get("output")
                proposal = await call("read_artifact", {"artifact_id": second["artifacts"]["proposal"]})
                assert proposal["values"] != original["values"] and proposal["axes"] == ["row", "column"]
                log = await call("get_run", {"run_id": second["run_id"]})
                assert log["usage"]["executions"] == 2 and log["patch_id"] == second["patch_id"]
                assert (Path(workspace) / "python" / "runs" / (second["run_id"] + ".py")).is_file()
                candidate = await call("apply_patch", {"parent_variant_id": base, "patch_id": second["patch_id"]})
                variant = candidate["variant_id"]
                current = await call("execute_python", {"session_id": "candidate", "variant_id": variant,
                    "code": f"float(load({target!r}, {region!r})[0, 0])"})
                applied = await call("read_tensor", {"variant_id": variant, "name": target, "region": region})
                assert current["status"] == "succeeded", current.get("output")
                assert float(current["output"]) == applied["values"][0][0] != float(first["output"])
                isolation = await call("execute_python", {**common, "code": "open('/model/write-check', 'w')"})
                assert isolation["status"] == "failed" and "Read-only file system" in isolation["output"]
                print(json.dumps({"base_value": float(first["output"]), "candidate_value": float(current["output"]),
                                  "persistent_cells": log["usage"]["executions"], "input_shape": second["output"].strip(),
                                  "read_only_model": True}))
        assert not list((Path(workspace) / "python").glob("session-*"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    asyncio.run(main(parser.parse_args().model))
