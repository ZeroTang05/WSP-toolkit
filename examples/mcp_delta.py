"""在真实模型上比较权重差分并压缩一份已有更新。"""

import argparse
import asyncio
import json
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(model):
    with TemporaryDirectory(prefix="wsp-delta-") as workspace:
        server = StdioServerParameters(command=sys.executable,
                                       args=["-m", "weight_programming.mcp_server", "--model", model,
                                             "--workspace", workspace, "--allow-edits"])
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
                tensor = next(item["name"] for item in listed["tensors"] if item["name"].endswith("mlp.down_proj.weight"))
                patch = await call("create_patch", {"parent_variant_id": base, "method": "two_axis_example",
                                                    "edits": [{"tensor": tensor, "op": "add",
                                                               "region": [[0, 2], [0, 2]], "values": [[0.5, 0], [0, 0.125]]}]})
                variant = (await call("apply_patch", {"parent_variant_id": base, "patch_id": patch["patch_id"]}))["variant_id"]
                observed = await call("model_forward", {"variant_id": base, "mode": "teacher_forced",
                                                        "texts": ["The capital of France is"],
                                                        "observe": ["block.0.mlp.down_proj.input"]})
                inputs = {tensor: observed["artifacts"]["mlp_0"]["artifact_id"]}
                delta = await call("analyze_weight_delta", {"base_variant_id": base, "variant_id": variant,
                                                            "other_variant_id": variant, "direction_rank": 2,
                                                            "input_artifacts": inputs})
                first = delta["top_tensors"][0]
                assert first["changed"] == 2 and first["output_change_rms"] > 0
                assert abs(delta["signed_delta_cosine"] - 1) < 1e-8
                request = {"parent_variant_id": base, "patch_id": patch["patch_id"], "rank": 1}
                compressed = await call("compress_update", {**request, "materialize": True})
                assert compressed["patch_id"] and compressed["stored_parameters"] == sum(first["shape"])
                # 同一个更新文件可直接按 ID 复用，分析写回原精度的误差。
                adapter = await call("analyze_weight_delta", {"base_variant_id": base,
                                                              "adapter_id": compressed["adapter_id"]})
                assert adapter["top_tensors"][0]["theoretical_update_norm"] > 0
                weighted = await call("compress_update", {**request, "method": "activation_weighted_svd",
                                                            "input_artifacts": inputs})
                assert weighted["stored_parameters"] == compressed["stored_parameters"]
                print(json.dumps({"tensor": tensor, "delta_norm": delta["delta_norm"],
                                  "singular_values": first["singular_values"],
                                  "stored_parameters": compressed["stored_parameters"],
                                  "activation_weighted_rank": 1}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.model))
