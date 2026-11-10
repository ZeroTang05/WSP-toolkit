"""Demonstrate patch creation, running, undoing and replaying across workspaces with real checkpoints."""

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from weight_programming import open_inspection


def observation(variant_id):
    """Use the same sentence and layer for all three forward passes."""
    return {"variant_id": variant_id, "mode": "teacher_forced", "texts": ["The capital of France is"],
            "observe": ["block.0.residual_out"]}


async def main(model):
    with TemporaryDirectory(prefix="wsp-patch-") as temporary:
        root = Path(temporary)
        workspace = root / "mcp"
        server = StdioServerParameters(command=sys.executable,
                                       args=["-m", "weight_programming.mcp_server", "--model", model,
                                             "--workspace", str(workspace), "--allow-edits"])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    result = await session.call_tool(name, arguments)
                    if result.is_error:
                        raise RuntimeError(result.structured_content or result.content)
                    return result.structured_content

                async def activate_first_channel(variant_id):
                    run = await call("model_forward", observation(variant_id))
                    result = await call("read_artifact", {"artifact_id": run["artifacts"]["hidden_1"]["artifact_id"],
                                                         "region": [[0, 1], [0, 1], [0, 1]]})
                    return result["values"][0][0][0]

                base = (await call("describe_model", {}))["variant_id"]
                tensors = await call("list_tensors", {"variant_id": base, "layer": 0, "limit": 100})
                name = next(item["name"] for item in tensors["tensors"] if item["name"].endswith("mlp.down_proj.weight"))
                region = [[0, 1], [0, 1]]
                read_request = {"name": name, "region": region}
                before_weight = await call("read_tensor", {"variant_id": base, **read_request})
                before = await activate_first_channel(base)
                patch = await call("create_patch", {"parent_variant_id": base, "method": "example_slice_add",
                                                    "edits": [{"tensor": name, "op": "add", "region": region,
                                                               "values": [[0.5]]}]})
                assert patch["tensors"][name]["changed"] == 1
                patch_id = patch["patch_id"]
                candidate = await call("apply_patch", {"parent_variant_id": base, "patch_id": patch_id})
                variant = candidate["variant_id"]
                after_weight = await call("read_tensor", {"variant_id": variant, **read_request})
                assert after_weight["dtype"] == before_weight["dtype"] and after_weight["values"] != before_weight["values"]
                after = await activate_first_channel(variant)
                assert after != before
                reverted = await call("revert_patch", {"variant_id": variant})
                assert reverted["variant_id"] == base and await activate_first_channel(base) == before
                bundle = await call("export_patch", {"patch_id": patch_id})
                exported = await call("export_model", {"variant_id": variant})
                projected = await call("project_weight", {"parent_variant_id": base, "tensor": name,
                                                           "direction": [1] + [0] * (before_weight["shape"][0] - 1),
                                                           "columns": [0]})
                projected_id = (await call("apply_patch", {"parent_variant_id": base,
                                                           "patch_id": projected["patch_id"]}))["variant_id"]
                assert (await call("read_tensor", {"variant_id": projected_id, **read_request}))["values"] == [[0]]
                merge_request = {"parent_variant_id": base, "patch_ids": [patch_id, projected["patch_id"]]}
                average = await call("merge_updates", {**merge_request, "method": "task_arithmetic"})
                ties = await call("merge_updates", {**merge_request, "method": "ties", "density": 1})
                assert average["realized_norm"] > 0 and ties["realized_norm"] > 0
                dare_request = {**merge_request, "method": "dare", "drop_probability": 0.5, "seed": 17}
                dare = await call("merge_updates", dare_request)
                replay_dare = await call("merge_updates", dare_request)
                assert dare["patch_id"] == replay_dare["patch_id"] and dare["realized_norm"] > 0
                assert dare["merge_evidence"]["dare"]["rescale"] == 2
                dare_ties = await call("merge_updates", {**dare_request, "drop_probability": 0, "dare_merge": "ties"})
                assert dare_ties["realized_norm"] == ties["realized_norm"]
                calibration = await call("model_forward", {**observation(base),
                                                            "observe": ["block.0.mlp.down_proj.input"]})
                prune_request = {"parent_variant_id": base, "tensor": name,
                                 "input_artifact_id": calibration["artifacts"]["mlp_0"]["artifact_id"]}
                wanda = await call("prune_and_compensate", {**prune_request, "method": "wanda", "sparsity": 0.1})
                fixed = await call("prune_and_compensate", {**prune_request, "method": "sparsegpt_fixed_mask",
                                                          "mask_artifact_id": wanda["mask_artifact_id"]})
                adaptive = await call("prune_and_compensate", {**prune_request, "method": "sparsegpt_adaptive",
                                                             "sparsity": 0.1})
                assert all(result["realized_zero_fraction"] >= 0.09 for result in (wanda, fixed, adaptive))

        # Import the patch in the new working directory and retrieve the candidate with the same ID.
        replay = root / "replay"
        shutil.copytree(workspace / bundle["bundle_path"], replay / "imports" / patch_id)
        python = open_inspection(model, replay, allow_edits=True).registry
        imported = python.call("import_patch", {"bundle_id": patch_id})
        replayed = python.call("apply_patch", {"parent_variant_id": base, "patch_id": imported["patch_id"]})
        assert replayed["variant_id"] == variant
        assert python.call("read_tensor", {"variant_id": variant, **read_request})["values"] == after_weight["values"]

        # The full checkpoint is reloaded from another directory and checked against the forward results of the original candidate.
        reloaded = open_inspection(workspace / exported["checkpoint_path"], root / "reload").registry
        reloaded_id = reloaded.call("describe_model", {})["variant_id"]
        restored_weight = reloaded.call("read_tensor", {"variant_id": reloaded_id, **read_request})
        assert restored_weight["values"] == after_weight["values"] and restored_weight["dtype"] == after_weight["dtype"]
        run = reloaded.call("model_forward", observation(reloaded_id))
        restored = reloaded.call("read_artifact", {"artifact_id": run["artifacts"]["hidden_1"]["artifact_id"],
                                                 "region": [[0, 1], [0, 1], [0, 1]]})
        assert restored["values"][0][0][0] == after
        print(json.dumps({"tensor": name, "dtype": after_weight["dtype"], "changed_weights": 1,
                          "before_activation": before, "candidate_activation": after,
                          "reverted_and_replayed": True, "checkpoint_reload_matches": True,
                          "projected_weight": 0, "task_arithmetic_norm": average["realized_norm"],
                          "ties_norm": ties["realized_norm"], "dare_norm": dare["realized_norm"],
                          "dare_kept": dare["merge_evidence"]["dare"]["sources"][0]["realized_keep_fraction"],
                          "wanda_zero_fraction": wanda["realized_zero_fraction"],
                          "fixed_mask_output_mse": fixed["output_mean_squared_error"],
                          "adaptive_zero_fraction": adaptive["realized_zero_fraction"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.model))
