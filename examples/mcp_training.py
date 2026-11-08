"""用一个真实检查点和少量 GSM8K 样本验证结构压缩与训练工具。"""

import argparse
import asyncio
from itertools import islice
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM

from delta_programming import load_effective
from weight_programming import open_inspection
from weight_programming.forward_service import PracticeSet


def training_rows(model, dataset, workspace):
    """为真实数学题保存标准答案与模型实际生成的回答。"""
    with dataset.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in islice(stream, 2)]
    teacher = open_inspection(model, workspace)
    call = teacher.registry.call
    row = rows[0]
    run = call("model_forward", {"variant_id": teacher.variant_id, "mode": "generate",
                                 "prompts": [row["problem"]], "max_new_tokens": 12})
    records = call("read_artifact", {"artifact_id": run["artifacts"]["original"]["artifact_id"]})
    reply = records["records"][0]["text"].strip()
    assert reply and reply != row["solution"]
    row.update(chosen=row["solution"], rejected=reply, teacher_answer=reply, teacher_model=teacher.variant_id)
    return rows


async def edit_and_train(model, root, rows):
    """通过真实 MCP 进程执行两种压缩、LoRA 与偏好训练。"""
    dataset = root / "training.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    args = ["-m", "weight_programming.mcp_server", "--model", str(model), "--workspace", str(root / "mcp"),
            "--allow-edits", "--allow-gradients", "--allow-training", "--training-jsonl", str(dataset)]
    server = StdioServerParameters(command=sys.executable, args=args)
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            async def call(name, arguments):
                result = await session.call_tool(name, arguments)
                if result.is_error:
                    raise RuntimeError(result.structured_content or result.content)
                return result.structured_content

            model_info = await call("describe_model", {})
            variant = model_info["variant_id"]
            tensors = await call("list_tensors", {"variant_id": variant, "layer": 0})
            name = next(item["name"] for item in tensors["tensors"] if item["name"].endswith(".q_proj.weight"))
            dropped = await call("compress_model", {"variant_id": variant, "method": "block_influence",
                                                    "calibration_texts": [row["problem"] for row in rows], "drop_layers": 1})
            taylor = await call("compress_model", {"variant_id": variant, "method": "taylor_mlp", "fraction": 0.125,
                                                   "calibration_case_ids": [rows[0]["id"]]})
            assert dropped["layer_count"] == model_info["layer_count"] - 1
            assert all(result["parameters"] < result["original_parameters"] for result in (dropped, taylor))
            train = {"parent_variant_id": variant, "dataset_id": model_info["training_dataset_id"],
                     "case_ids": [rows[0]["id"]], "targets": [name.removesuffix(".weight")], "rank": 2, "steps": 1}
            lora = await call("train_update", {**train, "method": "lora", "materialize": True})
            candidate = await call("apply_patch", {"parent_variant_id": variant, "patch_id": lora["patch_id"]})
            parent = candidate["variant_id"]
            dpo = await call("train_update", {**train, "method": "dpo", "parent_variant_id": parent})
            delta = await call("analyze_weight_delta", {"base_variant_id": parent, "adapter_id": dpo["adapter_id"],
                                                        "tensors": [name]})
            assert delta["changed_tensors"] == 1
            return dropped, taylor, lora, dpo, train


async def main(model, dataset):
    """重新加载导出检查点和适配器，并执行蒸馏与剪后恢复。"""
    with TemporaryDirectory(prefix="wsp-training-") as directory:
        root = Path(directory)
        rows = training_rows(model, dataset, root / "teacher")
        dropped, taylor, lora, dpo, train = await edit_and_train(model, root, rows)
        workspace = root / "mcp"
        base = AutoModelForCausalLM.from_pretrained(model, local_files_only=True, dtype=torch.float32)
        adapter = PeftModel.from_pretrained(base, workspace / lora["adapter_path"])
        _, factors = load_effective((workspace / lora["adapter_path"]).parent)
        target = train["targets"][0]
        a, b = factors[target]
        actual = adapter.get_base_model().get_submodule(target).get_delta_weight("default")
        torch.testing.assert_close(actual, (b @ a).float())
        del adapter, base
        depth = AutoModelForCausalLM.from_pretrained(workspace / dropped["checkpoint_path"], local_files_only=True)
        assert len(depth.model.layers) == dropped["layer_count"]
        del depth

        registered = PracticeSet(rows, role="train", name=train["dataset_id"])
        student = open_inspection(workspace / taylor["checkpoint_path"], root / "student",
                                  training_data=registered, allow_training=True)
        assert student.forward.backend.mlp_width == taylor["details"]["intermediate_size"]
        student_args = {**train, "parent_variant_id": student.variant_id}
        recovery = student.registry.call("train_update", {**student_args, "method": "post_pruning_recovery"})
        distilled = student.registry.call("train_update", {**student_args, "method": "distillation"})
        print(json.dumps({"depth_layers": dropped["layer_count"], "original_parameters": taylor["original_parameters"],
                          "depth_parameters": dropped["parameters"], "taylor_parameters": taylor["parameters"],
                          "taylor_mlp_width": taylor["details"]["intermediate_size"],
                          "losses": {run["method"]: run["losses"][0] for run in (lora, dpo, recovery, distilled)}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(main(args.model, args.dataset))
