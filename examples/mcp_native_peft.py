"""用真实的一步 rsLoRA 更新验证原生 PEFT 导入、分析和压缩。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


async def main(model):
    """用 PEFT 原生保存格式接入现有 MCP 工具，再核对合并与前向。"""
    torch.manual_seed(11)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    ids = tokenizer("The capital of France is Paris.", return_tensors="pt")
    target = "model.layers.0.self_attn.q_proj"
    second_target = "model.layers.0.self_attn.v_proj"
    base = AutoModelForCausalLM.from_pretrained(model, local_files_only=True, dtype="auto",
                                               attn_implementation="eager")
    source_dtype = base.get_submodule(target).weight.dtype
    base = base.to(torch.float32)
    adapter = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=6,
        use_rslora=True, target_modules=[target, second_target],
        rank_pattern={second_target: 3}, alpha_pattern={second_target: 9}))
    optimizer = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=0.001)
    loss = adapter(**ids, labels=ids["input_ids"]).loss
    loss.backward()
    optimizer.step()
    adapter.eval()
    with TemporaryDirectory(prefix="wsp-native-peft-") as directory:
        root = Path(directory)
        source = root / "native"
        adapter.save_pretrained(source, safe_serialization=True, save_embedding_layers=False)
        # 以实际 PEFT 重载和合并为参照，工具导入只消费磁盘上的原生文件。
        reloaded = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(model, local_files_only=True, dtype=torch.float32,
                                                attn_implementation="eager"), source).eval()
        with torch.no_grad():
            native_logits = reloaded(**ids).logits
            original = reloaded.get_base_model().get_submodule(target).base_layer.weight.detach().clone()
            merged = reloaded.merge_and_unload(safe_merge=True)
            torch.testing.assert_close(merged(**ids).logits, native_logits, rtol=1e-4, atol=1e-4)
            # 检查点按原精度保存，比较参照也写回同一精度。
            for module in (target, second_target):
                weight = merged.get_submodule(module).weight
                weight.copy_(weight.to(source_dtype).float())
            merged_weight = merged.get_submodule(target).weight.detach().clone()
            native_logits = merged(**ids).logits
        workspace = root / "workspace"
        server = StdioServerParameters(command=sys.executable, args=["-m", "weight_programming.mcp_server",
            "--model", str(model), "--workspace", str(workspace), "--peft-adapter", str(source), "--allow-edits"])
        async with stdio_client(server) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()

                async def call(name, arguments):
                    response = await session.call_tool(name, arguments)
                    if response.is_error:
                        raise RuntimeError(response.structured_content or response.content)
                    return response.structured_content

                info = await call("describe_model", {})
                registered, = info["registered_adapters"]
                assert registered["targets"][second_target + ".weight"]["rank"] == 3
                imported = registered["adapter_id"]
                name = target + ".weight"
                analysis = await call("analyze_weight_delta", {"base_variant_id": info["variant_id"],
                    "adapter_id": imported, "tensors": [name], "direction_rank": 1})
                assert analysis["changed_tensors"] == 1
                torch.testing.assert_close(torch.tensor(analysis["delta_norm"]), (merged_weight - original).norm(),
                                           rtol=1e-4, atol=1e-7)
                compressed = await call("compress_update", {"parent_variant_id": info["variant_id"],
                    "adapter_id": imported, "rank": {name: 2, second_target + ".weight": 3}, "materialize": True})
                approximate = await call("analyze_weight_delta", {"base_variant_id": info["variant_id"],
                    "adapter_id": imported, "tensors": [name], "direction_rank": 1, "direction_solver": "randomized"})
                torch.testing.assert_close(torch.tensor(analysis["top_tensors"][0]["singular_values"]),
                    torch.tensor(approximate["top_tensors"][0]["singular_values"]), rtol=1e-4, atol=1e-7)
                combination = await call("merge_updates", {"parent_variant_id": info["variant_id"],
                    "method": "task_arithmetic", "adapter_ids": [imported, compressed["adapter_id"]],
                    "coefficients": [0.5, 0.5]})
                candidate = await call("apply_patch", {"parent_variant_id": info["variant_id"],
                                                       "patch_id": combination["patch_id"]})
                exported = await call("export_model", {"variant_id": candidate["variant_id"]})
        checkpoint = AutoModelForCausalLM.from_pretrained(workspace / exported["checkpoint_path"],
                            local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
        with torch.no_grad():
            torch.testing.assert_close(checkpoint(**ids).logits, native_logits, rtol=1e-4, atol=1e-4)
        print(json.dumps({"method": "native_rsLoRA_import", "loss": float(loss.detach()),
            "changed_tensors": analysis["changed_tensors"], "delta_norm": analysis["delta_norm"],
            "targets": registered["targets"], "merged_forward_verified": True,
            "randomized_direction_verified": True, "adapter_merge_verified": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    asyncio.run(main(parser.parse_args().model))
