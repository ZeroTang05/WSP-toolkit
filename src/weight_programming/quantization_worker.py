"""Generate packed 4-bit checkpoints in a standalone LLM Compressor environment."""

from importlib.metadata import version
import json
from pathlib import Path
import sys


def main(request_path):
    """Replay the parent version, calibrate and save checkpoints with official GPTQ or AWQ."""
    import torch
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    directory = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if version("llmcompressor") != "0.15.0":
        raise ValueError("This worker requires llmcompressor==0.15.0")
    model = AutoModelForCausalLM.from_pretrained(request["checkpoint"], local_files_only=True,
                                                 dtype="auto", attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(request["checkpoint"], local_files_only=True)
    overlay = directory / "parent.safetensors"
    if overlay.is_file():
        with torch.no_grad():
            for name, value in load_file(overlay).items():
                model.get_parameter(name).copy_(value)
    modules = [module for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name != "lm_head"]
    group_size = request["group_size"]
    if not modules or any(module.in_features % group_size for module in modules):
        raise ValueError(f"Group size {group_size} must divide every quantized input width")
    method = request["method"]
    symmetric = method == "gptq"
    weights = QuantizationArgs(num_bits=4, type="int", strategy="group", group_size=group_size, symmetric=symmetric)
    scheme = QuantizationScheme(targets=["Linear"], weights=weights)
    settings = {"config_groups": {"weights": scheme}, "ignore": ["lm_head"]}
    recipe = GPTQModifier(**settings) if method == "gptq" else [AWQModifier(), QuantizationModifier(**settings)]
    ids = request["calibration_ids"]
    dataset = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]})
    parameters = sum(parameter.numel() for parameter in model.parameters())
    oneshot(model=model, tokenizer=tokenizer, dataset=dataset, recipe=recipe, pad_to_max_length=False,
            num_calibration_samples=len(ids), max_seq_length=max(map(len, ids)))
    output = directory / "checkpoint"
    model.save_pretrained(output, save_compressed=True, safe_serialization=True)
    tokenizer.save_pretrained(output)
    report = {"parameters": parameters, "layer_count": model.config.num_hidden_layers,
              "expected_quantized_linears": len(modules), "symmetric": symmetric,
              "runtime": {package: version(package) for package in ("llmcompressor", "compressed-tensors", "torch", "transformers")}}
    (directory / "result.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
