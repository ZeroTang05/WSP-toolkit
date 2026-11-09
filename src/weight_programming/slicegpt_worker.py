"""通过 Microsoft SliceGPT 执行旋转与切片，保留可重载的跳连矩阵。"""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


def load_sliced_checkpoint(checkpoint):
    """用上游切片配置重建层形状，再严格加载 SafeTensors。"""
    import torch
    from safetensors.torch import load_file
    from slicegpt.adapters.llama_adapter import LlamaModelAdapter
    from slicegpt.layernorm_fusion import fuse_modules, replace_layers
    from slicegpt.model_adapter import SlicingConfig
    from slicegpt.rotate import slice_rotated_model
    from transformers import LlamaConfig, LlamaForCausalLM

    checkpoint = Path(checkpoint)
    config = LlamaConfig.from_pretrained(checkpoint, local_files_only=True)
    dtype = config.torch_dtype
    model = LlamaForCausalLM(config).to(dtype=dtype)
    adapter = LlamaModelAdapter(model)
    with torch.no_grad():
        replace_layers(adapter, verbose=False)
        fuse_modules(adapter)
        for wrapper in adapter.get_layers():
            for name in ("attn_shortcut_Q", "mlp_shortcut_Q"):
                setattr(wrapper.layer, name, torch.nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size, dtype=dtype)))
        adapter.slicing_conf = SlicingConfig.from_json_string((checkpoint / "slicing_config.json").read_text(encoding="utf-8"))
        slice_rotated_model(adapter)
        model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True, assign=True)
    return model.eval().requires_grad_(False)


def main(request_path):
    """在独立环境调用作者的归一化合并、PCA 旋转和切片。"""
    import torch
    from safetensors.torch import load_file, save_file
    from slicegpt import layernorm_fusion, rotate
    from slicegpt.adapters.llama_adapter import LlamaModelAdapter
    from slicegpt.config import config
    from slicegpt.slicing_scheduler import ConstSlicingScheduler
    from transformers import LlamaForCausalLM

    directory = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if version("transformers") != "4.41.0":
        raise ValueError("SliceGPT worker requires the upstream Transformers 4.41.0 environment")
    model = LlamaForCausalLM.from_pretrained(request["checkpoint"], local_files_only=True, torch_dtype="auto").eval()
    original = sum(parameter.numel() for parameter in model.parameters())
    overlay = directory / "parent.safetensors"
    adapter = LlamaModelAdapter(model)
    width = request["target_hidden_size"]
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ids = request["calibration_ids"]
    batches = torch.utils.data.DataLoader([{"input_ids": torch.tensor(row),
                                           "attention_mask": torch.ones(len(row), dtype=torch.long)} for row in ids], batch_size=1)
    with torch.no_grad():
        if overlay.is_file():
            for name, value in load_file(overlay).items():
                model.get_parameter(name).copy_(value)
        model.config.use_cache = False
        layernorm_fusion.replace_layers(adapter, verbose=False)
        layernorm_fusion.fuse_modules(adapter)
        rotate.rotate_and_slice(adapter, batches, ConstSlicingScheduler(width))
    model.config.tie_word_embeddings = False
    output = directory / "checkpoint"
    output.mkdir()
    model.config.save_pretrained(output)
    model.generation_config.save_pretrained(output)
    (output / "slicing_config.json").write_text(adapter.slicing_conf.to_json_string(), encoding="utf-8")
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, output / "model.safetensors")
    report = {"original_parameters": original, "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "layer_count": model.config.num_hidden_layers, "original_hidden_size": model.config.hidden_size,
              "runtime": {package: version(package) for package in ("transformercompression", "torch", "transformers")},
              "rotation_source_sha256": hashlib.sha256(Path(rotate.__file__).read_bytes()).hexdigest()}
    (directory / "result.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
