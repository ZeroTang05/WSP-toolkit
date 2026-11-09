"""用 Transformers 的 DistilBERT 分类器复用权重版本、观测和产物接口。"""

from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
import threading

import numpy as np
import torch

from .hf_backend import HuggingFaceBackend


class DistilBertBackend(HuggingFaceBackend):
    """分类输出按标签排列，内部快照沿用样本、词元、通道坐标。"""

    modes = ("predict",)

    def __init__(self, model, tokenizer, store):
        if version("transformers") != "4.57.6" or model.config.model_type != "distilbert":
            raise ValueError("DistilBERT classification requires Transformers 4.57.6")
        if model.config.problem_type not in (None, "single_label_classification") or model.config.num_labels < 2:
            raise ValueError("Use a single-label DistilBERT sequence classifier with at least two labels")
        self.model = model.eval().requires_grad_(False)
        self.tokenizer, self.store = tokenizer, store
        self._execution_lock = threading.RLock()
        self.decoder = model.distilbert
        self.layers = self.decoder.transformer.layer
        config = model.config
        self.num_layers, self.hidden_size = len(self.layers), config.dim
        self.mlp_width, self.vocab_size = config.hidden_dim, config.vocab_size
        self.num_heads, self.context_length = config.n_heads, config.max_position_embeddings
        self.full_attention_layers = tuple(range(self.num_layers))
        self.device = next(model.parameters()).device
        if self.device.type not in ("cpu", "cuda") or any(p.device != self.device for p in model.parameters()):
            raise ValueError("Use one CPU or CUDA device")
        self.parameters = dict(model.named_parameters(remove_duplicate=False))
        if Path(config._name_or_path).resolve() != store.model_dir:
            raise ValueError("Loaded classifier path differs from registered checkpoint")
        state = model.state_dict()
        for name in store.names:
            if name not in state or not torch.equal(store._native(name).to(state[name].dtype), state[name].cpu()):
                raise ValueError(f"Loaded classifier differs from checkpoint: {name}")
        self.identity = {"backend": "hf-distilbert-classification-v1", "transformers": version("transformers"),
            "torch": torch.__version__, "device": str(self.device), "dtype": str(next(model.parameters()).dtype),
            "model_type": config.model_type, "source_verified_tensorwise": True,
            "attention_implementation": config._attn_implementation, "attention_direction": "bidirectional"}

    def encode(self, text):
        """沿用分类器原始分词规则，保留 CLS 与 SEP 特殊词元。"""
        return self.tokenizer.encode(text, add_special_tokens=True)

    def observation_sites(self):
        """提供与生成模型相同的逻辑坐标，登记真实编码器位置。"""
        sites = super().observation_sites()
        for site in sites.values():
            site["modes"] = ["predict"]
        return sites

    def linear_input_module(self, layer):
        """编码器前馈模块的第二个线性层接收中间激活。"""
        return self.layers[layer].ffn.lin2

    def projection_output_sites(self, tensor):
        """前馈输出允许使用同一模块的观测，层归一化后的残差另有坐标。"""
        for layer, block in enumerate(self.layers):
            if self.parameters.get(tensor) is block.ffn.lin2.weight:
                return [f"block.{layer}.mlp.out"]
        return []

    @contextmanager
    def _observe(self, hidden, mlp, attention, mlp_out=()):
        """只挂接请求的位置，退出前移除全部观测函数。"""
        captured, handles = {}, []

        def post(key, index=None):
            def capture(_module, _args, output):
                value = output[index] if index is not None else output
                captured[key] = value.detach().float().cpu().clone()
            return capture

        try:
            for layer in hidden:
                module = self.decoder.embeddings if layer == 0 else self.layers[layer - 1]
                # 开启注意力时，TransformerBlock 的首项为注意力，末项为残差。
                handles.append(module.register_forward_hook(post(f"hidden_{layer}", None if layer == 0 else -1)))
            for layer in mlp:
                def pre(_module, args, key=f"mlp_{layer}"):
                    captured[key] = args[0].detach().float().cpu().clone()
                handles.append(self.linear_input_module(layer).register_forward_pre_hook(pre))
            for layer in mlp_out:
                handles.append(self.layers[layer].ffn.register_forward_hook(post(f"mlp_out_{layer}")))
            for layer in attention:
                handles.append(self.layers[layer].attention.register_forward_hook(post(f"attention_{layer}", 1)))
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def prepare_predict(self, request, policy):
        """共用词元与观测预算，增加标签分数的数组说明。"""
        prepared = self.prepare_teacher(request, policy)
        count, labels = len(prepared["sequences"]), self.model.config.num_labels
        prepared["shapes"].update(class_logits=((count, labels), "float32"),
                                  class_probabilities=((count, labels), "float32"),
                                  predicted_label_ids=((count,), "int64"))
        prepared["estimated_payload_bytes"] += count * (labels * 8 + 8)
        if prepared["estimated_payload_bytes"] + 1_048_576 > policy.max_return_bytes:
            raise ValueError("Classification artifact exceeds the return budget")
        return prepared

    @torch.inference_mode()
    def predict(self, request, prepared, output):
        """同一次前向产生分类分数和内部快照，保留有效词元掩码。"""
        arrays = {}
        for name, (shape, dtype) in prepared["shapes"].items():
            array = np.lib.format.open_memmap(Path(output) / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            array[:] = -1 if name == "input_ids" else 0
            arrays[name] = array
        observe = request["return"]
        for row, ids in enumerate(prepared["sequences"]):
            length = len(ids)
            arrays["input_ids"][row, :length], arrays["lengths"][row] = ids, length
            arrays["valid_tokens"][row, :length] = True
            tokens = torch.tensor([ids], device=self.device)
            with self._observe(observe["hidden_states"], observe["mlp_act"], observe["attn_layers"],
                               observe.get("mlp_out", ())) as captured:
                result = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                    output_attentions=bool(observe["attn_layers"]))
            logits = result.logits[0].float()
            arrays["class_logits"][row] = logits.cpu().numpy()
            arrays["class_probabilities"][row] = logits.softmax(-1).cpu().numpy()
            arrays["predicted_label_ids"][row] = int(logits.argmax())
            for key, value in captured.items():
                if key.startswith("attention_"):
                    arrays[key][row, :, :length, :length] = value[0].numpy()
                else:
                    arrays[key][row, :length] = value[0].numpy()
        files = {}
        for name, array in arrays.items():
            array.flush()
            files[name] = {"file": f"{name}.npy", "shape": list(array.shape), "dtype": str(array.dtype)}
        return {"files": files, "input_tokens": sum(prepared["lengths"]),
                "executed_model_forwards": len(prepared["sequences"]),
                "label_names": [self.model.config.id2label[i] for i in range(self.model.config.num_labels)],
                "alignment": {"hidden": "Embedding output then raw encoder block output; includes special tokens",
                    "mlp": "Input of the second FFN linear layer", "mlp_out": "FFN output before residual addition",
                    "attention": "Bidirectional self-attention; padded query/key positions zero",
                    "class_logits": "Raw sequence scores in label_names order",
                    "class_probabilities": "Softmax over single-label classification scores",
                    "padding": "Right padded; consult lengths and valid_tokens"}}
