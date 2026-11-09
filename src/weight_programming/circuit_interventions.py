"""在残差流的写入端与读取端之间执行边干预和归因。"""

from collections import defaultdict
from contextlib import ExitStack

import torch

from .patching import token_metrics


class ResidualCircuit:
    """将 Llama 和 Qwen3 的残差流拆成嵌入、整块注意力和前馈写入。"""

    def __init__(self, backend):
        self.backend = backend
        self.writers, self.readers = {}, {}
        self.order = {"embedding.out": 0}
        for layer, block in enumerate(backend.layers):
            for name, reader, writer in (("attention", block.input_layernorm, block.self_attn),
                                         ("mlp", block.post_attention_layernorm, block.mlp)):
                node = f"block.{layer}.{name}"
                self.order[node] = len(self.order)
                self.writers[node], self.readers[node] = writer, reader
        self.order["output"] = len(self.order)
        self.readers["output"] = backend.decoder.norm

    def validate(self, edges):
        """边必须从已有写入端连接到其后的读取端。"""
        for source, target in edges:
            if source not in self.order or target not in self.readers or self.order[source] >= self.order[target]:
                raise ValueError(f"Invalid residual edge {source} -> {target}")
        if len(set(edges)) != len(edges):
            raise ValueError("Residual edges must be distinct")

    def forward(self, ids, query, usage, *, sources=(), targets=(), patches=None, gradients=False):
        """按需收集分支输出和归一化前输入，运行一次真实模型。"""
        model, device = self.backend.model, self.backend.device
        patches = patches or {}
        writes, reads = {}, {}

        def capture_writer(node):
            def hook(module, args, output):
                writes[node] = output[0] if isinstance(output, (tuple, list)) else output
            return hook

        def capture_reader(node):
            def hook(module, args):
                original = args[0]
                if node in targets:
                    reads[node] = original
                if node in patches:
                    return (patches[node](original, writes), *args[1:])
            return hook

        with torch.set_grad_enabled(gradients), ExitStack() as hooks:
            tokens = torch.tensor([ids], device=device)
            embeddings = model.get_input_embeddings()(tokens).detach().requires_grad_(gradients)
            if "embedding.out" in sources:
                writes["embedding.out"] = embeddings
            for node in sources:
                if node != "embedding.out":
                    handle = self.writers[node].register_forward_hook(capture_writer(node))
                    hooks.callback(handle.remove)
            for node in set(targets) | patches.keys():
                handle = self.readers[node].register_forward_pre_hook(capture_reader(node))
                hooks.callback(handle.remove)
            usage["executed_model_forwards"] += 1
            usage["input_tokens"] += len(ids)
            output = model(inputs_embeds=embeddings, attention_mask=torch.ones_like(tokens),
                           use_cache=False, logits_to_keep=torch.tensor([query], device=device))
            logits = output.logits[0, 0].float()
        return logits, writes, reads

    def isolated_path(self, clean_ids, query, usage, path, clean_writes, clean_reads, corrupt_writes, positions, strength):
        """只让受扰写入沿指定路径传递，各目标的其余输入固定为干净基线。"""
        def route(source, target):
            def patch(original, writes):
                root = source == path[0]
                selected = corrupt_writes[source] if root else writes[source]
                delta = selected.float() - clean_writes[source].float()
                return _add_delta(clean_reads[target], strength * delta if root else delta, positions)
            return patch

        patches = {target: route(source, target) for source, target in zip(path, path[1:])}
        return self.forward(clean_ids, query, usage, sources=path[1:-1], patches=patches)[0]

    def replace_edges(self, clean_ids, query, usage, edges, corrupt_writes, positions):
        """把选定边替换为受扰写入，每个发送端在其他连接上照常传递。"""
        if not edges:
            return self.forward(clean_ids, query, usage)[0]
        incoming = defaultdict(list)
        for source, target in edges:
            incoming[target].append(source)

        def replace(sources):
            def patch(original, writes):
                delta = sum(corrupt_writes[source].float() - writes[source].float() for source in sources)
                return _add_delta(original, delta, positions)
            return patch

        return self.forward(clean_ids, query, usage, sources=dict.fromkeys(source for source, _ in edges),
                            patches={node: replace(sources) for node, sources in incoming.items()})[0]

    def edge_estimates(self, clean_ids, corrupt_ids, query, usage, edges, pair, metric):
        """一次干净反向取得各读取端梯度，再与每条边的写入差值做点积。"""
        sources = list(dict.fromkeys(source for source, _ in edges))
        targets = list(dict.fromkeys(target for _, target in edges))
        _, corrupt, _ = self.forward(corrupt_ids, query, usage, sources=sources)
        with torch.enable_grad():
            logits, clean, reads = self.forward(clean_ids, query, usage, sources=sources,
                                                targets=targets, gradients=True)
            value = score(logits, pair, metric)
            usage["executed_model_backwards"] += 1
            gradients = dict(zip(targets, torch.autograd.grad(value, [reads[target] for target in targets])))
        positions = pair.get("positions", slice(None))
        with torch.no_grad():
            return [float(((corrupt[source][0, positions].float() - clean[source][0, positions].float())
                           * gradients[target][0, positions].float()).sum()) for source, target in edges]


def _add_delta(baseline, delta, positions):
    """只在指定词元位置替换边，保持模型执行精度。"""
    if positions is None:
        return (baseline.float() + delta).to(baseline.dtype)
    value = baseline.clone()
    value[:, positions] = (value[:, positions].float() + delta[:, positions]).to(value.dtype)
    return value


def score(logits, pair, metric):
    """与站点干预工具共用同一套下一词元分数。"""
    return token_metrics(logits, pair["target_token"], pair["alternative_token"])[metric]
