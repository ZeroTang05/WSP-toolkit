"""Perform edge intervention and attribution between the writing and reading sides of the residual stream."""

from collections import defaultdict
from contextlib import ExitStack

import torch

from .patching import token_metrics


class ResidualCircuit:
    """Split the Llama and Qwen3 residual stream into embedding, whole-attention, and feedforward writes."""

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
        """An edge must connect from an existing write end to a subsequent read end."""
        for source, target in edges:
            if source not in self.order or target not in self.readers or self.order[source] >= self.order[target]:
                raise ValueError(f"Invalid residual edge {source} -> {target}")
        if len(set(edges)) != len(edges):
            raise ValueError("Residual edges must be distinct")

    def forward(self, ids, query, usage, *, sources=(), targets=(), patches=None, gradients=False):
        """Collect branch outputs and pre-normalized inputs as needed and run the real model once."""
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
        """Only the disturbed writes are passed along the specified path, and the remaining input to each target is fixed to the clean baseline."""
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
        """Replace selected edges with perturbed writes while each sender passes normally along other connections."""
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
        """Use one clean backward pass to obtain gradients at each read end, then dot them with the write difference for each edge."""
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
    """Replace edges only at specified token positions while preserving model execution precision."""
    if positions is None:
        return (baseline.float() + delta).to(baseline.dtype)
    value = baseline.clone()
    value[:, positions] = (value[:, positions].float() + delta[:, positions]).to(value.dtype)
    return value


def score(logits, pair, metric):
    """Share the same next-token scores with the site intervention tool."""
    return token_metrics(logits, pair["target_token"], pair["alternative_token"])[metric]
