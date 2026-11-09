"""用 AtP* 的 GradDrop 预估恢复激活对输出的影响。"""

from contextlib import ExitStack

import numpy as np
import torch

from .patching import token_metrics


def _tensor(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def atp_star_scores(backend, snapshot, specs, prepared, pair, metric, usage):
    """复用受扰前向图，逐层屏蔽残差分支的反向梯度。"""
    clean_ids, corrupt_ids, query = prepared
    hidden = sorted({site["layer"] for site in specs if site["kind"] == "residual"})
    mlp = sorted({site["layer"] for site in specs if site["kind"] == "mlp_input"})
    keys = [("hidden" if site["kind"] == "residual" else "mlp") + f"_{site['layer']}" for site in specs]
    positions = pair["positions"]
    captures = {}
    dropped_layer = None

    def remember(key, *, pre=False):
        def hook(module, arguments, output=None):
            captures[key] = arguments[0] if pre else _tensor(output)
        return hook

    def drop_branch(layer):
        def forward_hook(module, arguments, output):
            def gradient_hook(gradient):
                return torch.zeros_like(gradient) if dropped_layer == layer else None
            _tensor(output).register_hook(gradient_hook)
        return forward_hook

    with backend.activate(snapshot), torch.enable_grad(), ExitStack() as hooks:
        usage["executed_model_forwards"] += 1
        _, clean = backend._read(clean_ids, hidden, mlp, [], False)
        for index in hidden:
            if index:
                handle = backend.layers[index - 1].register_forward_hook(remember(f"hidden_{index}"))
                hooks.callback(handle.remove)
        for index in mlp:
            handle = backend.layers[index].mlp.down_proj.register_forward_pre_hook(remember(f"mlp_{index}", pre=True))
            hooks.callback(handle.remove)
        for index, block in enumerate(backend.layers):
            # 两条分支共同构成该层写入残差流的内容。跳连的梯度照常传递。
            for module in (block.self_attn, block.mlp):
                handle = module.register_forward_hook(drop_branch(index))
                hooks.callback(handle.remove)
        input_ids = torch.tensor([corrupt_ids], device=backend.device)
        inputs = backend.model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        if 0 in hidden:
            captures["hidden_0"] = inputs
        usage["executed_model_forwards"] += 1
        output = backend.model(inputs_embeds=inputs, attention_mask=torch.ones_like(input_ids),
                               use_cache=False, logits_to_keep=len(corrupt_ids) - query)
        score = token_metrics(output.logits[0, 0], pair["target_token"], pair["alternative_token"])[metric]
        nodes = [captures[key] for key in keys]
        differences = [clean[key][0, positions].to(backend.device) - node.detach()[0, positions].float()
                       for key, node in zip(keys, nodes)]

        def estimates(retain_graph):
            usage["executed_model_backwards"] += 1
            grads = torch.autograd.grad(score, nodes, retain_graph=retain_graph, allow_unused=True)
            effects = torch.stack([(delta * grad[0, positions].float()).sum() if grad is not None else delta.new_zeros(())
                                   for delta, grad in zip(differences, grads)])
            return effects.detach().cpu().numpy()

        plain = estimates(True)
        magnitudes = np.zeros(len(nodes), dtype=np.float64)
        for layer in range(backend.num_layers):
            dropped_layer = layer
            magnitudes += np.abs(estimates(layer < backend.num_layers - 1))
    return np.stack((plain, magnitudes / (backend.num_layers - 1)), axis=-1)
