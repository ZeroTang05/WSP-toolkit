"""Calls Circuit Tracer and reports the output error of the feedforward substitution."""

from contextlib import ExitStack
from importlib.metadata import version
import json
from pathlib import Path
import sys


def fidelity(model, transcoders, tokens, target_ids, input_hook):
    """Compare the original model and the replacement model without error vectors on the same tokens."""
    import torch
    from torch.nn import functional as F

    native = model._model
    relative_error = []
    inputs = {}
    replace = False

    def capture(layer):
        def hook(module, args):
            inputs[layer] = args[0]
        return hook

    def compare(layer):
        def hook(module, args, original):
            source = inputs[layer] if input_hook == "hook_resid_mid" else args[0]
            reconstructed = transcoders[layer](source).to(original.dtype)
            # The attribution library retains the initial special token and does not generate feature nodes for that position.
            reconstructed[:, model.zero_positions] = original[:, model.zero_positions]
            if replace:
                return reconstructed
            difference = (original.float() - reconstructed.float()).norm()
            relative_error.append(float(difference / original.float().norm().clamp_min(1e-12)))
        return hook

    with torch.inference_mode(), ExitStack() as hooks:
        for index, layer in enumerate(native.model.layers):
            if input_hook == "hook_resid_mid":
                handle = layer.post_attention_layernorm.register_forward_pre_hook(capture(index))
                hooks.callback(handle.remove)
            handle = layer.mlp.register_forward_hook(compare(index))
            hooks.callback(handle.remove)
        reference = native(tokens[None, :], use_cache=False, logits_to_keep=1).logits[0, 0].float().log_softmax(-1)
        replace = True
        candidate = native(tokens[None, :], use_cache=False, logits_to_keep=1).logits[0, 0].float().log_softmax(-1)
    return {"kl_original_to_replacement": float(F.kl_div(candidate, reference, log_target=True, reduction="sum")),
            "original_top_token": int(reference.argmax()), "replacement_top_token": int(candidate.argmax()),
            "targets": [{"token_id": token, "original_log_probability": float(reference[token]),
                         "replacement_log_probability": float(candidate[token])} for token in target_ids],
            "relative_mlp_error_by_layer": relative_error,
            "replacement_scope": "feed_forward_without_error_vectors_except_prefix"}


def graph_records(graph, node_threshold, edge_threshold):
    """Preserve the order of upstream nodes and convert to a general directed graph format according to the same matrix index."""
    import torch
    from circuit_tracer.graph import prune_graph

    nodes = []

    def node(label, level, kind, **coordinates):
        nodes.append({"id": f"node_{len(nodes)}", "label": label, "level": level,
                      "kind": kind, "matrix_index": len(nodes), **coordinates})

    selected = graph.selected_features
    for index in selected.tolist():
        layer, position, feature = graph.active_features[index].tolist()
        node(f"L{layer} F{feature} @{position}", layer + 1, "feature", layer=layer,
             position=position, feature_id=feature, activation=float(graph.activation_values[index]))
    for layer in range(graph.cfg.n_layers):
        for position in range(graph.n_pos):
            node(f"L{layer} error @{position}", layer + 1, "reconstruction_error", layer=layer, position=position)
    for position, token in enumerate(graph.input_tokens.tolist()):
        node(f"Token {token} @{position}", 0, "embedding", position=position, token_id=token)
    for target, probability in zip(graph.logit_targets, graph.logit_probabilities.tolist()):
        node(f"Output {target.token_str.strip()}", graph.cfg.n_layers + 1, "logit", token_id=target.vocab_idx,
             probability=probability)
    matrix = graph.adjacency_matrix.cpu().float()
    if matrix.shape != (len(nodes), len(nodes)) or not torch.isfinite(matrix).all():
        raise ValueError("Circuit Tracer adjacency does not match the node index")
    pruned = prune_graph(graph, node_threshold=node_threshold, edge_threshold=edge_threshold)
    kept = pruned.node_mask.cpu()
    candidates = torch.nonzero(pruned.edge_mask.cpu() & kept[:, None] & kept[None, :] & matrix.ne(0))
    scores = matrix[candidates[:, 0], candidates[:, 1]]
    order = scores.abs().argsort(descending=True, stable=True)[:64]
    edges = [{"source": f"node_{source}", "target": f"node_{target}", "score": float(score)}
             for (target, source), score in zip(candidates[order].tolist(), scores[order].tolist())]
    return nodes, edges, matrix.numpy()


def main(request_path):
    """Generate a gradient graph on the bound real checkpoint, outputting raw values and a plottable summary."""
    import numpy as np
    import torch
    from circuit_tracer import ReplacementModel
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.graph import compute_graph_scores
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder_set
    from safetensors.torch import load_file

    packages = {name: version(name) for name in ("circuit-tracer", "nnsight", "transformers", "torch")}
    expected = {"circuit-tracer": "0.5.0", "nnsight": "0.6.0", "transformers": "4.57.3"}
    if any(packages[name] != release for name, release in expected.items()):
        raise ValueError("Use the documented Circuit Tracer runtime versions")
    directory = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transcoders = load_transcoder_set(dict(enumerate(request["files"])), request["transcoder_id"],
        request["feature_input_hook"], request["feature_output_hook"], device=device, dtype=torch.float32)
    model = ReplacementModel.from_pretrained_and_transcoders(request["checkpoint"], transcoders,
                                                             backend="nnsight", device=device, dtype=torch.float32)
    if len(transcoders) != model.cfg.n_layers or any(layer.d_model != model.cfg.d_model for layer in transcoders):
        raise ValueError("Transcoder layer count and widths must match the original model")
    overlay = directory / "parent.safetensors"
    if overlay.is_file():
        with torch.no_grad():
            for name, value in load_file(overlay).items():
                model._model.get_parameter(name).copy_(value)
    tokens = model.ensure_tokenized(request["token_ids"])
    targets = request["target_tokens"]
    report = fidelity(model, transcoders, tokens, targets, request["feature_input_hook"])
    graph = attribute(tokens, model, attribution_targets=torch.tensor(targets, device=device),
                      batch_size=4, max_feature_nodes=request["max_feature_nodes"])
    nodes, edges, matrix = graph_records(graph, request["node_threshold"], request["edge_threshold"])
    replacement_score, completeness = compute_graph_scores(graph)
    result = {"nodes": nodes, "edges": edges, "input_token_ids": graph.input_tokens.tolist(),
              "active_features": len(graph.active_features), "selected_features": len(graph.selected_features),
              "fidelity": report, "graph_scores": {"replacement_score": replacement_score, "completeness": completeness},
              "runtime": packages}
    np.save(directory / "adjacency.npy", matrix, allow_pickle=False)
    (directory / "result.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
