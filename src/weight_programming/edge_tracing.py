"""为三种回路方法共用连接坐标、预算和可读证据。"""

from math import isfinite
import random
from statistics import mean
from time import perf_counter

from torch.nn import functional as F

from .circuit_interventions import ResidualCircuit, score
from .patching import prepare_pair


_METRIC = "logit_difference"
_KL = "kl_divergence"
_MAX_TOKENS = 128


class EdgeTracing:
    """在已登记的版本上执行实际边干预或可微分的边归因。"""

    def __init__(self, tools):
        self.tools = tools
        backend = tools.forward.backend
        self.circuit = None
        if tools.allow_interventions and backend.identity["model_type"] in ("llama", "qwen3"):
            self.circuit = ResidualCircuit(backend)
        self.methods = ["path_patching", "acdc"] if self.circuit else []
        if self.circuit and tools.allow_gradients:
            self.methods.append("eap")

    def capabilities(self):
        """列出真正可以作为边端点的模块级坐标。"""
        if not self.circuit:
            return {}
        return {method: {"max_tokens": _MAX_TOKENS, "granularity": "residual_components"}
                for method in self.methods}

    def _prepare(self, pairs):
        backend, policy = self.tools.forward.backend, self.tools.forward.policy
        cases = []
        for pair in pairs:
            clean, corrupt, query = prepare_pair(backend, policy, pair["clean_text"], pair["corrupt_text"],
                                                 pair["target_token"], pair["alternative_token"], pair.get("query_position"))
            if len(clean) > _MAX_TOKENS or any(not 0 <= pos < len(clean) for pos in pair.get("positions", [])):
                raise ValueError("Circuit edges need aligned inputs of at most 128 tokens and valid positions")
            cases.append((pair, clean, corrupt, query))
        return cases

    def run(self, method, variant_id, pairs, *, metric=None, path=None, edges=None, review_pairs=(),
            forward_budget=64, threshold=None, control_seed=0, strength=1.0, top_k=8):
        """预先检查所有调用量，按方法保存实测或梯度产物。"""
        if method not in self.methods:
            raise ValueError(f"Circuit method is unavailable on this model or permission profile: {method}")
        circuit, tools = self.circuit, self.tools
        discovery, review = self._prepare(pairs), self._prepare(review_pairs)
        if {(tuple(case[1]), tuple(case[2])) for case in discovery} & {(tuple(case[1]), tuple(case[2])) for case in review}:
            raise ValueError("Discovery and review must use distinct aligned inputs")
        metric = metric or (_KL if method == "acdc" else _METRIC)
        if method == "path_patching":
            if path[-1] != "output" or not isfinite(strength):
                raise ValueError("An isolated path must end at output and use finite strength")
            links = list(zip(path, path[1:]))
            discovery_calls, review_calls = 3, 3
        else:
            links = [(row["source"], row["target"]) for row in edges]
            if method == "acdc" and (not isfinite(threshold) or threshold <= 0):
                raise ValueError("ACDC requires a finite positive pruning threshold")
            discovery_calls, review_calls = (2 + len(links), 5) if method == "acdc" else (2, 0)
        circuit.validate(links)
        forward_count = discovery_calls * len(discovery) + review_calls * len(review)
        tokens = discovery_calls * sum(len(case[1]) for case in discovery) + review_calls * sum(len(case[1]) for case in review)
        writers = len({source for source, _ in links})
        cache_bytes = sum(len(case[1]) for case in discovery + review) * writers * circuit.backend.hidden_size * 4
        if (forward_count > forward_budget or tokens > tools.forward.policy.max_total_text_tokens
                or cache_bytes > 256 * 1024**2):
            raise ValueError("Circuit intervention exceeds forward, token or activation-cache budget")
        snapshot = tools.versions.snapshot(variant_id)
        ledger = tools.forward.ledger
        request = {"type": "trace_circuit", "method": method, "metric": metric, "pairs": pairs,
                   "review_pairs": review_pairs, "edges": links, "threshold": threshold,
                   "strength": strength, "control_seed": control_seed, "top_k": top_k, "forward_budget": forward_budget}
        attempt = ledger.begin(request, variant_id)
        run_id = f"request_{attempt:04d}"
        usage = {"executed_model_forwards": 0, "executed_model_backwards": 0, "input_tokens": 0}
        started = perf_counter()
        try:
            with circuit.backend.activate(snapshot):
                if method == "path_patching":
                    rows, graph_edges, summary = self._path(path, discovery, review, metric, strength, usage)
                elif method == "eap":
                    rows, graph_edges, summary = self._eap(links, discovery, metric, top_k, usage)
                else:
                    rows, graph_edges, summary = self._acdc(links, discovery, review, metric, threshold, control_seed, usage)
            evidence = "gradient_estimate" if method == "eap" else "intervention_effect"
            report = tools.artifacts.save_records("circuit_results", rows)
            nodes = {node for edge in links for node in edge}
            graph = tools.artifacts.save_records("circuit_graph", [{
                "method": method, "evidence_type": evidence, "source_run_id": run_id, "source_artifact_id": report,
                "nodes": [{"id": node, "label": node, "level": level}
                          for node, level in circuit.order.items() if node in nodes],
                "edges": graph_edges}])
            usage.update(artifact_id=report, graph_artifact_id=graph, elapsed_seconds=perf_counter() - started)
            ledger.finish(attempt, True, usage)
        except Exception as error:
            ledger.finish(attempt, False, {**usage, "error_type": type(error).__name__, "error": str(error)[:500]})
            raise
        return {"method": method, "metric": metric, "variant_id": variant_id, "source_run_id": run_id,
                "source_artifact_ids": [], "evidence_type": evidence, "artifact_id": report,
                "graph_artifact_id": graph, "usage": usage, **summary}

    def _path(self, path, discovery, review, metric, strength, usage):
        circuit = self.circuit
        rows = []
        for split, cases in (("discovery", discovery), ("review", review)):
            for index, (pair, clean, corrupt, query) in enumerate(cases):
                base, writes, reads = circuit.forward(clean, query, usage, sources=path[:-1], targets=path[1:])
                changed, donor, _ = circuit.forward(corrupt, query, usage, sources=[path[0]])
                patched = circuit.isolated_path(clean, query, usage, path, writes, reads, donor,
                                               pair.get("positions"), strength)
                clean_score, corrupt_score, patched_score = [float(score(logits, pair, metric))
                                                             for logits in (base, changed, patched)]
                rows.append({"split": split, "pair": index, "clean_score": clean_score,
                             "corrupt_score": corrupt_score, "patched_score": patched_score,
                             "effect": patched_score - clean_score})
        discovery_effect = mean(row["effect"] for row in rows[:len(discovery)])
        # 路径只有一个实测分数，图上的各段引用同一次路径干预。
        graph = [{"source": source, "target": target, "score": discovery_effect, "score_scope": "whole_path"}
                 for source, target in zip(path, path[1:])]
        summary = {"mean_effect": discovery_effect, "path": path,
                   "review_mean_effect": mean(row["effect"] for row in rows[len(discovery):]) if review else None}
        return rows, graph, summary

    def _eap(self, edges, discovery, metric, top_k, usage):
        effects = [self.circuit.edge_estimates(clean, corrupt, query, usage, edges, pair, metric)
                   for pair, clean, corrupt, query in discovery]
        rows = [{"source": source, "target": target, "score": mean(values), "score_scope": "edge_estimate",
                 "mean_absolute_effect": mean(map(abs, values)), "pair_effects": list(values)}
                for (source, target), values in zip(edges, zip(*effects))]
        ranking = sorted(rows, key=lambda row: (-row["mean_absolute_effect"], row["source"], row["target"]))
        graph = [{key: value for key, value in row.items() if key != "pair_effects"} for row in ranking[:top_k]]
        return rows, graph, {"top_edges": graph, "edges": len(edges)}

    def _acdc(self, edges, discovery, review, metric, threshold, control_seed, usage):
        circuit = self.circuit
        sources = list(dict.fromkeys(source for source, _ in edges))

        def baselines(cases):
            for pair, clean, corrupt, query in cases:
                base, _, _ = circuit.forward(clean, query, usage)
                changed, writes, _ = circuit.forward(corrupt, query, usage, sources=sources)
                yield pair, clean, query, base, changed, writes

        def measure(logits, base, pair):
            if metric == _KL:
                return float(F.kl_div(logits.log_softmax(-1), base.log_softmax(-1), log_target=True, reduction="sum"))
            return float(score(logits, pair, metric))

        def trial(baseline, removed):
            pair, clean, query, base, _, writes = baseline
            logits = circuit.replace_edges(clean, query, usage, removed, writes, pair.get("positions"))
            return measure(logits, base, pair)

        data = list(baselines(discovery))
        previous = mean(measure(base, base, pair) for pair, _, _, base, _, _ in data)
        removed, kept, steps = [], [], []
        ordered = sorted(edges, key=lambda edge: (circuit.order[edge[1]], circuit.order[edge[0]]), reverse=True)
        for edge in ordered:
            candidate = mean(trial(row, [*removed, edge]) for row in data)
            degradation = candidate - previous if metric == _KL else previous - candidate
            drop = degradation < threshold
            steps.append({"source": edge[0], "target": edge[1], "score": degradation, "kept": not drop,
                          "before": previous, "candidate": candidate})
            if drop:
                removed.append(edge)
                previous = candidate
            else:
                kept.append(edge)
        rows = [{"stage": "pruning", **row} for row in steps]
        if review:
            control_kept = set(random.Random(control_seed).sample(edges, len(kept)))
            random_removed = [edge for edge in edges if edge not in control_kept]
            for index, row in enumerate(baselines(review)):
                pair, _, _, base, changed, _ = row
                rows.append({"stage": "review", "pair": index, "clean_score": measure(base, base, pair),
                             "corrupt_score": measure(changed, base, pair), "retained_score": trial(row, removed),
                             "removed_score": trial(row, kept), "random_retained_score": trial(row, random_removed)})
        return rows, steps, {"kept_edges": [{"source": source, "target": target} for source, target in kept],
                             "removed_edges": len(removed), "final_score": previous, "review_pairs": len(review)}
