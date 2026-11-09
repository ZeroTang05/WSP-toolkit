"""从已保存的自注意力矩阵计算跨层传播和最大流。"""

import numpy as np


_ATTENTION_AXES = ["sample", "head", "query_token", "key_token"]


def attention_layers(artifacts, run_id, site_ids, sample_index, max_tokens):
    """从一次快照读取对齐的注意力，并加入残差连接。"""
    references = artifacts.references(run_id)
    tokens, _ = artifacts.array(references["input_ids"])
    valid, _ = artifacts.array(references["valid_tokens"])
    if sample_index >= len(tokens):
        raise ValueError("Sample index is outside this forward run")
    positions = np.flatnonzero(valid[sample_index])
    if not 0 < len(positions) <= max_tokens:
        raise ValueError(f"This attention method requires 1 to {max_tokens} valid tokens")

    identity = np.eye(len(positions))
    matrices = []
    source_ids = []
    for layer, site_id in enumerate(site_ids):
        source_id = references.get(f"attention_{layer}")
        if source_id is None:
            raise ValueError(f"Forward run is missing {site_id}; capture every layer in the prefix")
        values, info = artifacts.array(source_id)
        if info["axes"] != _ATTENTION_AXES or info.get("site_id") != site_id:
            raise ValueError(f"Expected self-attention probabilities from {site_id}")
        attention = values[sample_index][:, positions[:, None], positions].mean(axis=0, dtype=np.float64)
        if not np.isfinite(attention).all() or (attention < 0).any() or not np.allclose(attention.sum(1), 1):
            raise ValueError(f"Invalid attention probabilities on valid tokens at {site_id}")
        # 单位矩阵表示残差沿同一词元位置传递。
        combined = attention + identity
        matrices.append(combined / combined.sum(axis=1, keepdims=True))
        source_ids.append(source_id)

    return matrices, source_ids, positions.tolist(), tokens[sample_index, positions].tolist()


def attention_rollout(matrices):
    """按 Abnar 与 Zuidema 的方法累乘含残差的注意力矩阵。"""
    rollout = matrices[0]
    for attention in matrices[1:]:
        rollout = attention @ rollout
    return rollout


def attention_flow(matrices, query_index):
    """按论文作者的做法，独立计算查询词元到每个输入词元的最大流。"""
    import networkx as nx

    width = matrices[0].shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from((layer, token) for layer in range(len(matrices) + 1) for token in range(width))
    for layer, attention in enumerate(matrices, start=1):
        queries, keys = attention.nonzero()
        graph.add_edges_from(((layer, int(query)), (layer - 1, int(key)), {"capacity": float(attention[query, key])})
                             for query, key in zip(queries, keys))
    source = (len(matrices), query_index)
    # 每个输入词元各自作为汇点，原始容量和归一化分数分别保留。
    raw = np.array([nx.maximum_flow_value(graph, source, (0, token), flow_func=nx.algorithms.flow.edmonds_karp)
                    for token in range(width)])
    return (raw / raw.sum())[None, :], raw.tolist()


def attention_flow_graph(tokenizer, positions, token_ids, query, scores, evidence, source_artifact_id):
    """把逐词元最大流保存为可画的有向图。每条边汇总跨层结果。"""
    decoded = tokenizer.batch_decode([[token] for token in token_ids])
    labels = [f"{position} · {text.strip() or repr(text)}" for position, text in zip(positions, decoded)]
    nodes = [{"id": "query", "label": labels[positions.index(query)], "level": 0, "token_position": query},
             *[{"id": f"input_{position}", "label": label, "level": 1, "token_position": position}
               for position, label in zip(positions, labels)]]
    edges = [{"source": "query", "target": f"input_{position}", "score": float(score), "raw_flow": raw}
             for position, score, raw in zip(positions, scores, evidence["raw_flow"]) if raw > 0]
    return {"nodes": nodes, "edges": edges, "method": "attention_flow",
            "evidence_type": evidence["evidence_type"], "source_artifact_id": source_artifact_id,
            "source_run_id": evidence["source_run_id"], "through_layer": evidence["through_layer"]}
