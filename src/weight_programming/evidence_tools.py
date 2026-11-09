"""从已登记产物渲染可追溯的热力图和有向图。"""

from math import isfinite

import numpy as np

from .tensor_store import slice_region


_VIEW_AXES = {"activation_heatmap": ("token", "channel"),
              "attention_heatmap": ("query_token", "key_token"),
              "patching_heatmap": ("pair", "site"),
              "similarity_heatmap": ("left_site", "right_site"),
              "attribution_heatmap": ("sample", "token")}


def render_evidence(artifacts, artifact_id, view, region=None, axes=None, format="png", max_edges=24):
    """按指定轴画图，保留范围、色标和重点数值坐标。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if view == "graph":
        return _render_graph(artifacts, artifact_id, format, max_edges)
    array, metadata = artifacts.array(artifact_id)
    available = metadata["axes"]
    axes = list(_VIEW_AXES[view] if axes is None else axes)
    if len(axes) != 2 or axes[0] == axes[1] or any(axis not in available for axis in axes):
        raise ValueError(f"Select two distinct plotted axes from {available}")
    plotted = [available.index(axis) for axis in axes]
    other = [index for index in range(array.ndim) if index not in plotted]
    if region is None:
        region = [[0, min(length, 128) if index in plotted else 1] for index, length in enumerate(array.shape)]
    index = slice_region(array.shape, region, max_values=16_384)
    if any(region[axis][1] - region[axis][0] != 1 for axis in other):
        raise ValueError("Select a single position on each axis outside the plot")
    selected = array[index].transpose(plotted + other)
    matrix = np.asarray(selected.reshape(selected.shape[:2]), dtype=np.float64)
    finite = np.isfinite(matrix)
    if not finite.any():
        raise ValueError("Selected matrix has no finite entries")
    figure = Figure(figsize=(7, 4), layout="constrained")
    FigureCanvasAgg(figure)
    plot = figure.subplots()
    signed = view in ("activation_heatmap", "patching_heatmap", "attribution_heatmap")
    if signed:
        maximum = float(np.abs(matrix[finite]).max()) or 1.0
        scale = (-maximum, maximum)
    else:
        scale = (0.0, 1.0)
    colors = "RdBu_r" if signed else "viridis"
    image = plot.imshow(np.ma.masked_invalid(matrix), cmap=colors, aspect="auto", vmin=scale[0], vmax=scale[1])
    plot.set(xlabel=axes[1], ylabel=axes[0], title=view.replace("_", " "))
    figure.colorbar(image, ax=plot, label=metadata.get("metric", "value"))
    starts = [region[axis][0] for axis in plotted]
    label_keys = {"site": "site_ids", "left_site": "left_sites", "right_site": "right_sites", "feature": "feature_ids",
                  "query_token": "query_positions", "key_token": "token_positions"}
    labels = {axis: metadata.get(label_keys.get(axis)) for axis in axes}
    for axis, setter, start, length in zip(axes, (plot.set_yticks, plot.set_xticks), starts, matrix.shape):
        if length <= 12:
            axis_labels = labels[axis]
            setter(range(length), axis_labels[start:start + length] if axis_labels else range(start, start + length))
    scores = np.abs(matrix) if signed else matrix
    ranking = np.argsort(-np.where(finite, scores, -np.inf), axis=None, kind="stable")[:min(5, int(finite.sum()))]
    top_cells = []
    for flat_index in ranking:
        row, column = np.unravel_index(flat_index, matrix.shape)
        coordinates = {axes[0]: int(row) + starts[0], axes[1]: int(column) + starts[1]}
        cell = {"coordinates": coordinates, "value": float(matrix[row, column])}
        token_positions = {axis: labels[axis][index] for axis, index in coordinates.items()
                           if axis in ("query_token", "key_token") and labels[axis] is not None}
        if token_positions:
            cell["token_positions"] = token_positions
        top_cells.append(cell)
    result = artifacts.save_plot(figure, format, artifact_id)
    return {**result, "view": view, "region": region, "axes": axes, "shape": list(matrix.shape),
            "color_scale": list(scale), "top_cells": top_cells}


def _render_graph(artifacts, artifact_id, format, max_edges):
    """按节点层级和边的分数画图，并保留证据类型与排序明细。"""
    import networkx as nx
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    source = artifacts.read(artifact_id, limit=1)
    if source["kind"] != "records" or not source["records"]:
        raise ValueError("Graph rendering requires a node-and-edge record")
    record = source["records"][0]
    if not {"nodes", "edges", "evidence_type", "method"} <= record.keys():
        raise ValueError("Graph record needs nodes, edges, an evidence type and a method")
    nodes = {node["id"]: node for node in record["nodes"]}
    edges = record["edges"]
    pairs = {(edge["source"], edge["target"]) for edge in edges}
    if (len(nodes) != len(record["nodes"]) or len(edges) != len(pairs) or not edges
            or any(left not in nodes or right not in nodes for left, right in pairs)
            or any(not isfinite(edge["score"]) for edge in edges)):
        raise ValueError("Graph needs unique nodes and directed edges with valid endpoints and finite scores")
    ordered = sorted(edges, key=lambda edge: (-abs(edge["score"]), edge["source"], edge["target"]))[:max_edges]
    graph = nx.DiGraph()
    visible = {edge[key] for edge in ordered for key in ("source", "target")}
    graph.add_nodes_from((node_id, {"level": node["level"]}) for node_id, node in nodes.items() if node_id in visible)
    links = [(edge["source"], edge["target"]) for edge in ordered]
    graph.add_edges_from(links)
    positions = nx.multipartite_layout(graph, subset_key="level")
    figure = Figure(figsize=(9, max(3.5, min(12, len(visible) * 0.36))), layout="constrained")
    FigureCanvasAgg(figure)
    plot = figure.subplots()
    peak = abs(ordered[0]["score"]) or 1.0
    nx.draw_networkx_nodes(graph, positions, node_color="#e2e8f0", node_size=1050, ax=plot)
    nx.draw_networkx_edges(graph, positions, edgelist=links,
                           edge_color=["#b44a4a" if edge["score"] < 0 else "#356ab7" for edge in ordered],
                           style=["solid" if edge.get("kept", True) else "dashed" for edge in ordered],
                           width=[0.6 + 3 * abs(edge["score"]) / peak for edge in ordered],
                           node_size=1050, arrowsize=14, ax=plot)
    nx.draw_networkx_labels(graph, positions, {node_id: nodes[node_id]["label"] for node_id in graph},
                            font_size=8, ax=plot)
    plot.set_title(f"{record['method'].replace('_', ' ')} · {record['evidence_type'].replace('_', ' ')}")
    plot.margins(0.18)
    plot.set_axis_off()
    result = artifacts.save_plot(figure, format, artifact_id)
    return {**result, "view": "graph", "evidence_type": record["evidence_type"], "nodes_rendered": len(visible),
            "edges_rendered": len(ordered), "total_edges": len(edges), "top_edges": ordered[:5]}
