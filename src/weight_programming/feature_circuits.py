"""校验模型和特征字典的绑定，并登记上游归因图。"""

import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
from safetensors.torch import save_file

from .tensor_store import sha
from .worker_runtime import run_model_worker


class FeatureCircuits:
    """在宿主登记的独立 Circuit Tracer 环境执行特征级归因。"""

    def __init__(self, tools, python=None):
        self.tools = tools
        self.root = tools.versions.root / "transcoders"
        self.python = None
        if python is not None:
            if not tools.allow_gradients or tools.forward.backend.identity["model_type"] not in ("llama", "qwen3"):
                raise ValueError("Circuit Tracer requires gradient permission and a Llama or Qwen3 decoder")
            self.python = Path(python).expanduser().absolute()
            if not self.python.is_file() or not os.access(self.python, os.X_OK):
                raise ValueError("Provide an executable Python from the Circuit Tracer environment")

    def _manifest(self, transcoder_id):
        """限制字典在工作区登记目录内，避免运行参数指定任意路径。"""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", transcoder_id):
            raise ValueError("Invalid transcoder ID")
        folder = self.root / transcoder_id
        path = folder / "manifest.json"
        if self.root.is_symlink() or folder.is_symlink() or path.is_symlink():
            raise ValueError("Use a local transcoder registration directory")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        model_type = self.tools.forward.backend.identity["model_type"]
        input_hooks = ("mlp.hook_in", "hook_resid_mid") if model_type == "llama" else ("mlp.hook_in",)
        output_hooks = ("mlp.hook_out", "hook_mlp_out") if model_type == "llama" else ("mlp.hook_out",)
        if (manifest.get("format_version") != 1 or manifest.get("model_kind") != "transcoder_set"
                or manifest.get("feature_input_hook") not in input_hooks
                or manifest.get("feature_output_hook") not in output_hooks):
            raise ValueError("Register a per-layer Circuit Tracer set with supported hooks")
        return folder, manifest

    def available(self, variant_id):
        """只列出与当前版本绑定的字典。"""
        if self.python is None:
            return []
        found = []
        for path in sorted(self.root.glob("*/manifest.json")):
            _, manifest = self._manifest(path.parent.name)
            if manifest["variant_id"] == variant_id:
                found.append(path.parent.name)
        return found

    def trace(self, variant_id, transcoder_id, text, target_tokens, max_feature_nodes=32,
              node_threshold=0.8, edge_threshold=0.98):
        """分别保存有向图、完整数值、节点索引和替代误差。"""
        if self.python is None:
            raise ValueError("Register a Circuit Tracer environment")
        tools = self.tools
        backend = tools.forward.backend
        token_ids = backend.encode(text)
        if len(token_ids) > min(31, backend.context_length - 1, tools.forward.policy.max_text_tokens - 1):
            raise ValueError("Feature tracing accepts at most 31 input tokens before the tracer prefix")
        if any(not 0 <= token < backend.vocab_size for token in target_tokens):
            raise ValueError("Attribution target is outside the model vocabulary")
        folder, manifest = self._manifest(transcoder_id)
        if manifest["variant_id"] != variant_id:
            raise ValueError("Transcoders must be registered to the selected model version")
        names = [f"layer_{i}.safetensors" for i in range(backend.num_layers)]
        digests = manifest["weights_sha256"]
        if set(digests) != set(names):
            raise ValueError("Register one transcoder weight file per model layer")
        for name in names:
            path = folder / name
            if path.is_symlink() or sha(path) != digests[name]:
                raise ValueError(f"Transcoder file differs from the registered digest: {name}")
        snapshot = tools.versions.snapshot(variant_id)
        request = {"type": "trace_circuit", "method": "feature_attribution_graph", "text": text,
                   "transcoder_id": transcoder_id, "target_tokens": target_tokens, "max_feature_nodes": max_feature_nodes,
                   "node_threshold": node_threshold, "edge_threshold": edge_threshold, "weights_sha256": digests}
        ledger = tools.forward.ledger
        attempt = ledger.begin(request, variant_id)
        run_id = f"request_{attempt:04d}"
        started = perf_counter()
        try:
            with TemporaryDirectory(prefix="feature-trace-", dir=tools.versions.root) as temporary:
                directory = Path(temporary)
                if snapshot["values"]:
                    save_file(snapshot["values"], directory / "parent.safetensors")
                worker_request = {**request, "checkpoint": str(tools.versions.store.model_dir), "token_ids": token_ids,
                                  "files": [str(folder / name) for name in names],
                                  "feature_input_hook": manifest["feature_input_hook"],
                                  "feature_output_hook": manifest["feature_output_hook"]}
                (directory / "request.json").write_text(json.dumps(worker_request), encoding="utf-8")
                result = run_model_worker(self.python, "feature_circuit_worker.py", directory, backend.device)
                adjacency = np.load(directory / "adjacency.npy", allow_pickle=False)
            nodes, edges = result.pop("nodes"), result.pop("edges")
            if adjacency.shape != (len(nodes), len(nodes)) or not np.isfinite(adjacency).all() or not edges:
                raise ValueError("Circuit Tracer produced an invalid graph")
            evidence = "replacement_model_attribution"
            artifact = tools.artifacts.save_array("feature_adjacency", adjacency, ["target_node", "source_node"],
                run_id=run_id, variant_id=variant_id, method=request["method"], evidence_type=evidence, source_run_id=run_id)
            nodes_id = tools.artifacts.save_records("feature_nodes", nodes)
            result["dictionary_provenance"] = manifest.get("provenance")
            report_id = tools.artifacts.save_records("feature_attribution_report", [result])
            visible = {edge[key] for edge in edges for key in ("source", "target")}
            graph = tools.artifacts.save_records("circuit_graph", [{"method": request["method"], "evidence_type": evidence,
                "source_run_id": run_id, "source_artifact_id": artifact, "nodes_artifact_id": nodes_id,
                "nodes": [node for node in nodes if node["id"] in visible], "edges": edges}])
            usage = {"elapsed_seconds": perf_counter() - started, "artifact_id": artifact,
                     "graph_artifact_id": graph, "nodes_artifact_id": nodes_id, "report_artifact_id": report_id}
            ledger.finish(attempt, True, usage)
        except Exception as error:
            ledger.finish(attempt, False, {"error_type": type(error).__name__, "error": str(error)[:500]})
            raise
        return {"method": request["method"], "variant_id": variant_id, "transcoder_id": transcoder_id,
                "source_run_id": run_id, "source_artifact_ids": [artifact, nodes_id, report_id], "evidence_type": evidence,
                "artifact_id": artifact, "graph_artifact_id": graph, "nodes_artifact_id": nodes_id,
                "report_artifact_id": report_id, "node_count": len(nodes), "edges_shown": len(edges),
                "fidelity": result["fidelity"], "usage": usage}
