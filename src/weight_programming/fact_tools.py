"""将 EasyEdit 的知识编辑结果发布为工具包的候选补丁。"""

import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

from safetensors.torch import load_file, save_file

from .editing_tools import _save
from .reconstruction import input_hessian


class FactTools:
    """在单独的 EasyEdit 环境计算改动，主模型继续用于校准和评测。"""

    def __init__(self, tools, python, source):
        self.tools = tools
        self.python = Path(python).expanduser().absolute()
        self.source = Path(source).expanduser().resolve()
        if not self.python.is_file() or not (self.source / "easyeditor" / "__init__.py").is_file():
            raise ValueError("Provide an EasyEdit Python executable and source checkout")
        if tools.forward.backend.identity["model_type"] not in ("llama", "qwen3"):
            raise ValueError("The fact editor requires a Llama or Qwen3 decoder")

    def _run_editor(self, directory):
        """通过文件交换数值，分离两个 Transformers 运行环境。"""
        script = Path(__file__).with_name("fact_worker.py").resolve()
        env = dict(os.environ, PYTHONPATH=str(self.source), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        log_path = directory / "editor.log"
        with log_path.open("wb") as log:
            result = subprocess.run([str(self.python), str(script), str(directory / "request.json")],
                                    cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=600, check=False)
        if result.returncode:
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 2400))
                raise ValueError("EasyEdit failed in its own runtime:\n" + log.read().decode(errors="replace"))
        report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        return load_file(directory / "edited.safetensors", device="cpu"), report

    def edit_facts(self, parent_variant_id, method, facts, layers, calibration_texts,
                   steps=20, learning_rate=0.3, relative_damping=0.01, mom2_weight=15000.0):
        """用登记模型的实际激活准备二阶统计，再执行 ROME 或 MEMIT。"""
        tools = self.tools
        if not tools.allow_edits or not tools.allow_gradients:
            raise PermissionError("Fact editing requires edit and gradient permissions")
        if method == "rome" and (len(facts) != 1 or len(layers) != 1):
            raise ValueError("ROME edits one fact at one layer per call")
        backend = tools.forward.backend
        if any(not 0 <= layer < backend.num_layers for layer in layers) or layers != sorted(set(layers)):
            raise ValueError("Provide distinct, ascending model layer indices")
        if not all(math.isfinite(value) and value > 0 for value in (learning_rate, relative_damping, mom2_weight)):
            raise ValueError("Fact editing numerical settings must be finite and positive")
        requests = []
        for index, fact in enumerate(facts):
            prompt = fact["prompt"]
            if prompt.count("{}") != 1 or any(brace in prompt.replace("{}", "") for brace in "{}"):
                raise ValueError("Each fact prompt needs exactly one {} subject placeholder")
            if any(not fact[key].strip() or any(brace in fact[key] for brace in "{}")
                   for key in ("subject", "target_new", "target_true")):
                raise ValueError("Subject and both targets must be nonempty plain text")
            if fact["target_new"].strip() == fact["target_true"].strip():
                raise ValueError("The new and old targets must differ")
            length = len(backend.encode(prompt.format(fact["subject"]))) + len(backend.encode(" " + fact["target_new"].strip()))
            if length > min(256, backend.context_length):
                raise ValueError("A fact prompt and completion can use at most 256 tokens")
            requests.append({"case_id": index, **fact})
        if len({row["prompt"].format(row["subject"]) for row in requests}) != len(requests):
            raise ValueError("Each fact must have a distinct rendered prompt")

        sites = [f"block.{layer}.mlp.down_proj.input" for layer in layers]
        names = [f"model.layers.{layer}.mlp.down_proj.weight" for layer in layers]
        if any(name not in tools.versions.groups for name in names):
            raise ValueError("A requested layer has no editable feed-forward output projection")
        if len(layers) * backend.mlp_width**2 * 4 > 1024**3:
            raise ValueError("Requested second-moment matrices exceed the 1 GiB transfer limit")
        started = perf_counter()
        run = tools.model_forward(parent_variant_id, "teacher_forced", texts=calibration_texts, observe=sites)
        refs = [run["artifacts"][f"mlp_{layer}"]["artifact_id"] for layer in layers]
        with TemporaryDirectory(prefix="wsp-fact-", dir=tools.versions.root) as temporary:
            directory = Path(temporary)
            token_count = 0
            for layer, name, ref in zip(layers, names, refs):
                rows = tools._input_rows(parent_variant_id, name, ref)
                token_count = len(rows)
                moment = input_hessian(rows).mul_(0.5)
                mean_diagonal = float(moment.diagonal().mean())
                if not mean_diagonal > 0:
                    raise ValueError(f"Calibration input has zero energy at layer {layer}")
                moment.diagonal().add_(relative_damping * mean_diagonal)
                save_file({"moment": moment.float()}, directory / f"moment_{layer}.safetensors")
            snapshot = tools.versions.snapshot(parent_variant_id)
            if snapshot["values"]:
                save_file(snapshot["values"], directory / "parent.safetensors")
            request = {"method": method, "checkpoint": str(tools.forward.store.model_dir), "device": str(backend.device),
                       "layers": layers, "names": names, "facts": requests, "steps": steps, "learning_rate": learning_rate,
                       "mom2_weight": mom2_weight, "calibration_samples": len(calibration_texts), "calibration_tokens": token_count,
                       "context_templates": ["{}", "As a reference, {}", "In general, {}"]}
            (directory / "request.json").write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
            replacements, editor = self._run_editor(directory)
        if set(replacements) != set(names):
            raise ValueError("EasyEdit returned weights outside the authorized layers")

        evidence = {"method": method, "parent_variant_id": parent_variant_id, "facts": requests, "layers": layers,
                    "calibration_run_id": run["run_id"], "calibration_tokens": token_count,
                    "relative_damping": relative_damping, "mom2_weight": mom2_weight if method == "memit" else None,
                    "steps": steps, "learning_rate": learning_rate, "context_templates": request["context_templates"],
                    "editor": editor, "elapsed_seconds": perf_counter() - started}
        report_id = tools.artifacts.save_records("fact_edit", [evidence])
        patch = _save(tools.versions, parent_variant_id, replacements, method, source_ids=[*refs, report_id])
        return {"patch_id": patch["patch_id"], "parent_variant_id": parent_variant_id, "method": method,
                "report_artifact_id": report_id, "changed_elements": sum(row["changed"] for row in patch["tensors"].values())}
