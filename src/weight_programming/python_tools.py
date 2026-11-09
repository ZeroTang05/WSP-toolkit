"""将隔离 Python 会话接到模型版本、数组产物和补丁登记。"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from tempfile import TemporaryDirectory
from threading import RLock
from time import perf_counter
from uuid import uuid4

import numpy as np
from safetensors.numpy import load as load_arrays, save_file as save_arrays
from safetensors.torch import save_file as save_weights

from .cpu_kernel import CPUKernel, MAX_CELL_ARRAY_BYTES


SESSION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
ARRAY_PATTERN = r"^[A-Za-z][A-Za-z0-9_]{0,63}$"
RUN_PATTERN = r"python_[0-9a-f]{32}"


@dataclass
class KernelSession:
    """持有一个模型版本及其独立运行目录。"""

    variant_id: str
    folder: TemporaryDirectory
    kernel: CPUKernel
    work: Path

    def close(self):
        if not self.kernel.dead:
            self.kernel.close()
        self.folder.cleanup()


class PythonTools:
    """管理隔离进程，所有公开产物由宿主校验和发布。"""

    def __init__(self, tools, python=None):
        if not CPUKernel.supported():
            raise ValueError("Python execution requires Linux with bubblewrap")
        self.tools = tools
        self.python = python
        self.root = tools.versions.root / "python"
        self.runs = self.root / "runs"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.sessions = {}
        self.lock = RLock()

    def _session(self, session_id, variant_id):
        """开启一次真实隔离进程，之后沿用同一命名空间。"""
        current = self.sessions.get(session_id)
        if current is not None:
            if current.variant_id != variant_id:
                raise ValueError("Python session is bound to a different variant; choose a new session ID")
            return current
        if len(self.sessions) >= 8:
            raise ValueError("This inspection service can hold at most eight Python sessions")
        snapshot = self.tools.versions.snapshot(variant_id)
        folder = TemporaryDirectory(prefix="session-", dir=self.root)
        directory = Path(folder.name)
        practice = directory / "practice"
        practice.mkdir()
        work = directory / "work"
        kernel = None
        try:
            kernel = CPUKernel(self.tools.forward.store.model_dir, practice, work, directory / "control",
                               python=self.python, executions=50, seconds=30, memory_gib=16, versioned=True)
            if kernel.ready["base_id"] != snapshot["base_id"]:
                raise ValueError("Kernel checkpoint does not match the registered base")
            if snapshot["values"]:
                save_weights(snapshot["values"], kernel.exports / "variant.safetensors")
        except BaseException:
            if kernel is not None:
                kernel.close()
            folder.cleanup()
            raise
        current = KernelSession(variant_id, folder, kernel, work)
        self.sessions[session_id] = current
        return current

    def _inputs(self, session, run_id, inputs):
        """复制本次明确授权的数组到隔离进程的只读目录。"""
        data = {}
        total = 0
        for name, artifact_id in inputs.items():
            array, _ = self.tools.artifacts.array(artifact_id)
            total += array.nbytes
            if total > MAX_CELL_ARRAY_BYTES or array.dtype.kind not in "biuf" or not array.size:
                raise ValueError("Input artifacts must be nonempty numeric arrays totaling at most 32 MiB")
            data[name] = np.array(array, copy=True, order="C")
        if data:
            folder = session.kernel.exports / run_id
            folder.mkdir()
            save_arrays(data, folder / "inputs.safetensors")
        return total

    @staticmethod
    def _outputs(folder, outputs):
        """只读取常规文件，检查实际字节数、数组轴和有限值。"""
        data = {}
        used = 0
        if not outputs:
            return data, used
        directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for name, axes in outputs.items():
                fd = os.open(name + ".safetensors", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
                with os.fdopen(fd, "rb") as file:
                    info = os.fstat(file.fileno())
                    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_CELL_ARRAY_BYTES + 4096:
                        raise ValueError("Output files must be regular bounded Safetensors files")
                    payload = load_arrays(file.read(info.st_size))
                if set(payload) != {"value"}:
                    raise ValueError("Each output file must contain exactly one array")
                array = payload["value"]
                used += array.nbytes
                if (used > MAX_CELL_ARRAY_BYTES or array.ndim != len(axes) or len(axes) != len(set(axes))
                        or not array.size or array.dtype.kind not in "biuf" or not np.isfinite(array).all()):
                    raise ValueError("Output arrays must have finite values, unique axes and at most 32 MiB in total")
                data[name] = array
        finally:
            os.close(directory)
        return data, used

    def _write_run(self, record):
        """将实际用量与产物登记到宿主可读的运行记录。"""
        path = self.runs / (record["run_id"] + ".json")
        staged = path.with_suffix(".tmp")
        staged.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        staged.replace(path)

    def execute_python(self, session_id, variant_id, code, input_artifacts=None, outputs=None, edits=None):
        """执行一段代码，返回有限文本和通过宿主校验的产物。"""
        inputs = input_artifacts or {}
        outputs = outputs or {}
        edits = edits or []
        if not re.fullmatch(SESSION_PATTERN, session_id) or any(not re.fullmatch(ARRAY_PATTERN, name)
                                                              for name in (*inputs, *outputs)):
            raise ValueError("Invalid session or array name")
        if edits and not self.tools.allow_edits:
            raise PermissionError("Python patch submission requires edit permission")
        if any(edit["value"] not in outputs for edit in edits):
            raise ValueError("Every Python edit must refer to a declared output array")
        with self.lock:
            session = self._session(session_id, variant_id)
            run_id = "python_" + uuid4().hex
            record = {"run_id": run_id, "session_id": session_id, "variant_id": variant_id,
                      "status": "failed", "artifacts": {}, "patch_id": None,
                      "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "input_artifacts": inputs, "usage": {}}
            (self.runs / (run_id + ".py")).write_text(code, encoding="utf-8")
            started = perf_counter()
            try:
                record["usage"]["input_bytes"] = self._inputs(session, run_id, inputs)
                (session.work / run_id).mkdir()
                cell = session.kernel.execute(code, context={"run_id": run_id, "outputs": list(outputs)})
                record.update(output=cell["output"], truncated=cell["truncated"])
                if cell["status"] == "ok":
                    data, output_bytes = self._outputs(session.work / run_id, outputs)
                    record["usage"]["output_bytes"] = output_bytes
                    for name, array in data.items():
                        record["artifacts"][name] = self.tools.artifacts.save_array(name, array, outputs[name],
                                                                                    variant_id=variant_id,
                                                                                    source_run_id=run_id)
                    if edits:
                        sources = [record["artifacts"][edit["value"]] for edit in edits]
                        patch = self.tools.versions.create_patch(variant_id, edits, "execute_python",
                                                                  lambda edit: data[edit["value"]], sources)
                        record["patch_id"] = patch["patch_id"]
                    record["status"] = "succeeded"
            except Exception as error:
                record["error_type"] = type(error).__name__
                raise
            finally:
                record["usage"].update(executions=session.kernel.count, elapsed_seconds=perf_counter() - started)
                self._write_run(record)
            return record

    @staticmethod
    def get_run(workspace, run_id):
        """查一个已记录的隔离计算。"""
        if not re.fullmatch(RUN_PATTERN, run_id):
            raise ValueError("Invalid Python run ID")
        path = workspace / "python" / "runs" / (run_id + ".json")
        return json.loads(path.read_text(encoding="utf-8"))

    def close(self):
        """退出时回收本实例启动的进程和临时目录。"""
        with self.lock:
            for session in self.sessions.values():
                session.close()
            self.sessions.clear()
