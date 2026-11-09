"""用隔离的 Python 环境执行基于检查点的上游算法。"""

import json
import os
from pathlib import Path
import subprocess


def run_model_worker(python, script_name, directory, device, timeout=1800):
    """在同一设备的独立进程交换文件，并返回上游日志和结果。"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", HF_DATASETS_CACHE=str(directory / "datasets"))
    visible = env.get("CUDA_VISIBLE_DEVICES")
    env["CUDA_VISIBLE_DEVICES"] = "" if device.type == "cpu" else (
        str(device.index) if visible is None else visible.split(",")[device.index])
    env.pop("PYTHONPATH", None)
    script = Path(__file__).with_name(script_name).resolve()
    log_path = directory / "worker.log"
    with log_path.open("wb") as log:
        result = subprocess.run([str(python), str(script), str(directory / "request.json")], cwd=directory,
                                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    if result.returncode:
        with log_path.open("rb") as log:
            log.seek(max(0, log_path.stat().st_size - 4200))
            raise ValueError(f"{script_name} failed:\n" + log.read().decode(errors="replace"))
    return json.loads((directory / "result.json").read_text(encoding="utf-8"))
