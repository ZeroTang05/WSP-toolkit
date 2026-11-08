"""由宿主登记本地检查点或固定提交的 Hub 模型。"""

import json
from hashlib import sha256
from pathlib import Path
import re

from .tensor_store import WeightStore, sha


def register_model(source, workspace, *, revision=None):
    """核对来源并登记模型，返回权重存储及模型描述。"""
    workspace = Path(workspace).resolve()
    if revision is None:
        model_dir = Path(source).resolve(strict=True)
        origin = {"kind": "local", "path": str(model_dir)}
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Hub checkpoints require a full commit SHA")
        from huggingface_hub import snapshot_download

        key = sha256(f"{source}@{revision}".encode()).hexdigest()
        model_dir = Path(snapshot_download(repo_id=str(source), revision=revision,
                                           local_dir=workspace / "checkpoints" / key)).resolve()
        origin = {"kind": "hub", "repo_id": str(source), "revision": revision}
    model_type = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["model_type"]
    store = WeightStore(model_dir, workspace / "edits")
    base = store.snapshot(with_edits=False)
    tokenizer_files = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json",
                       "merges.txt", "special_tokens_map.json", "chat_template.jinja")
    tokenizer_hashes = {name: sha(model_dir / name) for name in tokenizer_files if (model_dir / name).is_file()}
    info = {"base_id": base["base_id"], "variant_id": base["variant_id"], "model_type": model_type,
            "source": origin, "tokenizer_files": tokenizer_hashes, "tensor_count": len(store.names)}
    manifest = workspace / "registered_model.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != info:
            raise ValueError("Workspace is already registered to a different checkpoint or tokenizer")
    else:
        staged = manifest.with_suffix(".tmp")
        staged.write_text(json.dumps(info, indent=2), encoding="utf-8")
        staged.replace(manifest)
    return store, info
