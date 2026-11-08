"""保存不可变的原精度补丁，并按候选版本解析权重。"""

from collections import defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .tensor_store import WeightStore, sha, slice_region


class ParentMismatch(ValueError):
    """补丁所需的父版本与当前提供的版本不同。"""


def _digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _tensor_digest(tensor):
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def _check_id(value):
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a content ID with 64 lowercase hex digits")
    return value


class VersionStore:
    """给一个基座登记候选分支，复用 WeightStore 读取和校验权重。"""

    def __init__(self, root, store, parameters):
        self.root = Path(root).resolve()
        if self.root == store.model_dir or store.model_dir in self.root.parents:
            raise ValueError("Patch workspace must be outside the source checkpoint")
        self.store = store
        self.base = store.snapshot(with_edits=False)
        self.patches = self.root / "patches"
        self.variants = self.root / "variants"
        self.parameters = parameters
        groups = defaultdict(list)
        for name in store.names:
            if name in parameters:
                groups[id(parameters[name])].append(name)
        self.groups = {name: tuple(names) for names in groups.values() for name in names}

    @contextmanager
    def _publish_lock(self):
        """同一个工作目录的进程共用发布锁。"""
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".publish.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def variant(self, variant_id):
        """读取已登记的候选，基座使用现有快照 ID。"""
        if variant_id == self.base["variant_id"]:
            return {"variant_id": variant_id, "base_id": self.base["base_id"],
                    "parent_variant_id": None, "patch_id": None}
        path = self.variants / f"{_check_id(variant_id)}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if _digest(record) != variant_id or record["base_id"] != self.base["base_id"]:
            raise ValueError("Invalid variant manifest")
        return {"variant_id": variant_id, **record}

    def _read_bundle(self, folder, patch_id):
        """验证清单与 Safetensors 文件确实属于同一份补丁。"""
        record = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if record["version"] != 1 or _digest(record) != patch_id:
            raise ValueError("Invalid patch manifest or ID")
        if record["base_id"] != self.base["base_id"]:
            raise ParentMismatch("Patch belongs to a different base checkpoint")
        if not record["tensors"] or sha(folder / "weights.safetensors") != record["file_sha256"]:
            raise ValueError("Patch payload does not match its manifest")
        return record

    def _sources(self, variant_id):
        """从子到父查找，保留每个张量最近的一次修改。"""
        sources = {}
        variant = self.variant(variant_id)
        while variant["patch_id"] is not None:
            patch_id = _check_id(variant["patch_id"])
            folder = self.patches / patch_id
            patch = self._read_bundle(folder, patch_id)
            if patch["parent_variant_id"] != variant["parent_variant_id"]:
                raise ParentMismatch("Variant and patch disagree on the parent")
            for name in patch["tensors"]:
                sources.setdefault(name, (folder / "weights.safetensors", variant["variant_id"]))
            variant = self.variant(variant["parent_variant_id"])
        return sources

    def _native(self, name, sources):
        if name in sources:
            with safe_open(str(sources[name][0]), framework="pt", device="cpu") as stream:
                return stream.get_tensor(name)
        return self.store._native(name)

    def read_slice(self, variant_id, name, region):
        """从最近的补丁或基座按原精度读取小切片。"""
        source = self._sources(variant_id).get(name)
        if source is None:
            return self.store.read_native_slice(name, region), self.base["variant_id"]
        with safe_open(str(source[0]), framework="pt", device="cpu") as stream:
            view = stream.get_slice(name)
            values = view[slice_region(view.get_shape(), region, max_values=4096)]
        return values, source[1]

    def tensor_versions(self, variant_id):
        """返回哪些张量来自候选分支。"""
        return {name: source[1] for name, source in self._sources(variant_id).items()}

    def read_native(self, variant_id, name):
        """为本地算法读取一个完整张量。"""
        return self._native(name, self._sources(variant_id))

    def read_tensors(self, variant_id, names):
        """多张量分析共用一次版本解析。"""
        sources = self._sources(variant_id)
        return {name: self._native(name, sources) for name in names}

    def snapshot(self, variant_id):
        """为一次前向准备一组固定的原精度替换张量。"""
        sources = self._sources(variant_id)
        return {"variant_id": variant_id, "base_id": self.base["base_id"], "native_patch": True,
                "values": {name: self._native(name, sources) for name in sources}}

    def _group(self, name):
        if name not in self.groups:
            raise ValueError(f"Tensor is not an editable checkpoint parameter: {name}")
        return self.groups[name]

    @torch.no_grad()
    def create_patch(self, parent_variant_id, edits, method, read_value, source_artifact_ids=(), source_patch_ids=()):
        """先完成整组编辑，再写一份可重放的补丁。"""
        sources = self._sources(parent_variant_id)
        grouped = defaultdict(list)
        for edit in edits:
            grouped[self._group(edit["tensor"])].append(edit)
        if not grouped:
            raise ValueError("A patch requires at least one edit")
        tensors, entries = {}, {}
        for group, operations in grouped.items():
            parent = self._native(group[0], sources)
            if parent.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"Unsupported patch precision: {group[0]}")
            proposed = parent.float().clone()
            regions = []
            for edit in operations:
                region = edit.get("region", [[0, size] for size in parent.shape])
                index = slice_region(parent.shape, region)
                if any(all(a < d and c < b for (a, b), (c, d) in zip(region, other)) for other in regions):
                    raise ValueError(f"Overlapping edits for {group[0]}")
                regions.append(region)
                value = torch.as_tensor(read_value(edit), dtype=torch.float32)
                if "values" in edit and value.numel() > 4096:
                    raise ValueError("Inline patch values are limited to 4096 elements")
                if value.shape != proposed[index].shape:
                    raise ValueError(f"Replacement shape does not match {group[0]}")
                if edit["op"] == "add":
                    proposed[index] += value
                elif edit["op"] == "replace":
                    proposed[index] = value
                else:
                    raise ValueError(f"Unknown patch operation: {edit['op']}")
            replacement = proposed.to(parent.dtype).contiguous()
            for name in group:
                old = self._native(name, sources) if name != group[0] else parent
                if not torch.equal(old, parent) or old.dtype != parent.dtype:
                    raise ValueError(f"Shared weights disagree in the parent: {name}")
                self.store._validate_edit(name, replacement)
                tensors[name] = replacement.clone() if name != group[0] else replacement
                entries[name] = self._entry(old, replacement, proposed)
        self.patches.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.patches) as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            save_file(tensors, str(bundle / "weights.safetensors"))
            record = {"version": 1, "base_id": self.base["base_id"], "parent_variant_id": parent_variant_id,
                      "method": method, "source_artifact_ids": list(source_artifact_ids), "tensors": entries,
                      "source_patch_ids": list(source_patch_ids),
                      "file_sha256": sha(bundle / "weights.safetensors")}
            (bundle / "manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            patch_id = _digest(record)
            self._publish_bundle(bundle, self.patches / patch_id, patch_id)
        return {"patch_id": patch_id, **record}

    @staticmethod
    def _entry(parent, replacement, proposed):
        """记录写回原精度后真正改变的数值。"""
        changed = replacement != parent
        return {"parent_sha256": _tensor_digest(parent), "replacement_sha256": _tensor_digest(replacement),
                "shape": list(parent.shape), "dtype": str(parent.dtype), "changed": int(changed.sum()),
                "rounded_away": int(((proposed != parent.float()) & ~changed).sum())}

    def _publish_bundle(self, source, destination, patch_id):
        with self._publish_lock():
            if destination.exists():
                self._read_bundle(destination, patch_id)
            else:
                source.rename(destination)

    def _verify_parent(self, folder, patch, parent_variant_id):
        if patch["parent_variant_id"] != parent_variant_id:
            raise ParentMismatch("Patch parent does not match the selected variant")
        sources = self._sources(parent_variant_id)
        tensors = load_file(str(folder / "weights.safetensors"), device="cpu")
        if tensors.keys() != patch["tensors"].keys():
            raise ValueError("Patch tensor names differ from the manifest")
        for name, tensor in tensors.items():
            parent = self._native(name, sources)
            entry = patch["tensors"][name]
            if entry["parent_sha256"] != _tensor_digest(parent):
                raise ParentMismatch(f"Parent tensor differs from patch: {name}")
            if (tensor.dtype != parent.dtype or tensor.shape != parent.shape
                    or str(tensor.dtype) != entry["dtype"] or list(tensor.shape) != entry["shape"]
                    or _tensor_digest(tensor) != entry["replacement_sha256"]):
                raise ValueError(f"Invalid replacement tensor: {name}")
            if any(alias not in tensors or not torch.equal(tensors[alias], tensor) for alias in self._group(name)):
                raise ValueError(f"Patch must preserve shared weights: {name}")
            self.store._validate_edit(name, tensor)
        return tensors

    def patch_tensors(self, parent_variant_id, patch_id):
        """为本地算法读取已经按共同父版本验证的替换张量。"""
        folder = self.patches / _check_id(patch_id)
        patch = self._read_bundle(folder, patch_id)
        return self._verify_parent(folder, patch, parent_variant_id)

    def apply_patch(self, parent_variant_id, patch_id):
        """校验补丁的父张量并发布候选版本。"""
        folder = self.patches / _check_id(patch_id)
        patch = self._read_bundle(folder, patch_id)
        self._verify_parent(folder, patch, parent_variant_id)
        record = {"base_id": self.base["base_id"], "parent_variant_id": parent_variant_id, "patch_id": patch_id}
        variant_id = _digest(record)
        with self._publish_lock():
            self.variants.mkdir(exist_ok=True)
            path = self.variants / f"{variant_id}.json"
            if not path.exists():
                staged = path.with_suffix(".tmp")
                staged.write_text(json.dumps(record, indent=2), encoding="utf-8")
                staged.replace(path)
            result = self.variant(variant_id)
        return {**result, "tensors": sorted(patch["tensors"])}

    def revert_patch(self, variant_id):
        """返回父版本 ID，供下一次运行或读权重使用。"""
        parent = self.variant(variant_id)["parent_variant_id"]
        if parent is None:
            raise ValueError("The base variant has no parent")
        return {"variant_id": self.variant(parent)["variant_id"], "reverted_variant_id": variant_id}

    def export_patch(self, patch_id):
        """把补丁清单和权重复制到宿主的导出目录。"""
        source = self.patches / _check_id(patch_id)
        self._read_bundle(source, patch_id)
        root = self.root / "exports" / "patches"
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as temporary:
            bundle = Path(temporary) / "bundle"
            shutil.copytree(source, bundle)
            self._read_bundle(bundle, patch_id)
            self._publish_bundle(bundle, root / patch_id, patch_id)
        return {"patch_id": patch_id, "bundle_path": f"exports/patches/{patch_id}"}

    def import_patch(self, bundle_id):
        """从宿主登记的导入目录验证并接纳补丁。"""
        source = self.root / "imports" / _check_id(bundle_id)
        self.patches.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=self.patches) as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            for name in ("manifest.json", "weights.safetensors"):
                shutil.copyfile(source / name, bundle / name)
            patch = self._read_bundle(bundle, bundle_id)
            self._verify_parent(bundle, patch, patch["parent_variant_id"])
            self._publish_bundle(bundle, self.patches / bundle_id, bundle_id)
        return {"patch_id": bundle_id, "parent_variant_id": patch["parent_variant_id"]}

    def export_model(self, variant_id, format="transformers_safetensors"):
        """导出完整检查点，原样复制配置与分词器，只替换改动过的分片。"""
        if format != "transformers_safetensors":
            raise ValueError(f"Unsupported export format: {format}")
        sources = self._sources(variant_id)
        root = self.root / "exports" / "models"
        root.mkdir(parents=True, exist_ok=True)
        destination = root / variant_id
        if destination.exists():
            raise FileExistsError(f"Checkpoint has already been exported: {variant_id}")
        changed_shards = defaultdict(list)
        for name in sources:
            changed_shards[self.store.tensor_info(name)["shard"]].append(name)

        def skip_rewritten_shards(directory, names):
            omitted = {".cache", ".git"}
            if Path(directory) == self.store.model_dir:
                omitted.update(changed_shards)
            return omitted.intersection(names)

        with TemporaryDirectory(dir=root) as temporary:
            folder = Path(temporary) / "checkpoint"
            shutil.copytree(self.store.model_dir, folder, ignore=skip_rewritten_shards)
            for shard, names in changed_shards.items():
                # Safetensors 保留分片里的所有原始张量及元数据。
                source = self.store._source_path(names[0])
                with safe_open(str(source), framework="pt", device="cpu") as stream:
                    weights = {name: stream.get_tensor(name) for name in stream.keys()}
                    for name in names:
                        weights[name] = self._native(name, sources)
                    save_file(weights, str(folder / shard), metadata=stream.metadata())
            # 重验导出的张量索引，并记录导出检查点自己的身份。
            exported = WeightStore(folder, Path(temporary) / "verify")
            checkpoint = exported.snapshot(with_edits=False)
            with self._publish_lock():
                if destination.exists():
                    raise FileExistsError(f"Checkpoint has already been exported: {variant_id}")
                folder.rename(destination)
        return {"variant_id": variant_id, "format": format,
                "checkpoint_path": f"exports/models/{variant_id}", "base_id": checkpoint["base_id"]}
