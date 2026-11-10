"""Register forward results and read small data slices by artifact ID."""

from itertools import islice
import json
from math import prod
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np


_GENERATION_FIELDS = ("prompt_last", "answer_mean")


def observation_axes(name):
    """Label each axis according to the existing forward backend output."""
    if name in ("lengths", "predicted_label_ids"):
        return ["sample"]
    if name in ("class_logits", "class_probabilities"):
        return ["sample", "label"]
    if name.startswith("attention_"):
        return ["sample", "head", "query_token", "key_token"]
    if name.startswith("topk_"):
        return ["sample", "token", "rank"]
    if name.startswith(("hidden_", "mlp_")):
        return ["sample", "token", "channel"]
    return ["sample", "token"]


class ArtifactStore:
    """Read published arrays and generation records in the run directory."""

    def __init__(self, root):
        self.root = Path(root).resolve()

    def references(self, run_id):
        """List published artifacts. The manifest is not yet available while the run is in progress."""
        manifest = self.root / run_id / "artifacts.json"
        if not manifest.is_file():
            return {}
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        return {name: f"{run_id}/{name}" for name, entry in entries.items() if "member" not in entry}

    def publish_forward(self, result, sites):
        """Reuse files saved by ForwardService and add an artifact manifest."""
        folder = Path(result["artifact_dir"]).resolve()
        if folder.parent != self.root:
            raise ValueError("Forward result belongs to a different workspace")
        entries = {}
        for name, info in result.get("files", {}).items():
            site = sites.get(name)
            axes = observation_axes(name) if site is None else site["axes"]
            entries[name] = dict(info, kind="array", axes=axes)
            if site is not None:
                entries[name]["site_id"] = site["site_id"]
        public_names = list(entries)
        for name, filename in result.get("generation_files", {}).items():
            entries[name] = {"kind": "records", "file": filename}
            public_names.append(name)
            if sites:
                self._register_generation_arrays(folder / filename, name, entries, sites)
        staging = folder / "artifacts.tmp"
        staging.write_text(json.dumps(entries, allow_nan=False), encoding="utf-8")
        staging.replace(folder / "artifacts.json")
        return {name: dict({key: value for key, value in entries[name].items() if key != "file"},
                           artifact_id=f"{folder.name}/{name}") for name in public_names}

    @staticmethod
    def _register_generation_arrays(path, name, entries, sites):
        """Record generation replay arrays and return artifact IDs by sample page."""
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                for key, item in json.loads(line).get("activations", {}).items():
                    site = sites[key]
                    for member in _GENERATION_FIELDS:
                        array_name = f"{name}_{index}_{key}_{member}"
                        entries[array_name] = {"kind": "array", "file": item["file"], "member": member,
                                               "shape": [site["width"]], "dtype": "float32",
                                               "axes": ["channel"], "site_id": site["site_id"]}

    def read(self, artifact_id, region=None, offset=None, limit=None):
        """Check artifact ownership, then read an array slice or a page of records."""
        run_id, name, entry, path = self._resolve(artifact_id)
        if entry["kind"] == "plot":
            if region is not None or offset is not None or limit is not None:
                raise ValueError("Plot artifacts have no numeric slices or record pages")
            return {"artifact_id": artifact_id, "run_id": run_id, "kind": "plot", "format": entry["format"],
                    "source_artifact_id": entry["source_artifact_id"], "bytes": path.stat().st_size,
                    "plot_path": str(path.relative_to(self.root.parent))}
        if entry["kind"] == "array":
            if offset is not None or limit is not None:
                raise ValueError("Use region for array artifacts")
            return {"artifact_id": artifact_id, "run_id": run_id, **self._read_array(path, entry, region)}
        if region is not None:
            raise ValueError("Use offset and limit for record artifacts")
        start, count = offset or 0, limit or 8
        with path.open(encoding="utf-8") as stream:
            lines = list(islice(stream, start, start + count))
        if sum(len(line.encode("utf-8")) for line in lines) > 65_536:
            raise ValueError("Record preview exceeds 64 KiB")
        records = []
        for index, line in enumerate(lines, start):
            record = json.loads(line)
            for key, item in record.get("activations", {}).items():
                item.pop("file")
                item.update({member: f"{run_id}/{name}_{index}_{key}_{member}"
                             for member in _GENERATION_FIELDS})
            records.append(record)
        return {"artifact_id": artifact_id, "run_id": run_id, "kind": "records",
                "offset": start, "records": records}

    def _resolve(self, artifact_id):
        if not re.fullmatch(r"(?:request_[0-9]+|analysis_[0-9a-f]{32})/[a-zA-Z0-9_]+", artifact_id):
            raise ValueError("Invalid artifact ID")
        run_id, name = artifact_id.split("/")
        folder = self.root / run_id
        if folder.is_symlink():
            raise ValueError("Invalid artifact directory")
        entry = json.loads((folder / "artifacts.json").read_text(encoding="utf-8"))[name]
        path = (folder / entry["file"]).resolve()
        if path.parent != folder:
            raise ValueError("Artifact file is outside its run")
        return run_id, name, entry, path

    def array(self, artifact_id):
        """Let algorithms within worker processes use registered arrays, eliminating the need for JSON conversion."""
        run_id, _, entry, path = self._resolve(artifact_id)
        if entry["kind"] != "array":
            raise ValueError("An array artifact is required")
        return self._load_array(path, entry), {**entry, "run_id": run_id}

    def save_array(self, name, array, axes, *, run_id=None, **metadata):
        """Publish the analysis artifact and axis descriptions for subsequent tools to read by ID."""
        run_id = run_id or "analysis_" + uuid4().hex
        entry = {"file": f"{name}.npy", "kind": "array", "shape": list(array.shape),
                 "dtype": str(array.dtype), "axes": axes, **metadata}
        with TemporaryDirectory(dir=self.root) as temporary:
            folder = Path(temporary) / "array"
            folder.mkdir()
            np.save(folder / entry["file"], array, allow_pickle=False)
            (folder / "artifacts.json").write_text(json.dumps({name: entry}), encoding="utf-8")
            folder.rename(self.root / run_id)
        return f"{run_id}/{name}"

    def save_records(self, name, records):
        """Save itemized reports as paginated JSONL artifacts."""
        run_id = "analysis_" + uuid4().hex
        filename = f"{name}.jsonl"
        with TemporaryDirectory(dir=self.root) as temporary:
            folder = Path(temporary) / "records"
            folder.mkdir()
            with (folder / filename).open("w", encoding="utf-8") as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            (folder / "artifacts.json").write_text(json.dumps({name: {"file": filename, "kind": "records"}}),
                                                   encoding="utf-8")
            folder.rename(self.root / run_id)
        return f"{run_id}/{name}"

    def save_plot(self, figure, format, source_artifact_id):
        """Save the association between the plot file and its numerical artifact."""
        run_id = "analysis_" + uuid4().hex
        filename = f"plot.{format}"
        entry = {"file": filename, "kind": "plot", "format": format, "source_artifact_id": source_artifact_id}
        with TemporaryDirectory(dir=self.root) as temporary:
            folder = Path(temporary) / "plot"
            folder.mkdir()
            figure.savefig(folder / filename, format=format, dpi=120)
            (folder / "artifacts.json").write_text(json.dumps({"plot": entry}), encoding="utf-8")
            folder.rename(self.root / run_id)
        return self.read(f"{run_id}/plot")

    @staticmethod
    def _load_array(path, entry):
        if "member" in entry:
            with np.load(path, allow_pickle=False) as archive:
                array = archive[entry["member"]]
        else:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
            raise ValueError("Artifact shape or precision differs from its manifest")
        return array

    def _read_array(self, path, entry, region):
        array = self._load_array(path, entry)
        bounds = [[0, length] for length in array.shape] if region is None else region
        if len(bounds) != array.ndim or any(not 0 <= start < stop <= length
                                            for (start, stop), length in zip(bounds, array.shape)):
            raise ValueError("Region must specify a valid half-open interval on every axis")
        if prod(stop - start for start, stop in bounds) > 4096:
            raise ValueError("Select a region of at most 4096 values")
        selected = array[tuple(slice(start, stop) for start, stop in bounds)]
        # Raw observations mark padding positions with NaN; JSON represents them as null.
        values = np.where(np.isfinite(selected), selected.astype(object), None).tolist()
        fields = ("shape", "axes", "dtype", "site_id", "method", "evidence_type", "source_run_id", "variant_id",
                  "source_artifact_ids", "through_layer", "sample_index", "token_positions", "query_positions", "raw_flow",
                  "encoder_id", "feature_ids", "source_workspaces", "dictionary_sha256", "train_indices",
                  "evaluation_indices", "token_position", "cross_covariance_norms", "site_ids", "fitting", "estimator_names")
        metadata = {key: entry[key] for key in fields if key in entry}
        return {"kind": "array", **metadata, "region": bounds, "values": values}
