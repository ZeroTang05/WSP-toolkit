"""从固定版本的 SST-2 训练划分提取分类工具示例输入。"""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


def main(output):
    """保留真实句子、标签、原行号和来源摘要。"""
    repo, revision = "stanfordnlp/sst2", "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
    path = Path(hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=revision))
    rows = pq.read_table(path).slice(0, 512).to_pylist()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"id": f"sst2:train:{index}", "sentence": row["sentence"], "label": row["label"],
                                     "source_split": "train"}, ensure_ascii=False) + "\n")
    provenance = {"repo_id": repo, "revision": revision, "split": "train", "first_rows": len(rows),
                  "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "jsonl_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args().output)
