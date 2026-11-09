#!/usr/bin/env python3
"""从 Git 提交导出源码，或根据暂存区更新源码文件校验清单。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "SOURCE_MANIFEST.json"


def git_bytes(*arguments: str) -> bytes:
    """直接调用 Git；命令失败时立即输出错误并停止。"""
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def update_manifest() -> None:
    """校验暂存区内容，使未暂存的实验修改保持独立。"""
    conflicts = git_bytes("diff", "--name-only", "--diff-filter=U", "-z")
    if conflicts:
        raise ValueError("暂存区存在未解决的合并冲突，请先处理冲突。")

    manifest = json.loads(git_bytes("show", f":{MANIFEST}"))
    paths = git_bytes("ls-files", "-z").decode("utf-8").split("\0")
    manifest["files_sha256"] = {
        name: hashlib.sha256(git_bytes("show", f":{name}")).hexdigest()
        for name in sorted(paths)
        if name and name != MANIFEST
    }
    (ROOT / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": MANIFEST, "files": len(manifest["files_sha256"])}))


def export_source(output: Path, revision: str) -> None:
    """导出指定提交，并保存提交号、文件数量和压缩包摘要。"""
    commit = git_bytes(
        "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
    ).decode("ascii").strip()
    output = output.resolve()
    metadata = output.with_name(output.name + ".json")
    if output.exists() or metadata.exists():
        raise FileExistsError(f"输出已存在，请使用新的文件名：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        subprocess.run(
            ["git", "-C", str(ROOT), "archive", "--format=tar.gz",
             "--prefix=WSP-toolkit/", commit],
            check=True,
            stdout=stream,
        )

    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with tarfile.open(output, "r:gz") as archive:
        file_count = sum(not member.isdir() for member in archive)
    record = {
        "revision": commit,
        "archive": output.name,
        "sha256": digest.hexdigest(),
        "file_count": file_count,
    }
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False))


def main() -> None:
    """解析导出与清单更新参数；两个操作分别执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="源码 tar.gz 压缩包的保存路径")
    action.add_argument("--update-manifest", action="store_true", help="更新暂存文件清单")
    parser.add_argument("--revision", help="导出提交，默认 HEAD")
    arguments = parser.parse_args()
    if arguments.update_manifest:
        if arguments.revision is not None:
            parser.error("--revision 仅用于源码导出。")
        update_manifest()
    else:
        export_source(arguments.output, arguments.revision or "HEAD")


if __name__ == "__main__":
    main()
