#!/usr/bin/env python3
"""Export source from a Git commit or update checksums from the staging area."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "SOURCE_MANIFEST.json"


def git_bytes(*arguments: str) -> bytes:
    """Invoke Git directly and stop immediately if the command fails."""
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def update_manifest() -> None:
    """Hash staged content while keeping unstaged experimental changes separate."""
    conflicts = git_bytes("diff", "--name-only", "--diff-filter=U", "-z")
    if conflicts:
        raise ValueError("Resolve merge conflicts in the staging area before updating the manifest.")

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
    """Export a commit and record its revision, file count, and archive checksum."""
    commit = git_bytes(
        "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
    ).decode("ascii").strip()
    output = output.resolve()
    metadata = output.with_name(output.name + ".json")
    if output.exists() or metadata.exists():
        raise FileExistsError(f"Output already exists; choose a new filename: {output}")
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
    """Parse the mutually exclusive source export and manifest update actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="Destination path for the source tar.gz archive")
    action.add_argument("--update-manifest", action="store_true", help="Update the staged file manifest")
    parser.add_argument("--revision", help="Git revision to export; defaults to HEAD")
    arguments = parser.parse_args()
    if arguments.update_manifest:
        if arguments.revision is not None:
            parser.error("--revision is only available for source exports.")
        update_manifest()
    else:
        export_source(arguments.output, arguments.revision or "HEAD")


if __name__ == "__main__":
    main()
