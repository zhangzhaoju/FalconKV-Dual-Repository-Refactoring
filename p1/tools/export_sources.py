# SPDX-License-Identifier: Apache-2.0
"""Export ordinary source tarballs, NOT sdist/wheels; never invoke a backend.

Copies the current P1 source tree including uncommitted changes, excluding Git and
generated outputs. Intranet materialization is required before compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}
GENERATED = {
    "build",
    "dist",
    ".deps",
    "output",
    "ascend/build",
    "ascend/dist",
    "ascend/.deps",
    "ascend/output",
    "ascend/csrc/build",
    "ascend/csrc/output",
}


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(
            part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if any(str(parent) in GENERATED for parent in (relative, *relative.parents)):
            continue
        if path.suffix in {".pyc", ".pyo"} or (path.is_dir() and not path.is_symlink()):
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"External symlink is not a source delivery input: {path}")
        yield path


def export(workspace: Path, output: Path) -> dict:
    if output.is_relative_to(workspace):
        raise ValueError("The delivery output must be outside the source workspace")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "kind": "ordinary_source_archive_not_sdist",
        "submodule_materialization_required": True,
        "repositories": [],
    }
    for name in ("vllm", "LMCache"):
        root = workspace / name
        if not (root / "p1_build.py").is_file():
            raise ValueError(f"Not a P1 source tree: {root}")
        archive = output / f"{name}-p1-source.tar.gz"
        records = []
        with tarfile.open(archive, "w:gz") as stream:
            for path in source_files(root):
                relative = path.relative_to(root)
                stream.add(path, arcname=str(Path(name) / relative), recursive=False)
                if path.is_symlink():
                    digest = hashlib.sha256(str(path.readlink()).encode()).hexdigest()
                else:
                    with path.open("rb") as content:
                        digest = hashlib.file_digest(content, "sha256").hexdigest()
                records.append({"path": str(relative), "sha256": digest})
        with archive.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        report["repositories"].append(
            {
                "repository": name,
                "archive": archive.name,
                "archive_sha256": sha,
                "files": records,
            }
        )
    (output / "delivery-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(
        "".join(
            item["archive_sha256"] + "  " + item["archive"] + "\n"
            for item in report["repositories"]
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.workspace.resolve(), args.output.resolve())
    print(
        json.dumps(
            [
                {k: v for k, v in item.items() if k != "files"}
                for item in result["repositories"]
            ],
            indent=2,
        )
    )
