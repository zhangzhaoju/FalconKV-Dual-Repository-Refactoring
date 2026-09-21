# SPDX-License-Identifier: Apache-2.0
"""Copy verified intranet Git submodules into a new P1 source delivery.

Read-only for the original repositories. No downloads, imports or builds.
The destination payloads must be absent or empty; existing data is not replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

MATERIALS = (
    (
        "vllm",
        "vllm-ascend",
        "csrc/third_party/catlass",
        "716fd7baa7fb7f6cac0488bb628fd1dd0e875641",
    ),
    (
        "LMCache",
        "LMCache-Ascend",
        "third_party/kvcache-ops",
        "9f18d2339bc58a43429f7d5bdaef1628c820eff5",
    ),
)


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def file_hash(path: Path) -> str:
    if path.is_symlink():
        return hashlib.sha256(str(path.readlink()).encode()).hexdigest()
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(path: Path) -> dict:
    return {
        str(item.relative_to(path)): {
            "sha256": file_hash(item),
            "symlink": str(item.readlink()) if item.is_symlink() else None,
        }
        for item in sorted(path.rglob("*"))
        if item.is_symlink() or item.is_file()
    }


def materialize(sources: Path, destination: Path) -> None:
    plans = []
    for base, donor, relative, commit in MATERIALS:
        source = sources / donor / relative
        target = destination / base / "ascend" / relative
        manifest = destination / base / "ascend/submodule-materials.json"
        if not (destination / base / "p1_build.py").is_file():
            raise ValueError(f"Not a P1 destination: {destination / base}")
        if (
            target.is_symlink()
            or target.is_file()
            or (target.exists() and any(target.iterdir()))
        ):
            raise FileExistsError(f"Refusing to overwrite material: {target}")
        if manifest.exists():
            raise FileExistsError(f"Use a fresh source delivery: {manifest}")
        if (
            Path(git(source, "rev-parse", "--show-toplevel")).resolve()
            != source.resolve()
        ):
            raise ValueError(f"Submodule is not initialized: {source}")
        if git(source, "rev-parse", "HEAD") != commit or git(
            source, "status", "--porcelain"
        ):
            raise ValueError(f"Expected clean pinned submodule {source} @ {commit}")
        if any(
            line.startswith("160000 ")
            for line in git(source, "ls-tree", "-r", "HEAD").splitlines()
        ):
            raise ValueError(f"Nested submodules need explicit review: {source}")
        plans.append((source, target, manifest, relative, commit))
    for source, target, manifest, relative, commit in plans:
        archive = subprocess.check_output(["git", "-C", str(source), "archive", commit])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="p1-material-", dir=target.parent
        ) as staging:
            payload = Path(staging) / "payload"
            payload.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
                stream.extractall(payload, filter="data")
            files = inventory(payload)
            if not files:
                raise ValueError(f"Empty material: {source}")
            payload.rename(target)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "path": relative,
                    "commit": commit,
                    "archive_sha256": hashlib.sha256(archive).hexdigest(),
                    "files": files,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {
                    "repository": target.parents[len(Path(relative).parts)],
                    "commit": commit,
                    "files": len(files),
                },
                default=str,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    materialize(args.sources.resolve(), args.destination.resolve())
