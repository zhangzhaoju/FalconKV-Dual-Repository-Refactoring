#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install only the two explicitly hashed wheels in a clean prepared base image."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from importlib.machinery import PathFinder
from pathlib import Path


def main() -> None:
    """Reject contaminated images, preserve foundation packages and check metadata."""
    root = Path(__file__).resolve().parent
    data = json.loads((root / "image-input.json").read_text())
    for name in ("vllm", "lmcache", "vllm-ascend", "lmcache-ascend"):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Base image contains {name}; use a clean foundation image")
    foundation = {
        name: metadata.version(name)
        for name in ("torch", "torch-npu", "transformers", "triton-ascend")
    }
    expected = {
        "torch": "2.9.0",
        "torch-npu": "2.9.0.post2",
        "transformers": "5.2.0",
        "triton-ascend": "3.2.0.dev20260322",
    }
    if any(
        foundation[name].split("+")[0] != version for name, version in expected.items()
    ):
        raise RuntimeError(f"Base image foundation differs: {foundation}")
    wheels = []
    if set(data["wheels"]) != {"vllm", "lmcache"}:
        raise ValueError("Expected exactly two candidate wheels")
    for row in data["wheels"].values():
        if Path(row["filename"]).name != row["filename"]:
            raise ValueError("Wheel name must be a basename")
        path = root / "wheels" / row["filename"]
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError("Changed wheel input")
        wheels.append(str(path))
    for suffix in ("before", "after"):
        if suffix == "after":
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "--disable-pip-version-check",
                    "install",
                    "--no-index",
                    "--no-deps",
                    "--no-build-isolation",
                    *wheels,
                ],
                check=True,
            )
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(root / "check_pip_dependencies.py"),
                "--output",
                str(root / f"pip-{suffix}"),
            ],
            check=True,
        )
    installed = {}
    for name, row in data["wheels"].items():
        dist = metadata.distribution(name)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        spec = PathFinder.find_spec(name)
        if (
            dist.version != row["version"]
            or direct.get("dir_info", {}).get("editable")
            or spec is None
            or spec.origin is None
            or Path(spec.origin).resolve().parent
            != Path(dist.locate_file(name)).resolve()
        ):
            raise RuntimeError(f"Wrong/shadowed installation: {name}")
        installed[name] = {"version": dist.version, "origin": spec.origin}
    if foundation != {name: metadata.version(name) for name in foundation}:
        raise RuntimeError("Foundation packages changed")
    with (root / "installed.json").open("x") as stream:
        json.dump(
            {
                "scope": "image_install_metadata_not_ABI_or_NPU",
                "passed": True,
                "installed": installed,
                "foundation": foundation,
                "pair_sha256": data["pair_sha256"],
            },
            stream,
            indent=2,
        )


if __name__ == "__main__":
    main()
