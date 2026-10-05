#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare an offline image context; do not build, pull, push or run an image."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from qualify import DESIGN, digest, evidence_path, read, save


def prepare(
    pair_path: Path, artifacts_path: Path, base_image: str, output: Path
) -> dict:
    """Copy exact checked wheels and bind a digest-pinned prepared base image."""
    if not re.fullmatch(r"[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}", base_image):
        raise ValueError(
            "Use an explicitly approved base image repository@sha256:digest"
        )
    pair, artifacts = read(pair_path), read(artifacts_path)
    if artifacts.get("passed") is not True or artifacts.get("pair_sha256") != digest(
        pair_path
    ):
        raise ValueError("Artifact report is not a successful build of this pair")
    if set(artifacts.get("repositories", {})) != {"vllm", "LMCache"}:
        raise ValueError("Expected both repository artifacts")
    selected = {}
    for name, rows in artifacts["repositories"].items():
        candidates = [
            row
            for row in rows
            if Path(row["path"]).parts[0] == f"wheel-{name}"
            and row["path"].endswith(".whl")
        ]
        if len(candidates) != 1:
            raise ValueError(f"{name}: select exactly one regular candidate wheel")
        path = evidence_path(artifacts_path.parent, candidates[0])
        package = "lmcache" if name == "LMCache" else "vllm"
        version = pair["repositories"][name]["version"]
        with zipfile.ZipFile(path) as wheel:
            metas = [
                entry
                for entry in wheel.namelist()
                if entry.endswith(".dist-info/METADATA")
            ]
            if len(metas) != 1 or wheel.testzip():
                raise ValueError("Invalid wheel")
            meta = BytesParser().parsebytes(wheel.read(metas[0]))
            if meta["Name"].lower() != package or meta["Version"] != version:
                raise ValueError("Wheel identity differs from pair")
        selected[package] = (path, version)
    output.mkdir(parents=True, exist_ok=False)
    (output / "wheels").mkdir()
    wheels = {}
    for package, (path, version) in selected.items():
        shutil.copyfile(path, output / "wheels" / path.name)
        wheels[package] = {
            "filename": path.name,
            "version": version,
            "sha256": digest(path),
        }
    for name in ("Dockerfile", "install_image.py"):
        shutil.copyfile(DESIGN / "p5/image" / name, output / name)
    shutil.copyfile(
        DESIGN / "p1/tools/check_pip_dependencies.py",
        output / "check_pip_dependencies.py",
    )
    report = {
        "scope": "image_inputs_not_built_image_or_runtime",
        "base_image": base_image,
        "pair_sha256": digest(pair_path),
        "artifact_manifest_sha256": digest(artifacts_path),
        "wheels": wheels,
        "image_built": False,
    }
    save(output / "image-input.json", report)
    return report


def main() -> int:
    """Prepare the exact context only; a human controls the intranet image build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.pair, args.artifacts, args.base_image, args.output)
        print(f"Prepared {args.output}; no image operation performed")
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
