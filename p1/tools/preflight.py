# SPDX-License-Identifier: Apache-2.0
"""Intranet read-only P1 dependency gate. Does not import torch or initialize NPU.

Checks direct requirements; pip check after installation still checks transitive
metadata. Missing packages are reported, never installed or silently skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import sys
from importlib import metadata
from pathlib import Path

import tomllib
from packaging.requirements import Requirement


def collect(workspace: Path) -> dict:
    checks, errors = [], []
    if sys.version_info[:2] != (3, 11) or platform.machine() != "aarch64":
        errors.append("Candidate requires Python 3.11 / aarch64")
    requirements = {"build>=1.2"}
    for repo in ("vllm", "LMCache"):
        root = workspace / repo
        config = tomllib.loads((root / "pyproject.toml").read_text())
        requirements.update(config["build-system"]["requires"])
        requirements.update(
            line.split("#", 1)[0].strip()
            for line in (root / "requirements/ascend.txt").read_text().splitlines()
            if line.split("#", 1)[0].strip()
        )
    for raw in sorted(requirements):
        req = Requirement(raw)
        if req.marker and not req.marker.evaluate():
            continue
        try:
            installed = metadata.version(req.name)
        except metadata.PackageNotFoundError:
            installed = None
        passed = installed is not None and req.specifier.contains(
            installed, prereleases=True
        )
        checks.append({"requirement": raw, "installed": installed, "passed": passed})
        if not passed:
            errors.append(f"Missing or mismatched: {raw}; installed={installed}")
    for command in ("cmake", "git", "g++", "gcc", "make", "bash"):
        if shutil.which(command) is None:
            errors.append(f"Missing tool: {command}")
    cann = Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent/p1-cann"))
    info = cann / "aarch64-linux/ascend_toolkit_install.info"
    if not info.is_file() or not re.search(
        r"(?m)^version\s*=\s*8\.5\.1\s*$", info.read_text()
    ):
        errors.append("ASCEND_HOME_PATH does not identify the approved CANN 8.5.1")
    return {
        "scope": "direct_metadata_only_not_ABI_or_runtime",
        "python": sys.version,
        "machine": platform.machine(),
        "requirements": checks,
        "errors": errors,
        "passed": not errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.workspace.resolve())
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
