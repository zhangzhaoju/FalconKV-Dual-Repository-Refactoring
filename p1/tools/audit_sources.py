# SPDX-License-Identifier: Apache-2.0
"""Audit P1 snapshot conservation without importing/building the frameworks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

import tomllib


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(workspace: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    report = {
        "scope": "source_only_not_build_or_runtime",
        "repositories": [],
        "errors": [],
    }
    for item in manifest["imports"]:
        root = workspace / item["repository"]
        changed, missing, unchanged, pending = [], [], 0, []
        for record in item["files"]:
            relative = record["destination"]
            path = root / relative
            if record["mode"] == "160000":
                pending.append(
                    {
                        "path": relative,
                        "commit": record["git_object"],
                        "payload_present": path.is_dir() and any(path.iterdir()),
                    }
                )
                continue
            if not path.is_file():
                missing.append(relative)
            elif sha256(path) != record["sha256_before_adaptation"]:
                changed.append(relative)
            else:
                unchanged += 1
        allowed_build = {"ascend/csrc/build_aclnn.sh", "ascend/CMakeLists.txt"}
        unexpected = [
            path
            for path in changed
            if path not in allowed_build
            and not path.startswith("ascend/tests/standalone/")
        ]
        if missing or unexpected:
            report["errors"].append(
                {
                    "repository": item["repository"],
                    "missing": missing,
                    "unexpected_changes": unexpected,
                }
            )
        primary = "vllm" if item["repository"] == "vllm" else "lmcache"
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())
        if pyproject["project"]["name"] != primary:
            report["errors"].append(f"Wrong distribution at {root}")
        if (root / "ascend/setup.py").exists() or (
            root / "ascend/pyproject.toml"
        ).exists():
            report["errors"].append(f"Second active packaging entry at {root}")
        parsed = 0
        for directory in (root / primary, root / "ascend" / (primary + "_ascend")):
            for path in directory.rglob("*.py"):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                parsed += 1
        # Worktree audit only. Ordinary delivery archives have no Git dependency.
        base_changes = None
        if (root / ".git").exists():
            base_changes = subprocess.check_output(
                ["git", "-C", str(root), "diff", "HEAD", "--name-only", "--", primary],
                text=True,
            ).splitlines()
            if base_changes:
                report["errors"].append({"base_production_changes": base_changes})
        report["repositories"].append(
            {
                "repository": item["repository"],
                "donor_commit": item["donor_commit"],
                "unchanged_donor_files": unchanged,
                "intentional_adaptations": changed,
                "missing_donor_files": missing,
                "base_production_changes": base_changes,
                "parsed_runtime_python_files": parsed,
                "submodules": pending,
                "distribution_version": pyproject["project"]["version"],
            }
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.workspace.resolve(), args.manifest.resolve())
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(report["errors"]))
