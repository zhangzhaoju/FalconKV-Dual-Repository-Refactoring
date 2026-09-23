# SPDX-License-Identifier: Apache-2.0
"""Audit P1 snapshot conservation without importing/building the frameworks."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

import tomllib


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_updates(
    workspace: Path, manifest_path: Path, updates_path: Path, repositories: set[str]
) -> tuple[dict[str, set[str]], dict]:
    """Permit only exact, recorded updates to a pinned historical manifest."""
    verified = {name: set() for name in repositories}
    result = {"manifest": str(updates_path), "verified_files": [], "errors": []}
    try:
        updates = json.loads(updates_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        result["errors"].append(f"Cannot read updates manifest: {exc}")
        return verified, result
    if (
        not isinstance(updates, dict)
        or updates.get("schema_version") != 1
        or updates.get("base_manifest_sha256") != sha256(manifest_path)
        or not isinstance(updates.get("files"), list)
        or not updates["files"]
    ):
        result["errors"].append("Invalid updates schema or base manifest SHA-256")
        return verified, result
    seen = set()
    for record in updates["files"]:
        if not isinstance(record, dict):
            result["errors"].append("Invalid update record")
            continue
        repo = record.get("repository")
        relative = record.get("destination")
        digest = record.get("sha256_after_integration")
        if not isinstance(repo, str) or repo not in repositories:
            result["errors"].append(f"Unknown update repository: {repo!r}")
            continue
        if (
            not isinstance(relative, str)
            or not relative
            or relative == "."
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or PurePosixPath(relative).as_posix() != relative
            or "\\" in relative
        ):
            result["errors"].append(f"Unsafe update path: {relative!r}")
            continue
        key = (repo, relative)
        if key in seen:
            result["errors"].append(f"Duplicate update: {repo}/{relative}")
            continue
        seen.add(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            result["errors"].append(f"Invalid update SHA-256: {repo}/{relative}")
            continue
        root = (workspace / repo).resolve()
        path = root / relative
        if not path.resolve().is_relative_to(root) or not path.is_file():
            result["errors"].append(f"Missing/unsafe updated file: {repo}/{relative}")
        elif sha256(path) != digest:
            result["errors"].append(f"Updated file hash mismatch: {repo}/{relative}")
        else:
            verified[repo].add(relative)
            result["verified_files"].append(f"{repo}/{relative}")
    return verified, result


def audit(
    workspace: Path, manifest_path: Path, updates_path: Path | None = None
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    report = {
        "scope": "source_only_not_build_or_runtime",
        "repositories": [],
        "errors": [],
    }
    repositories = {item["repository"] for item in manifest["imports"]}
    verified = {name: set() for name in repositories}
    if updates_path is not None:
        verified, updates_report = verify_updates(
            workspace, manifest_path, updates_path, repositories
        )
        report["source_updates"] = updates_report
        report["errors"].extend(updates_report["errors"])
    for item in manifest["imports"]:
        root = workspace / item["repository"]
        approved = verified[item["repository"]]
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
            and path not in approved
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
        # Git checkout audit only. Ordinary delivery archives have no Git dependency.
        base_changes = None
        if (root / ".git").exists():
            base_changes = subprocess.check_output(
                ["git", "-C", str(root), "diff", "HEAD", "--name-only", "--", primary],
                text=True,
            ).splitlines()
            unexpected_base = [path for path in base_changes if path not in approved]
            if unexpected_base:
                report["errors"].append(
                    {
                        "repository": item["repository"],
                        "base_production_changes": unexpected_base,
                    }
                )
        report["repositories"].append(
            {
                "repository": item["repository"],
                "donor_commit": item["donor_commit"],
                "unchanged_donor_files": unchanged,
                "intentional_adaptations": changed,
                "missing_donor_files": missing,
                "base_production_changes": base_changes,
                "verified_source_updates": sorted(approved),
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
    parser.add_argument(
        "--updates",
        type=Path,
        help="Optional exact-hash update manifest; does not replace historical inputs",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(
        args.workspace.resolve(),
        args.manifest.resolve(),
        args.updates.resolve() if args.updates is not None else None,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(bool(report["errors"]))
