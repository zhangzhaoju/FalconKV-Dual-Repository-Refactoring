#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Back up committed HEAD histories and verify local restores; never retire sources.

No network, branch changes, service actions or deletion. Git bundles do not back
up uncommitted/ignored files, LFS object payloads, models, wheels or environments.
Missing submodule payloads are explicit blockers, not an implicit complete backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    """Return streaming SHA256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, data: dict) -> None:
    """Write a new immutable report."""
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def git(root: Path, *args: str) -> str:
    """Run Git locally, retaining failure details for the caller."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def child_path(root: Path, relative: str) -> Path:
    """Reject absolute paths, traversal and symlinks outside the archive root."""
    path = Path(relative)
    target = (root / path).resolve()
    if (
        path.is_absolute()
        or ".." in path.parts
        or not target.is_relative_to(root.resolve())
    ):
        raise ValueError("Backup member escapes root")
    return target


def backup_one(
    source: Path, output: Path, label: str, entries: list, errors: list
) -> None:
    """Bundle one committed HEAD and recurse through locally present pinned gitlinks."""
    if Path(git(source, "rev-parse", "--show-toplevel")).resolve() != source.resolve():
        raise ValueError(f"{label}: not a repository root")
    if git(source, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(
            f"{label}: dirty/untracked state; preserve it explicitly before backup"
        )
    commit = git(source, "rev-parse", "HEAD")
    tree = git(source, "rev-parse", "HEAD^{tree}")
    bundle = output / f"{len(entries):03d}.bundle"
    git(source, "bundle", "create", str(bundle), "HEAD")
    ignored = git(
        source, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
    )
    entry = {
        "label": label,
        "source": str(source),
        "commit": commit,
        "tree": tree,
        "bundle": bundle.name,
        "sha256": sha256(bundle),
        "ignored_paths_not_backed_up": len([p for p in ignored.split("\0") if p]),
        "submodules": [],
    }
    entries.append(entry)
    for row in git(source, "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if not row:
            continue
        header, path = row.split("\t", 1)
        mode, _, pin = header.split()
        if mode != "160000":
            continue
        child = child_path(source, path)
        child_label = f"{label}/{path}"
        entry["submodules"].append({"path": path, "commit": pin, "label": child_label})
        try:
            if not (child / ".git").exists() or git(child, "rev-parse", "HEAD") != pin:
                raise ValueError("Missing/uninitialized/mismatched pinned submodule")
            backup_one(child, output, child_label, entries, errors)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"{child_label}: {exc}")
    if git(source, "rev-parse", "HEAD") != commit or git(
        source, "status", "--porcelain", "--untracked-files=all"
    ):
        raise ValueError(f"{label}: source changed during backup")


def create(repos: dict[str, Path], output: Path) -> dict:
    """Create committed-source bundles; preserve partial evidence on any failure."""
    for name, root in repos.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(
                "Repository labels must contain only letters, digits, _ or -"
            )
        if output.resolve().is_relative_to(root.resolve()):
            raise ValueError(
                "Backup destination must be outside every source repository"
            )
    output.mkdir(parents=True, exist_ok=False)
    entries, errors = [], []
    for name, root in repos.items():
        try:
            backup_one(root.resolve(), output, name, entries, errors)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"{name}: {exc}")
    report = {
        "schema_version": 1,
        "scope": "committed_HEAD_history_and_present_submodules_only",
        "requested_repositories": list(repos),
        "entries": entries,
        "errors": errors,
        "passed": bool(entries) and not errors,
        "old_checkouts_retired": False,
        "not_backed_up": [
            "ignored files",
            "uncommitted changes",
            "LFS object payloads",
            "model weights",
            "images/environments",
            "built wheels",
            "external config/data",
        ],
    }
    save(output / "manifest.json", report)
    return report


def verify(archive: Path, output: Path) -> dict:
    """Recreate and verify local worktrees; never copy back to originals."""
    manifest_path = archive / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    errors = list(manifest.get("errors", []))
    entries = manifest.get("entries", [])
    if (
        manifest.get("schema_version") != 1
        or manifest.get("passed") is not True
        or not entries
    ):
        errors.append("Backup is incomplete or has an unknown schema")
    labels = {row["label"]: row for row in entries}
    if len(labels) != len(entries) or not manifest.get("requested_repositories"):
        errors.append("Missing or duplicate repository labels")
    for label in manifest.get("requested_repositories", []):
        if label not in labels:
            errors.append(f"{label}: missing root backup")
    restored = []
    roots = {}
    for index, row in enumerate(entries):
        try:
            bundle = child_path(archive, row["bundle"])
            if sha256(bundle) != row["sha256"]:
                raise ValueError("Bundle checksum mismatch")
            root_name, separator, relative = row["label"].partition("/")
            if separator:
                target = child_path(roots[root_name], relative)
            else:
                target = output / f"repo-{index:03d}"
                roots[root_name] = target
            log_path = output / f"repo-{index:03d}.log"
            commands = [
                ["git", "clone", "--no-checkout", str(bundle), str(target)],
                ["git", "-C", str(target), "bundle", "verify", str(bundle)],
                ["git", "-C", str(target), "fsck", "--full"],
                [
                    "git",
                    "-C",
                    str(target),
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "filter.lfs.required=false",
                    "-c",
                    "filter.lfs.smudge=",
                    "-c",
                    "filter.lfs.process=",
                    "checkout",
                    "--detach",
                    row["commit"],
                ],
            ]
            with log_path.open("x") as stream:
                for command in commands:
                    subprocess.run(
                        command, check=True, stdout=stream, stderr=subprocess.STDOUT
                    )
            if git(target, "rev-parse", "HEAD^{tree}") != row["tree"]:
                raise ValueError("Restored tree differs")
            for child in row["submodules"]:
                if labels.get(child["label"], {}).get("commit") != child["commit"]:
                    raise ValueError(
                        f"Missing/wrong submodule backup: {child['label']}"
                    )
            restored.append(
                {
                    "label": row["label"],
                    "commit": row["commit"],
                    "tree": row["tree"],
                    "path": str(target),
                }
            )
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            errors.append(f"{row.get('label')}: {exc}")
    result = {
        "scope": "source_object_and_worktree_restore_not_model_or_environment_recovery",
        "manifest_sha256": sha256(manifest_path),
        "restored": restored,
        "errors": errors,
        "passed": not errors,
        "production_recovery_verified": False,
        "old_checkouts_may_be_deleted": False,
    }
    save(output / "restore-report.json", result)
    return result


def main() -> int:
    """Only create a new backup or verify into a new directory; no deletion action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument(
        "--repo", action="append", default=[], help="label=/absolute/repository"
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.action == "create":
            repos = {}
            for spec in args.repo:
                name, path = spec.split("=", 1)
                if name in repos or not Path(path).is_absolute():
                    raise ValueError("Use unique labels and absolute paths")
                repos[name] = Path(path)
            if not repos:
                raise ValueError("At least one explicit --repo is required")
            report = create(repos, args.output.resolve())
        else:
            if args.archive is None:
                raise ValueError("--archive required")
            report = verify(args.archive.resolve(), args.output.resolve())
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
