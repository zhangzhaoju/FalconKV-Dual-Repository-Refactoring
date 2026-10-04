#!/usr/bin/env python3
"""Archive the native-layout source/host gates without building native artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--test-python", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New result directory"
    )
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(
        (workspace / "design/native-layout/baseline/frozen-input.json").read_text()
    )
    result = {
        "scope": "source_and_selected_host_contracts_not_build_ABI_or_NPU",
        "repositories": {},
        "checks": [],
        "errors": [],
    }
    for name, baseline in frozen["repositories"].items():
        root = workspace / "p1-repos" / name

        def git(*command):
            return subprocess.check_output(
                ["git", "-C", str(root), *command], text=True
            ).strip()

        refs = {
            **baseline["refs_before_freeze"],
            "refs/tags/" + frozen["frozen_tag"]: baseline["tag_object"],
        }
        for ref, expected in refs.items():
            if git("rev-parse", ref) != expected:
                result["errors"].append(f"Frozen ref changed: {name} {ref}")
        manifest = json.loads((root / "docs/design/layout-migration.json").read_text())
        result["repositories"][name] = {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status": git("status", "--porcelain"),
            "version": manifest["layout_version"],
            "frozen_commit": baseline["commit"],
            "preserved_refs": refs,
            "migration_manifest_sha256": hashlib.sha256(
                (root / "docs/design/layout-migration.json").read_bytes()
            ).hexdigest(),
        }
        commands = [
            (
                "host",
                [
                    str(args.test_python.absolute()),
                    "-B",
                    "tools/run_layout_host_checks.py",
                ],
            ),
            ("whitespace", ["git", "diff", "--check", "HEAD"]),
        ]
        if name == "LMCache":
            commands.append(
                (
                    "sphinx",
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "sphinx",
                        "-E",
                        "-W",
                        "--keep-going",
                        "-b",
                        "html",
                        "docs/source",
                        str(output / "sphinx-html"),
                    ],
                )
            )
        for label, command in commands:
            log = output / f"{name}-{label}.log"
            with log.open("x") as stream:
                run = subprocess.run(
                    command,
                    cwd=root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    check=False,
                )
            result["checks"].append(
                {
                    "repository": name,
                    "check": label,
                    "command": command,
                    "returncode": run.returncode,
                    "log": log.name,
                    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            )
            if run.returncode:
                result["errors"].append(f"{name} {label}: see {log.name}")
            print(
                f"{'PASS' if run.returncode == 0 else 'FAIL'}: {name} {label}",
                flush=True,
            )
    result["passed"] = not result["errors"]
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
