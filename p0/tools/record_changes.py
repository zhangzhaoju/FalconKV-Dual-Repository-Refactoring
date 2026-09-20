# SPDX-License-Identifier: Apache-2.0
"""Record the narrowly scoped P0 fixes and local test evidence for handoff.

Fail if HEADs changed or framework modifications extend beyond the P0 allowlist.
No commits, installs, checkout changes or network operations are performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from source_inventory import digest_file, git, git_text, write_json


CHANGES = {
    "vllm": {"pyproject.toml", "requirements/build.txt"},
    "vllm-ascend": {"tests/standalone/test_connector_control_idle.py"},
    "LMCache": {"pyproject.toml"},
    "LMCache-Ascend": {
        "pyproject.toml",
        "tests/standalone/test_cold_abort_completion.py",
        "tests/standalone/test_checkpoint_page_keys.py",
        "tests/standalone/test_checkpoint_miss_retry.py",
        "tests/standalone/test_dense_checkpoint_partial_pages.py",
        "tests/standalone/test_checkpoint_dispatch.py",
        "tests/standalone/test_local_checkpoint_restore.py",
        "tests/standalone/test_preemption_checkpoint.py",
        "tests/standalone/test_checkpoint_initialization.py",
        "tests/standalone/test_checkpoint_prefix_agreement.py",
    },
}


def main() -> None:
    """Emit reviewed patch boundaries and evidence hashes into a new directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads((args.baseline / "source-manifest.json").read_text())
    by_repo = {r["repository"]: r for r in baseline["repos"]}
    patches = []
    for name, allowed in CHANGES.items():
        repo = args.workspace.resolve() / name
        if git_text(repo, "rev-parse", "HEAD") != by_repo[name]["head"]:
            raise RuntimeError(
                f"{name}: HEAD changed; review a new baseline before export"
            )
        changed = set(git_text(repo, "diff", "--name-only", "HEAD").splitlines())
        untracked = git_text(repo, "ls-files", "--others", "--exclude-standard")
        if changed - allowed or untracked:
            raise RuntimeError(
                f"{name}: additional user changes detected; not packaging them"
            )
        git(repo, "diff", "--check", "HEAD")
        diff = git(repo, "diff", "--binary", "HEAD", "--", *sorted(allowed))
        patches.append((name, changed, diff))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, changed, diff in patches:
        path = args.output / f"{name}.patch"
        path.write_bytes(diff)
        if diff:
            git(
                args.workspace / name,
                "apply",
                "--check",
                "--reverse",
                str(path.resolve()),
            )
        records.append(
            {
                "repository": name,
                "base_commit": by_repo[name]["head"],
                "patch": path.name,
                "patch_sha256": hashlib.sha256(diff).hexdigest(),
                "changed_files": [
                    {"path": p, "sha256_after": digest_file(args.workspace / name / p)}
                    for p in sorted(changed)
                ],
                "reverse_apply_check": "passed",
                "intranet_build_verified": False,
            }
        )
    host_results = []
    for path in sorted(args.results.glob("*/summary.json")):
        value = json.loads(path.read_text())
        host_results.append(
            {
                "batch": path.parent.name,
                "summary_sha256": digest_file(path),
                "summary": value,
            }
        )
    write_json(
        args.output / "change-manifest.json",
        {
            "schema_version": 1,
            "phase": "P0",
            "commits_created": False,
            "patches": records,
            "compatibility": "metadata_aligned_but_runtime_and_ABI_pending",
            "host_results": host_results,
        },
    )
    print(
        json.dumps(
            {
                "patches": len(records),
                "changed_files": sum(len(r["changed_files"]) for r in records),
                "host_result_batches": len(host_results),
            }
        )
    )


if __name__ == "__main__":
    main()
