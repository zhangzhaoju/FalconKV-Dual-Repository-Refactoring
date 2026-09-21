# SPDX-License-Identifier: Apache-2.0
"""Import the approved Ascend snapshots into isolated P1 worktrees.

Mechanical source import only: no framework imports, builds, downloads or
installs. Original checkouts are never changed. Refuse dirty inputs and existing
destinations. Submodule payloads must be supplied separately by intranet CI.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

EXPECTED = {
    "vllm": "ded5ce2a5388e1abb799d9b099849f3bd4a72360",
    "vllm-ascend": "d22f0b7cffde1b6ddb87cb44368e46193e811cc9",
    "LMCache": "802e4167afa75c1601b9f9d8619672a686b416bd",
    "LMCache-Ascend": "d959c7a9640681e414dc9dbca644627af67a51e8",
}
PAIRS = {"vllm": "vllm-ascend", "LMCache": "LMCache-Ascend"}
BRANCH = "p1/ascend-unified"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def prepare(workspace: Path, output: Path, evidence: Path) -> None:
    for name, commit in EXPECTED.items():
        repo = workspace / name
        if git(repo, "rev-parse", "HEAD") != commit:
            raise ValueError(f"Unexpected HEAD: {name}; review the new input first")
        if git(repo, "status", "--porcelain"):
            raise ValueError(f"Dirty input: {name}; preserve/review changes first")
        if name in PAIRS:
            found = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "show-ref",
                    "--verify",
                    "--quiet",
                    f"refs/heads/{BRANCH}",
                ],
                check=False,
            )
            if found.returncode != 1:
                raise ValueError(f"Branch already exists or cannot be checked: {name}")
    if output.exists() or evidence.exists():
        raise FileExistsError("Output and evidence directories must both be new")
    output.mkdir(parents=True)
    evidence.mkdir(parents=True)
    sources = []
    for name, commit in EXPECTED.items():
        repo = workspace / name
        bundle = evidence / f"{name}.bundle"
        git(repo, "bundle", "create", str(bundle), "HEAD")
        git(repo, "bundle", "verify", str(bundle))
        with bundle.open("rb") as stream:
            bundle_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        sources.append(
            {
                "repository": name,
                "commit": commit,
                "tree": git(repo, "rev-parse", "HEAD^{tree}"),
                "bundle": bundle.name,
                "bundle_sha256": bundle_sha256,
            }
        )
    imports = []
    for base, donor in PAIRS.items():
        target = output / base
        git(
            workspace / base,
            "worktree",
            "add",
            "-b",
            BRANCH,
            str(target),
            EXPECTED[base],
        )
        destination = target / "ascend"
        destination.mkdir()
        archive = subprocess.check_output(
            ["git", "-C", str(workspace / donor), "archive", EXPECTED[donor]]
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(destination, filter="data")
        # Preserve upstream packaging as reference, not a second build entry.
        upstream = destination / "upstream-build"
        upstream.mkdir()
        for name in ("setup.py", "pyproject.toml", ".gitmodules"):
            source = destination / name
            if source.exists():
                source.rename(upstream / (name + ".txt"))
        tree = subprocess.check_output(
            [
                "git",
                "-C",
                str(workspace / donor),
                "ls-tree",
                "-r",
                "-z",
                EXPECTED[donor],
            ]
        )
        records = []
        for entry in tree.split(b"\0"):
            if not entry:
                continue
            header, raw_path = entry.split(b"\t", 1)
            mode, kind, object_id = header.decode().split()
            relative = raw_path.decode()
            mapped = f"ascend/{relative}"
            if relative in ("setup.py", "pyproject.toml", ".gitmodules"):
                mapped = f"ascend/upstream-build/{relative}.txt"
            record = {
                "source": relative,
                "destination": mapped,
                "git_object": object_id,
                "mode": mode,
            }
            if kind == "commit":
                original_modules = workspace / donor / ".gitmodules"
                module_name = git(
                    workspace / donor,
                    "config",
                    "-f",
                    str(original_modules),
                    "--get-regexp",
                    r"^submodule\..*\.path$",
                )
                key = next(
                    line.split()[0][:-5]
                    for line in module_name.splitlines()
                    if line.split(maxsplit=1)[1] == relative
                )
                url = git(
                    workspace / donor,
                    "config",
                    "-f",
                    str(original_modules),
                    "--get",
                    key + ".url",
                )
                git(
                    target,
                    "config",
                    "-f",
                    str(target / ".gitmodules"),
                    f"submodule.{relative}.path",
                    mapped,
                )
                git(
                    target,
                    "config",
                    "-f",
                    str(target / ".gitmodules"),
                    f"submodule.{relative}.url",
                    url,
                )
                git(
                    target,
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"160000,{object_id},{mapped}",
                )
                record["payload"] = "pending_intranet_materialization"
            elif (target / mapped).is_file():
                with (target / mapped).open("rb") as source:
                    record["sha256_before_adaptation"] = hashlib.file_digest(
                        source, "sha256"
                    ).hexdigest()
            records.append(record)
        imports.append(
            {
                "repository": base,
                "donor": donor,
                "donor_commit": EXPECTED[donor],
                "files": records,
            }
        )
    (evidence / "source-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "P1",
                "branch": BRANCH,
                "sources": sources,
                "imports": imports,
                "runtime_validation": "delegated_to_intranet_developers_pending_archive",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.workspace.resolve(), args.output.resolve(), args.evidence.resolve())
