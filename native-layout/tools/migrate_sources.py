#!/usr/bin/env python3
"""One-shot, reviewed P4 source relocation; never used by package installation.

Run only on the clean frozen input plus the authorized AGENTS.md amendment.
Git moves preserve executable modes and submodule identities. Every retired
text file is recoverable from its frozen blob recorded in the migration map.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


FROZEN = "p4-frozen-20261004"
KEPT_VLLM_TOOLS = {
    "aisbench.py",
    "inspect_target_sfa_snapshots.py",
    "send_request.py",
    "staged_sfa_graph_smoke.md",
    "staged_sfa_graph_smoke.py",
    "summarize_mtp_decode_offload.py",
    "vllm_bench.py",
}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def relocate(root, apply):
    primary = "vllm" if root.name == "vllm" else "lmcache"
    if git(root, "branch", "--show-current").strip() != b"refactor/native-layout":
        raise RuntimeError("Refusing to modify a non-layout branch")
    if (
        git(root, "rev-parse", "HEAD").strip()
        != git(root, "rev-parse", FROZEN + "^{}").strip()
    ):
        raise RuntimeError("Migration must begin at the frozen input")
    entries = {}
    for line in git(root, "ls-tree", "-rz", FROZEN).split(b"\0"):
        if line:
            meta, path = line.decode().split("\t", 1)
            mode, kind, blob = meta.split()
            entries[path] = {"mode": mode, "kind": kind, "blob": blob}
    records = []
    for path, meta in entries.items():
        if not path.startswith("ascend/"):
            continue
        short = path.removeprefix("ascend/")
        dest = None
        action, reason = (
            "retired",
            "Obsolete donor repository configuration; root configuration is authoritative",
        )
        if short.startswith(
            ("csrc/", "cmake/", "third_party/", "examples/", "benchmarks/")
        ):
            dest, action, reason = (
                short,
                "move",
                "Preserve implementation at its native repository owner",
            )
        elif short == "CMakeLists.txt":
            dest, action, reason = (
                "cmake/npu_extensions.cmake",
                "move",
                "Single root CMake entry includes the native extension targets",
            )
        elif short.startswith("tests/"):
            dest, action, reason = (
                short,
                "move",
                "Preserve the test contract in the unified test tree",
            )
            if short in entries:
                if (
                    meta["blob"] == entries[short]["blob"]
                    or short == "tests/pytest.ini"
                ):
                    action, reason = (
                        "deduplicated",
                        "Identical package marker or equivalent pytest configuration",
                    )
                elif short == "tests/conftest.py":
                    dest, action, reason = (
                        "tests/conftest.py",
                        "merged",
                        "Keep root fixtures; replace bootstrap test patch with native test helpers",
                    )
                else:
                    dest = str(Path(short).with_stem(Path(short).stem + "_npu"))
                    reason = "Keep both distinct test/helper contracts; NPU variant has an explicit suffix"
            if short == "tests/bootstrap.py":
                dest, action, reason = (
                    "tests/v1/utils.py",
                    "merged",
                    "Unified tests package needs no synthetic lmcache_tests alias",
                )
            if (
                primary == "lmcache"
                and short == "tests/v1/cache_controller/__init__.py"
            ):
                dest, action, reason = (
                    short,
                    "deduplicated",
                    "Empty donor package marker is superseded by the root licensed package marker",
                )
            if primary == "lmcache" and short in {
                "tests/v1/multiprocess/test_cache_server.py",
                "tests/v1/multiprocess/test_custom_types.py",
            }:
                dest = (
                    "docs/design/retired-tests/multiprocess/"
                    + Path(short).name
                    + ".txt"
                )
                reason = (
                    "Archive unchanged wrapper for the multiprocess server already "
                    "removed/excluded in P4; not a live test or a new skip"
                )
        elif short.startswith("docs/") and primary == "lmcache":
            dest = (
                "docs/source/_static/lmcache-ascend-logo.png"
                if short.endswith(".png")
                else "docs/design/" + Path(short).name
            )
            action, reason = (
                "move",
                "Preserve technical notes/assets; current user guide is in Sphinx",
            )
        elif (
            short.startswith("tools/")
            and primary == "vllm"
            and Path(short).name in KEPT_VLLM_TOOLS
        ):
            dest, action, reason = (
                short,
                "move",
                "Preserve deployment and diagnostic tooling",
            )
        elif short in {
            "LICENSE",
            "DCO",
            "CODE_OF_CONDUCT.md",
            "tools/check_spdx_header.py",
            "tools/check_repo.sh",
        }:
            dest, action, reason = (
                short,
                "deduplicated",
                "Root legal/development entry is authoritative; file notices remain intact",
            )
        records.append(
            {
                "source": path,
                "destination": dest,
                "action": action,
                "reason": reason,
                **meta,
            }
        )
    destinations = set()
    for row in records:
        if row["action"] == "move":
            dst = row["destination"]
            if dst in destinations or (root / dst).exists():
                raise RuntimeError(f"Destination collision: {dst}")
            destinations.add(dst)
    report = {
        "schema_version": 1,
        "frozen_tag": FROZEN,
        "frozen_commit": git(root, "rev-parse", FROZEN + "^{}").decode().strip(),
        "primary": primary,
        "entries": records,
        "runtime_files": {
            path: hashlib.sha256(git(root, "show", FROZEN + ":" + path)).hexdigest()
            for path, meta in entries.items()
            if path.startswith(primary + "/") and meta["kind"] == "blob"
        },
    }
    if apply:
        for row in records:
            src, dst = root / row["source"], row["destination"]
            if row["action"] == "move":
                (root / dst).parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    ["git", "-C", str(root), "mv", "--", row["source"], dst], check=True
                )
            else:
                # Exact frozen, tracked text paths only. No recursive deletes.
                if src.read_bytes() != git(root, "show", FROZEN + ":" + row["source"]):
                    raise RuntimeError(f"Refusing to retire modified source: {src}")
                subprocess.run(
                    ["apply_patch"],
                    input=f"*** Begin Patch\n*** Delete File: {src}\n*** End Patch\n",
                    text=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
        target = root / "docs/design/layout-migration.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        # Generated mechanical inventory, not handwritten source.
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "repo": str(root),
                "entries": len(records),
                "actions": {
                    action: sum(x["action"] == action for x in records)
                    for action in sorted({x["action"] for x in records})
                },
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    relocate(args.repo.resolve(), args.apply)
