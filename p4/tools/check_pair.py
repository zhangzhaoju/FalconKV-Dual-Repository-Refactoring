#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only per-container P4 pair/source identity; no product or NPU imports."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
from importlib.machinery import PathFinder
import json
from pathlib import Path
import socket
import subprocess
import sys


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected = json.loads(args.pair.read_text())["repositories"]
    result = {"scope": "pair_and_installed_python_files_not_ABI_or_NPU",
              "host": socket.gethostname(), "python": sys.executable, "repositories": {}, "errors": []}
    for repo, package in (("vllm", "vllm"), ("LMCache", "lmcache")):
        source = args.workspace.resolve() / repo
        try:
            actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
            if actual != expected[repo]["commit"]:
                raise ValueError(f"{repo}: checkout {actual} != pinned {expected[repo]['commit']}")
            for diff in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
                subprocess.run(["git", "-C", str(source), *diff], check=True)
            dist = metadata.distribution(package)
            if dist.version != expected[repo]["version"]:
                raise ValueError(f"{repo}: installed {dist.version} != {expected[repo]['version']}")
            spec = PathFinder.find_spec(package)
            if spec is None or spec.origin is None:
                raise ValueError(f"{package}: package not discoverable")
            installed = Path(spec.origin).absolute().parent  # Do not resolve strict editable root.
            source_files = subprocess.check_output(["git", "-C", str(source), "ls-files", "--", package], text=True).splitlines()
            checked = 0
            for relative in source_files:
                if not relative.endswith(".py"):
                    continue
                file = source / relative
                counterpart = installed / file.relative_to(source / package)
                if not counterpart.is_file() or digest(file) != digest(counterpart):
                    raise ValueError(f"{package}: missing/stale installed Python file {relative}")
                checked += 1
            native = {str(p.relative_to(installed)): digest(p) for p in installed.rglob("*.so")}
            if not native:
                raise ValueError(f"{package}: no native artifacts in installed package")
            result["repositories"][repo] = {"commit": actual, "version": dist.version,
                "installed_path": str(installed), "python_files_checked": checked,
                "native_sha256": native,
                "direct_url": json.loads(dist.read_text("direct_url.json") or "{}")}
        except Exception as exc:
            result["errors"].append(str(exc))
    result["passed"] = not result["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
