#!/usr/bin/env python3
"""Mechanical path/version rewrites for the reviewed native-layout migration."""

import json
from pathlib import Path
import re
import subprocess


WORKSPACE = Path(__file__).resolve().parents[3]
for repo, primary in (("vllm", "vllm"), ("LMCache", "lmcache")):
    root = WORKSPACE / "p1-repos" / repo
    branch = subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True
    ).strip()
    commits = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD", "p4-frozen-20261004^{}"],
        text=True,
    ).splitlines()
    if branch != "refactor/native-layout" or len(set(commits)) != 1:
        raise RuntimeError("One-shot path migration is only valid at the frozen input")
    inventory = json.loads((root / "docs/design/layout-migration.json").read_text())
    moves = {
        r["destination"]: r["source"]
        for r in inventory["entries"]
        if r["action"] == "move"
    }
    files = (
        subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    for relative in files:
        path = root / relative
        if (
            not relative
            or not path.is_file()
            or relative.startswith((primary + "/", "csrc/", "docs/", "ascend/"))
        ):
            continue
        if relative == "AGENTS.md":
            continue
        try:
            original = path.read_text()
        except UnicodeError:
            continue
        text = original
        for old, new in (
            ("ascend/csrc/", "csrc/"),
            ("ascend/cmake/", "cmake/"),
            ("ascend/third_party/", "third_party/"),
            ("ascend/submodule-materials.json", "submodule-materials.json"),
            ("ascend/tests/", "tests/"),
            ("ascend/examples/", "examples/"),
            ("ascend/CMakeLists.txt", "cmake/npu_extensions.cmake"),
            ("0.18.0+ascend.p4", "0.18.0+ascend.layout1"),
            ("0.4.3+ascend.p4", "0.4.3+ascend.layout1"),
            ('self.root / "ascend" / relative', "self.root / relative"),
        ):
            text = text.replace(old, new)
        if relative in moves and relative.startswith("tests/"):
            old = moves[relative]
            root_index = len(Path(old).parts) - 1

            def parent(match):
                n = int(match[2])
                return match[1] + str(n - 1 if n >= root_index else n) + "]"

            text = re.sub(
                r"(Path\(__file__\)(?:\.resolve\(\))?\.parents\[)(\d+)\]", parent, text
            )
            text = text.replace(
                "WORKSPACE = ROOT.parents[1]", "WORKSPACE = ROOT.parent"
            )
            if primary == "vllm":
                text = text.replace('"../vllm/', '"vllm/')
            else:
                # Alias used solely to distinguish the two former tests trees.
                text = text.replace(
                    "from tests.v1.utils import", "from tests.v1.utils_npu import"
                )
                text = text.replace("lmcache_tests.", "tests.")
                text = re.sub(
                    r"^from tests.bootstrap import prepare_environment\n\nprepare_environment\(\)\n",
                    "",
                    text,
                    flags=re.M,
                )
                text = text.replace('"lmcache_ascend/', '"lmcache/')
                text = text.replace('"vllm/ascend"', '"vllm"')
                text = text.replace(
                    '"../../vllm/ascend/vllm_ascend/core/recompute_scheduler.py"',
                    '"../vllm/vllm/v1/core/sched/recompute_scheduler.py"',
                )
                text = text.replace(
                    '"vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py"',
                    '"vllm/distributed/kv_transfer/ascend/ascend_multi_connector.py"',
                )
                text = text.replace(
                    '"lmcache/integration/vllm/lmcache_ascend_connector_v1.py"',
                    '"lmcache/integration/vllm/lmcache_connector_v1.py"',
                )
        if text != original:
            path.write_text(text)
