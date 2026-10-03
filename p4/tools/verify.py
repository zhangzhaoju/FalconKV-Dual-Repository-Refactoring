#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run host-only P4 checks and archive exact commands/results in a new directory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--with-installed-torch", action="store_true",
                        help="include CPU tensor tests using an already installed torch; never install it")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for repo in ("vllm", "LMCache"):
        cwd = ROOT / "p1-repos" / repo
        commands = [("source", [args.python, "-B", "tools/check_p4_profile.py"]),
                    ("native", [args.python, "-B", "tools/check_npu_native.py"]),
                    ("diff", ["git", "diff", "--check"])]
        tests = ["tests/standalone"]
        excluded = []
        if repo == "vllm":
            tests += ["ascend/tests/standalone"]
            if not args.with_installed_torch:
                excluded = ["--ignore=ascend/tests/standalone/test_cold_resume_native_metadata.py",
                            "--ignore=ascend/tests/standalone/test_glm52_topk_ownership.py",
                            "-k", "not production_draft_expansion_can_exceed_target_capacity"]
        else:
            tests += ["ascend/tests/v1/test_cache_engine_close_cpu.py", "ascend/tests/v1/test_direct_store_plan.py", "ascend/tests/v1/test_remote_fill_config.py"]
            if not args.with_installed_torch:
                excluded = ["--ignore=tests/standalone/test_glm52_metadata.py"]
        commands.append(("host", [args.python, "-B", "-m", "pytest", "-q", "--noconftest",
                                   "-p", "no:cacheprovider", "--tb=short", *tests, *excluded,
                                   "--junitxml=" + str(output / (repo + "-host.xml"))]))
        for name, command in commands:
            process = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, timeout=180)
            (output / f"{repo}-{name}.log").write_text(process.stdout)
            results.append({"repository": repo, "check": name, "argv": command,
                            "returncode": process.returncode, "passed": process.returncode == 0})
            print(repo, name, "PASS" if process.returncode == 0 else "FAIL", flush=True)
    result = {"scope": "host_only_not_build_ABI_NPU_or_inference", "checks": results,
              "torch_tests_included": args.with_installed_torch,
              "passed": all(r["passed"] for r in results)}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
