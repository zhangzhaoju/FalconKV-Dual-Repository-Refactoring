#!/usr/bin/env python3
"""Reproduce P2 host/source checks without installing or importing torch/CANN."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
WORKSPACE = OUTPUT.parents[3]
BASELINE = WORKSPACE / "design/p2/baseline/p1-before-p2-20260927.json"


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def run(repo, name, command, *, env=None):
    done = subprocess.run(command, cwd=repo, env=env, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    (OUTPUT / f"{name}.log").write_text("$ " + " ".join(command) + "\n" + done.stdout)
    return {"name": name, "command": command, "returncode": done.returncode,
            "log": f"{name}.log", "passed": done.returncode == 0}, done.stdout


def main():
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "source_and_host_only_not_native_build_ABI_inference_or_performance",
              "python": sys.version, "repositories": {}, "checks": [], "errors": []}
    frozen = json.loads(BASELINE.read_text())["repositories"]
    for name, baseline in frozen.items():
        repo = WORKSPACE / "p1-repos" / name
        refs = {ref: git(repo, "rev-parse", ref) for ref in ("p1", "main", "HEAD")}
        refs["p1_tree"] = git(repo, "rev-parse", "p1^{tree}")
        refs["head_tree"] = git(repo, "rev-parse", "HEAD^{tree}")
        refs["branch"] = git(repo, "branch", "--show-current")
        refs["worktree_clean"] = not git(repo, "status", "--porcelain")
        if not (refs["p1"] == refs["main"] == baseline["p1_commit"]
                and refs["p1_tree"] == baseline["p1_tree"] and refs["branch"] == "p2"):
            report["errors"].append(f"{name}: preserved refs or branch mismatch")
        changed = set(git(repo, "diff", "--name-only", "p1").splitlines())
        changed.update(git(repo, "ls-files", "--others", "--exclude-standard").splitlines())
        refs["changed_file_sha256"] = {
            relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            for relative in sorted(changed) if (repo / relative).is_file()
        }
        production = [relative for relative in refs["changed_file_sha256"]
                      if relative.endswith(".py") and not relative.startswith(
                          ("ascend/legacy_", "ascend/tests/", "tests/"))]
        check, _ = run(repo, f"{name}-ruff", [sys.executable, "-m", "ruff", "check",
                        "--select", "E4,E7,E9,F,I", "--ignore", "E731", *production])
        check["python_files"] = len(production)
        check["scope"] = "changed_production_and_tools_selected_rules_not_full_lint"
        report["checks"].append(check)
        primary = "vllm" if name == "vllm" else "lmcache"
        areas = [repo / primary, repo / "tests/standalone"]
        areas += [repo / "ascend/tests"] if name == "vllm" else [repo / "ascend/lmcache_ascend"]
        python_files = sorted({p for area in areas for p in area.rglob("*.py")})
        syntax_errors = []
        for path in python_files:
            try:
                source = path.read_bytes()
                ast.parse(source, filename=str(path.relative_to(repo)), feature_version=(3, 11))
                compile(source, str(path.relative_to(repo)), "exec")
            except Exception as error:
                syntax_errors.append(f"{path}: {error}")
        report["checks"].append({"name": f"{name}-syntax", "python_files": len(python_files),
                                 "errors": syntax_errors, "passed": not syntax_errors})
        check, _ = run(repo, f"{name}-diff-whitespace", ["git", "diff", "--check", "p1"])
        report["checks"].append(check)
        report["repositories"][name] = refs

    repo = WORKSPACE / "p1-repos/vllm"
    check, output = run(repo, "native-source-gate", [sys.executable, "tools/check_npu_native.py"])
    check["details"] = json.loads(output)
    report["checks"].append(check)
    check, _ = run(repo, "native-package-init", [sys.executable, "ascend/tools/check_python_src_init.py"])
    report["checks"].append(check)
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    suites = [
        ("vllm", 167, ["ascend/tests/standalone", "tests/standalone",
                      "ascend/tests/ut/core/test_kv_connector_worker_metadata_patch.py",
                      "--ignore=ascend/tests/standalone/test_cold_resume_native_metadata.py",
                      "--ignore=ascend/tests/standalone/test_glm52_topk_ownership.py",
                      "-k", "not production_draft_expansion_can_exceed_target_capacity"]),
        ("LMCache", 22, ["tests/standalone/test_p1_development.py"]),
    ]
    report["host_passed_count"] = 0
    for name, count, files in suites:
        command = [sys.executable, "-m", "pytest", "--noconftest", *files,
                   f"--junitxml={OUTPUT / (name + '-host.xml')}", "-q"]
        check, output = run(WORKSPACE / "p1-repos" / name, f"{name}-host", command, env=environment)
        match = re.search(r"(\d+) passed", output)
        check["passed_count"] = int(match.group(1)) if match else 0
        check["expected_count"] = count
        check["passed"] = check["passed"] and check["passed_count"] == count
        report["host_passed_count"] += check["passed_count"]
        report["checks"].append(check)
    report["not_executed_here"] = [
        "test_cold_resume_native_metadata.py (requires torch)",
        "test_glm52_topk_ownership.py (requires torch)",
        "test_mc2_recovery.py::test_production_draft_expansion_can_exceed_target_capacity (requires torch)",
        "full Ascend UT/e2e, real imports and spawn, wheel/sdist/editable build/install",
        "native extensions/TorchAir ABI, NPU kernels, HCCL, GLM52, DSA/MTP/P-D/recovery, performance/soak",
    ]
    for check in report["checks"]:
        if not check["passed"]:
            report["errors"].append(check["name"])
        print(check["name"], "PASS" if check["passed"] else "FAIL")
    report["passed"] = not report["errors"]
    (OUTPUT / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Host tests: {report['host_passed_count']}; result: {report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
