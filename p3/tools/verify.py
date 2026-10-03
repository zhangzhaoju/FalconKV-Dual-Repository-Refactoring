#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Record P3 native source/host evidence without installing or compiling frameworks."""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[3]
BASELINE = WORKSPACE / "design/p3/baseline/p2-before-p3-20260928.json"


def git(repo: Path, *args: str) -> str:
    """Read Git state; raise CalledProcessError if the requested input is absent."""
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, timeout=30
    ).strip()


def config_schema_check(repo: Path, commit: str) -> dict[str, Any]:
    """Compare all former injected types/defaults/converters without imports.

    Args:
        repo: LMCache checkout.
        commit: Frozen P2 commit containing the original plugin.

    Returns:
        Source-level equivalence result; this is not a runtime/ABI test.
    """
    legacy = ast.parse(git(repo, "show", f"{commit}:ascend/lmcache_ascend/__init__.py"))
    hook = next(
        node
        for node in legacy.body
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_config"
    )
    old = {
        node.targets[0].slice.value: node.value
        for node in hook.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
    }
    source = (repo / "lmcache/v1/config.py").read_text(encoding="utf-8")
    schema = next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_CONFIG_DEFINITIONS"
    )
    current = {ast.literal_eval(k): v for k, v in zip(schema.keys, schema.values)}
    errors = []
    for name, definition in old.items():
        before = {
            ast.literal_eval(k): ast.dump(v)
            for k, v in zip(definition.keys, definition.values)
        }
        after = (
            {
                ast.literal_eval(k): ast.dump(v)
                for k, v in zip(current[name].keys, current[name].values)
            }
            if name in current
            else {}
        )
        for attribute in ("type", "default", "env_converter"):
            if before.get(attribute) != after.get(attribute):
                errors.append(f"{name}: changed {attribute}")
    if len(old) != 23:
        errors.append("unexpected baseline field count")
    return {
        "name": "P2-injected-config-schema-equivalence",
        "fields": sorted(old),
        "field_count": len(old),
        "compared": ["type", "default", "env_converter"],
        "errors": errors,
        "passed": not errors,
    }


def run_check(
    output: Path,
    repo: Path,
    name: str,
    command: list[str],
    expected_tests: int | None = None,
) -> dict[str, Any]:
    """Run a bounded host check and preserve its command, log and exit code.

    Args:
        output: Fresh evidence directory.
        repo: Command working directory.
        name: Unique check/log name.
        command: Argument list, never evaluated through a shell.
        expected_tests: Exact pytest passing count when applicable.

    Returns:
        Check status including output log and timeout, if any.
    """
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(repo),
    }
    try:
        done = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
        code, log = done.returncode, done.stdout
    except subprocess.TimeoutExpired as error:
        code = 124
        partial = error.stdout or b""
        log = (
            partial.decode(errors="replace") if isinstance(partial, bytes) else partial
        )
        log += "\nHOST CHECK TIMEOUT\n"
    (output / f"{name}.log").write_text(
        "$ " + " ".join(command) + "\n" + log, encoding="utf-8"
    )
    result = {
        "name": name,
        "command": command,
        "cwd": str(repo),
        "log": f"{name}.log",
        "returncode": code,
        "passed": code == 0,
    }
    if expected_tests is not None:
        match = re.search(r"(\d+) passed", log)
        result["passed_count"] = int(match.group(1)) if match else 0
        result["expected_count"] = expected_tests
        result["passed"] = result["passed"] and result["passed_count"] == expected_tests
    print(f"{name}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    return result


def main() -> int:
    """Check frozen ancestry and source/host contracts; write a fresh report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(BASELINE.read_text(encoding="utf-8"))["repositories"]
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "P3",
        "batch": "P3-01-through-P3-05-native-source",
        "scope": "source_and_host_not_native_build_ABI_or_NPU_acceptance",
        "python": sys.version,
        "repositories": {},
        "checks": [],
        "errors": [],
        "not_executed": [
            "wheel/sdist/editable native builds or installs",
            "installed LMCache/vLLM import-order and spawn checks",
            "torch-dependent tests, native extension/TorchAir ABI, kernels and HCCL",
            "GLM52 DSA/MTP, CPU KV, 2P2D, RemoteFill/checkpoint/recovery, performance",
        ],
    }
    for name, baseline in frozen.items():
        repo = WORKSPACE / baseline["path"]
        refs = {
            ref: git(repo, "rev-parse", ref) for ref in ("p1", "main", "p2", "HEAD")
        }
        refs.update(
            branch=git(repo, "branch", "--show-current"),
            p2_tree=git(repo, "rev-parse", "p2^{tree}"),
            head_tree=git(repo, "rev-parse", "HEAD^{tree}"),
            worktree_clean=not git(repo, "status", "--porcelain"),
        )
        ancestry = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo),
                    "merge-base",
                    "--is-ancestor",
                    baseline["p2_commit"],
                    "HEAD",
                ],
                check=False,
                timeout=30,
            ).returncode
            == 0
        )
        refs["contains_P2"] = ancestry
        if not (
            ancestry
            and refs["branch"] == "p3"
            and refs["p2"] == baseline["p2_commit"]
            and refs["p2_tree"] == baseline["p2_tree"]
            and refs["p1"] == refs["main"] == baseline["p1_and_main_commit"]
        ):
            report["errors"].append(f"{name}: frozen refs/branch/ancestry mismatch")
        changed = set(git(repo, "diff", "--name-only", "p2").splitlines())
        changed.update(
            git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
        )
        refs["changed_file_sha256"] = {
            relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
            for relative in sorted(changed)
            if (repo / relative).is_file()
        }
        syntax_errors = []
        for relative in refs["changed_file_sha256"]:
            if relative.endswith(".py"):
                try:
                    source = (repo / relative).read_text(encoding="utf-8")
                    ast.parse(source, filename=relative, feature_version=(3, 11))
                    compile(source, relative, "exec")
                except (SyntaxError, ValueError) as error:
                    syntax_errors.append(f"{relative}: {error}")
        report["checks"].append(
            {
                "name": f"{name}-python311-syntax",
                "errors": syntax_errors,
                "passed": not syntax_errors,
            }
        )
        report["checks"].append(
            run_check(
                output, repo, f"{name}-diff-check", ["git", "diff", "--check", "p2"]
            )
        )
        report["repositories"][name] = refs
    lmcache = WORKSPACE / frozen["LMCache"]["path"]
    report["checks"].append(
        config_schema_check(lmcache, frozen["LMCache"]["p2_commit"])
    )
    vllm = WORKSPACE / frozen["vllm"]["path"]
    report["checks"].append(
        run_check(
            output,
            vllm,
            "vllm-native-source",
            [sys.executable, "-B", "tools/check_npu_native.py"],
        )
    )
    report["checks"].append(
        run_check(output, lmcache, "lmcache-native-source",
                  [sys.executable, "-B", "tools/check_npu_native.py"])
    )
    report["checks"].append(
        run_check(output, WORKSPACE, "method-equivalence",
                  [sys.executable, "-B", "design/p3/tools/check_method_equivalence.py"])
    )
    suites = [
        (
            vllm,
            "vllm-host",
            196,
            [
                "ascend/tests/standalone",
                "tests/standalone",
                "ascend/tests/ut/core/test_kv_connector_worker_metadata_patch.py",
                "--ignore=ascend/tests/standalone/test_cold_resume_native_metadata.py",
                "--ignore=ascend/tests/standalone/test_glm52_topk_ownership.py",
                "-k",
                "not production_draft_expansion_can_exceed_target_capacity",
            ],
        ),
        (
            lmcache,
            "lmcache-host",
            115,
            [
                "tests/standalone",
                "--ignore=tests/standalone/test_glm52_metadata.py",
                "ascend/tests/v1/test_cache_engine_close_cpu.py",
                "ascend/tests/v1/test_direct_store_plan.py",
                "ascend/tests/v1/test_remote_fill_config.py",
            ],
        ),
    ]
    for repo, name, count, files in suites:
        report["checks"].append(
            run_check(
                output,
                repo,
                name,
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "pytest",
                    "--noconftest",
                    "-p",
                    "no:cacheprovider",
                    *files,
                    "-q",
                    f"--junitxml={output / (name + '.xml')}",
                ],
                count,
            )
        )
    report["checks"].append(
        run_check(
            output, lmcache, "remote-fill-protocol",
            [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider",
             "--confcutdir=tests/v1/remote_fill", "tests/v1/remote_fill", "-q",
             f"--junitxml={output / 'remote-fill-protocol.xml'}"],
            98,
        )
    )
    for check in report["checks"]:
        if not check["passed"]:
            report["errors"].append(check["name"])
    report["host_passed_count"] = sum(
        check.get("passed_count", 0) for check in report["checks"]
    )
    report["passed"] = not report["errors"]
    (output / "verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Host tests: {report['host_passed_count']}; passed: {report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
