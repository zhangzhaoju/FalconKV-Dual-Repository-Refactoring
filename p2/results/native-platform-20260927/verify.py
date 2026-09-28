#!/usr/bin/env python3
"""Reproduce this P2 batch's host checks; never build or install a framework."""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
WORKSPACE = OUTPUT.parents[3]
BASELINE = WORKSPACE / "design/p2/baseline/p1-before-p2-20260927.json"


def run(repo: Path, args: list[str], source: str | None = None):
    return subprocess.run(
        args, cwd=repo, input=source, capture_output=True, text=True, timeout=60
    )


def git(repo: Path, *args: str) -> str:
    result = run(repo, ["git", *args])
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def methods(source: str) -> dict[str, str]:
    cls = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "NPUPlatform"
    )
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
    }


def main() -> int:
    report = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "host_only_not_full_framework_import_or_NPU_ABI_acceptance",
        "python": sys.version,
        "repositories": {},
        "tests": [],
        "errors": [],
    }
    expected_versions = {"vllm": "0.18.0+ascend.p2", "lmcache": "0.4.3+ascend.p1"}
    frozen = json.loads(BASELINE.read_text())["repositories"]
    for name, baseline in frozen.items():
        repo = WORKSPACE / "p1-repos" / name
        retained = git(repo, "rev-parse", "p1")
        tree = git(repo, "rev-parse", "p1^{tree}")
        unchanged = retained == baseline["p1_commit"] and tree == baseline["p1_tree"]
        if not unchanged or git(repo, "rev-parse", "main") != retained:
            report["errors"].append(f"{name}: P1/main preservation mismatch")
        helper = ast.parse((repo / "p1_dev.py").read_text())
        versions = next(
            ast.literal_eval(node.value)
            for node in helper.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "VERSIONS" for t in node.targets
            )
        )
        if versions != expected_versions:
            report["errors"].append(f"{name}: paired installer versions differ")
        changed = set(
            git(repo, "diff", "--name-only", baseline["p1_commit"]).splitlines()
        )
        changed.update(
            git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
        )
        hashes, lint = {}, {}
        for relative in sorted(changed):
            path = repo / relative
            if not path.is_file():
                continue
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.suffix != ".py":
                continue
            source = path.read_text()
            compile(source, relative, "exec")
            donor = (
                "ascend/vllm_ascend/platform.py"
                if relative == "vllm/platforms/npu.py"
                else relative
            )
            previous = run(repo, ["git", "show", f"{baseline['p1_commit']}:{donor}"])
            args = [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--output-format",
                "json",
                "--stdin-filename",
                str(path),
                "-",
            ]
            current = run(repo, args, source)
            prior = run(repo, args, previous.stdout if previous.returncode == 0 else "")
            if current.returncode not in (0, 1) or prior.returncode not in (0, 1):
                raise RuntimeError(current.stderr + prior.stderr)
            findings, old_findings = (
                json.loads(current.stdout),
                json.loads(prior.stdout),
            )
            counts = collections.Counter(item["code"] for item in findings)
            old_counts = collections.Counter(item["code"] for item in old_findings)
            increase = counts - old_counts
            lint[relative] = {
                "p1_rule_counts": dict(old_counts),
                "p2_rule_counts": dict(counts),
                "increased_counts": dict(increase),
                "p2_findings": findings,
            }
            if increase:
                report["errors"].append(
                    f"{name}/{relative}: Ruff count increase {increase}"
                )
        result = {
            "retained_p1_commit": retained,
            "retained_p1_tree": tree,
            "p1_and_main_preserved": unchanged,
            "branch": git(repo, "branch", "--show-current"),
            "head_at_check": git(repo, "rev-parse", "HEAD"),
            "paired_installer_versions": versions,
            "changed_file_sha256": hashes,
            "ruff": lint,
            "full_ruff_passed": all(
                not entry["p2_findings"] for entry in lint.values()
            ),
        }
        if name == "vllm":
            old = methods(
                git(repo, "show", f"{retained}:ascend/vllm_ascend/platform.py")
            )
            new = methods((repo / "vllm/platforms/npu.py").read_text())
            changed_methods = sorted(
                key for key in old.keys() & new.keys() if old[key] != new[key]
            )
            added, removed = (
                sorted(new.keys() - old.keys()),
                sorted(old.keys() - new.keys()),
            )
            result["platform_method_ast"] = {
                "p1_count": len(old),
                "p2_count": len(new),
                "identical_count": sum(
                    old[key] == new[key] for key in old.keys() & new.keys()
                ),
                "changed": changed_methods,
                "added": added,
                "removed": removed,
            }
            if (
                changed_methods != ["import_kernels"]
                or added != ["register_builtin_components"]
                or removed
            ):
                report["errors"].append("Unexpected NPUPlatform method migration delta")
        report["repositories"][name] = result

    suites = [
        ("vllm", "tests/standalone/test_p2_platform.py", 15),
        ("vllm", "ascend/tests/standalone/test_p1_resources.py", 2),
        ("vllm", "tests/standalone/test_p1_development.py", 22),
        ("LMCache", "tests/standalone/test_p1_development.py", 22),
    ]
    for name, suite, expected_count in suites:
        args = [sys.executable, "-B", suite, "-v"]
        result = run(WORKSPACE / "p1-repos" / name, args)
        output = result.stdout + result.stderr
        log = f"{name}-{Path(suite).stem}.log"
        (OUTPUT / log).write_text("$ " + " ".join(args) + "\n" + output)
        match = re.search(r"Ran (\d+) tests? in", output)
        count = int(match.group(1)) if match else None
        passed = result.returncode == 0 and count == expected_count
        report["tests"].append(
            {
                "repository": name,
                "command": args,
                "log": log,
                "returncode": result.returncode,
                "count": count,
                "passed": passed,
            }
        )
        if not passed:
            report["errors"].append(f"{name}/{suite}: host tests failed")
        print(f"{name}/{suite}: {'PASS' if passed else 'FAIL'} ({count})")

    report["passed"] = not report["errors"]
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Source/host checks:", "PASS" if report["passed"] else "FAIL")
    for error in report["errors"]:
        print(error)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
