# SPDX-License-Identifier: Apache-2.0
"""Run a fixed, host-only P0 subset without builds or framework test bootstrap.

Uses the existing interpreter and installed pytest/numpy; initialization cases
also require torch indirectly. Missing dependencies remain failed evidence; this
script does not install them or skip failures. No conftest files, pytest plugin
auto-loading or NPU imports are requested. This
is NOT a replacement for the intranet framework/NPU regression suite.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


CASES = {
    "vllm": ["tests/standalone/test_kv_transfer_lifetime.py"],
    "vllm-ascend": [
        "tests/standalone/test_preemption_dispatch.py",
        "tests/standalone/test_connector_control_idle.py",
        "tests/standalone/test_checkpoint_graph_route.py",
    ],
    "LMCache": [
        "tests/standalone/test_checkpoint_adapter.py",
        "tests/standalone/test_checkpoint_idle_path.py",
        "tests/standalone/test_checkpoint_reclaim.py",
        "tests/standalone/test_checkpoint_wrapper.py",
    ],
    "LMCache-Ascend": [
        "tests/standalone/test_checkpoint_dispatch.py",
        "tests/standalone/test_checkpoint_initialization.py",
        "tests/standalone/test_checkpoint_prefix_agreement.py",
        "tests/standalone/test_cold_abort_completion.py",
    ],
}


def junit_counts(path: Path) -> dict:
    """Summarize pytest JUnit leaf suites; missing reports are never success."""
    if not path.is_file():
        return {"report_missing": True}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {
        key: sum(int(suite.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def main() -> None:
    """Execute each repo in a separate process and preserve raw local results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", choices=list(CASES), action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    # Avoid an inherited pytest option enabling unrelated plugins/test suites.
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    report = {
        "scope": "local_host_ast_and_mock_only",
        "is_npu_acceptance": False,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "pytest": metadata.version("pytest"),
        "cases": [],
    }
    failed = False
    for repo in args.repo or CASES:
        directory = args.workspace.resolve() / repo
        junit = output / f"{repo}.xml"
        argv = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--noconftest",
            "-c",
            "/dev/null",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit}",
            *CASES[repo],
        ]
        started = time.monotonic()
        try:
            proc = subprocess.run(
                argv,
                cwd=directory,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
            code, log = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired as exc:
            code = 124
            log = exc.stdout or ""
            if isinstance(log, bytes):
                log = log.decode("utf-8", errors="replace")
            log += "\nP0 host check timeout (180 seconds).\n"
        (output / f"{repo}.log").write_text(log, encoding="utf-8")
        sha = hashlib.sha256(log.encode()).hexdigest()
        counts = junit_counts(junit)
        item = {
            "repository": repo,
            "argv": argv,
            "returncode": code,
            "counts": counts,
            "log_sha256": sha,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        report["cases"].append(item)
        failed |= (
            code != 0
            or counts.get("report_missing", False)
            or counts.get("tests", 0) == 0
        )
        print(
            json.dumps({"repository": repo, "returncode": code, "counts": counts}),
            flush=True,
        )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
