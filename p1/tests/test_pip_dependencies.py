# SPDX-License-Identifier: Apache-2.0
"""Approved pip metadata exceptions only; tests never run pip or install packages."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from importlib import util
from pathlib import Path
from unittest.mock import patch

P1 = Path(__file__).resolve().parents[1]
SPEC = util.spec_from_file_location(
    "p1_pip_gate", P1 / "tools/check_pip_dependencies.py"
)
GATE = util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


class PipDependencyWaivers(unittest.TestCase):
    """Ensure the user-approved exception never suppresses unrelated failures."""

    def setUp(self) -> None:
        self.available = dict.fromkeys(GATE.STDLIB_MODULES, True)
        self.warnings = "\n".join(GATE.APPROVED_MESSAGES) + "\n"

    def test_clean_pip_check_passes_without_waivers(self) -> None:
        self.assertEqual(GATE.stdlib_availability(), self.available)
        report = GATE.assess(0, "No broken requirements found.\n", self.available)
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["waived_messages"], [])

    def test_exact_three_messages_pass_with_original_failure_code(self) -> None:
        report = GATE.assess(1, self.warnings, self.available)
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "passed_with_waivers")
        self.assertEqual(report["raw_returncode"], 1)
        self.assertEqual(set(report["waived_messages"]), set(GATE.APPROVED_MESSAGES))
        self.assertEqual(report["blocking_errors"], [])

    def test_subset_and_line_order_do_not_require_all_three_errors(self) -> None:
        for line in reversed(list(GATE.APPROVED_MESSAGES)):
            with self.subTest(line=line):
                self.assertTrue(
                    GATE.assess(1, f"\r\n  {line}\r\n", self.available)["passed"]
                )

    def test_real_dependencies_and_version_conflicts_remain_blocking(self) -> None:
        for diagnostic in (
            "op-compile-tool 0.1.0 requires absl-py, which is not installed.",
            "op-compile-tool 0.1.0 requires ml-dtypes, which is not installed.",
            "op-compile-tool 0.1.0 requires tornado, which is not installed.",
            "example 1.0 requires torch==2.9.0, but you have torch 2.10.0.",
            "WARNING: invalid metadata in another distribution",
        ):
            with self.subTest(diagnostic=diagnostic):
                report = GATE.assess(1, self.warnings + diagnostic, self.available)
                self.assertFalse(report["passed"])
                self.assertIn(diagnostic, report["blocking_errors"])

    def test_other_package_version_or_message_is_not_approved(self) -> None:
        for diagnostic in (
            self.warnings.replace("0.1.0", "0.1.1"),
            self.warnings.replace("op-compile-tool", "another-tool"),
            "op-compile-tool 0.1.0 requires inspect>=1, which is not installed.",
            "op-compile-tool 0.1.0 requires inspect, which is not installed. EXTRA",
        ):
            self.assertFalse(GATE.assess(1, diagnostic, self.available)["passed"])

    def test_unexpected_status_or_empty_output_is_never_success(self) -> None:
        for code, output in (
            (0, self.warnings),
            (0, ""),
            (1, ""),
            (2, self.warnings),
            (-9, self.warnings),
        ):
            with self.subTest(code=code, output=output):
                self.assertFalse(GATE.assess(code, output, self.available)["passed"])

    def test_missing_actual_stdlib_module_is_not_waived(self) -> None:
        self.available["inspect"] = False
        report = GATE.assess(1, self.warnings, self.available)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["waived_messages"]), 2)

    def test_runner_preserves_raw_output_and_separate_gate_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "check"
            raw = self.warnings.encode()
            with (
                patch.object(
                    GATE.subprocess,
                    "run",
                    return_value=subprocess.CompletedProcess([], 1, stdout=raw),
                ) as run,
                patch.object(GATE, "stdlib_availability", return_value=self.available),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(GATE.main(["--output", str(output)]), 0)
            self.assertEqual((output / "pip-check.txt").read_bytes(), raw)
            self.assertEqual((output / "pip-check.exitcode").read_text(), "1\n")
            report = json.loads((output / "report.json").read_text())
            self.assertTrue(report["passed"])
            command = run.call_args.args[0]
            self.assertEqual(command[0], GATE.sys.executable)
            self.assertEqual(command[-1], "check")
            self.assertIn("--disable-pip-version-check", command)
            self.assertNotIn("install", command)

    def test_existing_results_are_not_overwritten_or_rechecked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "pip-check.txt").write_text("earlier evidence")
            with patch.object(GATE.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    GATE.run_check(output)
            run.assert_not_called()
            self.assertEqual((output / "pip-check.txt").read_text(), "earlier evidence")

    def test_spawn_failure_and_timeout_keep_failure_evidence(self) -> None:
        for failure in (
            OSError("fixture spawn error"),
            subprocess.TimeoutExpired(["pip"], 60, output=b"partial fixture log"),
        ):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "failed"
                with patch.object(GATE.subprocess, "run", side_effect=failure):
                    report = GATE.run_check(output)
                self.assertFalse(report["passed"])
                self.assertIsNone(report["raw_returncode"])
                self.assertEqual(
                    (output / "pip-check.exitcode").read_text(), "unavailable\n"
                )
                self.assertTrue((output / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
