# SPDX-License-Identifier: Apache-2.0
"""Read-only candidate version gates using synthetic installed metadata, no NPU."""

from __future__ import annotations

import os
import tempfile
import unittest
from importlib import util
from pathlib import Path
from unittest.mock import patch

P1 = Path(__file__).resolve().parents[1]
SPEC = util.spec_from_file_location("p1_preflight_test", P1 / "tools/preflight.py")
PREFLIGHT = util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


class PreflightCandidates(unittest.TestCase):
    """Verify exact candidates without importing torch/Transformers or installing."""

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="p1-preflight-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ("vllm", "LMCache"):
            repo = self.root / name
            (repo / "requirements").mkdir(parents=True)
            (repo / "pyproject.toml").write_text(
                '[build-system]\nrequires = ["torch==2.9.0", '
                '"torch-npu==2.9.0.post2"]\n'
            )
            (repo / "requirements/ascend.txt").write_text(
                "torch==2.9.0\ntorch-npu==2.9.0.post2\ntransformers==5.2.0\n"
            )
        self.cann_info = self.root / "sdk/aarch64-linux/ascend_toolkit_install.info"
        self.cann_info.parent.mkdir(parents=True)
        self.cann_info.write_text("version=8.5.1\n")
        self.installed = {
            "build": "1.2.2",
            "torch": "2.9.0+cpu",
            "torch-npu": "2.9.0.post2",
            "transformers": "5.2.0",
        }
        self.enterContext(
            patch.object(PREFLIGHT.metadata, "version", side_effect=self.version)
        )
        self.enterContext(patch.object(PREFLIGHT.sys, "version_info", (3, 11, 14)))
        self.enterContext(
            patch.object(PREFLIGHT.platform, "machine", return_value="aarch64")
        )
        self.enterContext(
            patch.object(PREFLIGHT.shutil, "which", return_value="/fixture/tool")
        )
        self.enterContext(
            patch.dict(os.environ, {"ASCEND_HOME_PATH": str(self.root / "sdk")})
        )

    def version(self, name: str) -> str:
        """Return synthetic package metadata or the real missing-package exception."""
        if name not in self.installed:
            raise PREFLIGHT.metadata.PackageNotFoundError(name)
        return self.installed[name]

    def test_reported_intranet_candidates_pass_metadata_gate(self) -> None:
        report = PREFLIGHT.collect(self.root)
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["workspace"], str(self.root.resolve()))
        self.assertEqual(report["scope"], "direct_metadata_only_not_ABI_or_runtime")
        self.assertEqual(len(report["requirements"]), 4)
        self.assertTrue(all(item["passed"] for item in report["requirements"]))

    def test_previous_candidates_are_not_silently_accepted(self) -> None:
        for name, old in (
            ("torch-npu", "2.9.0.post1+gitee7ba04"),
            ("transformers", "4.57.4"),
        ):
            with self.subTest(name=name), patch.dict(self.installed, {name: old}):
                report = PREFLIGHT.collect(self.root)
                self.assertFalse(report["passed"])
                self.assertEqual(len(report["errors"]), 1)
                self.assertIn(f"installed={old}", report["errors"][0])

    def test_unapproved_future_versions_still_fail(self) -> None:
        for name, future in (
            ("torch", "2.10.0"),
            ("torch-npu", "2.9.0.post3"),
            ("transformers", "5.3.0"),
        ):
            with self.subTest(name=name), patch.dict(self.installed, {name: future}):
                report = PREFLIGHT.collect(self.root)
                self.assertFalse(report["passed"])
                self.assertEqual(len(report["errors"]), 1)
                self.assertIn(f"installed={future}", report["errors"][0])

    def test_missing_dependency_is_not_skipped(self) -> None:
        del self.installed["transformers"]
        report = PREFLIGHT.collect(self.root)
        self.assertFalse(report["passed"])
        self.assertIn(
            "Missing or mismatched: transformers==5.2.0; installed=None",
            report["errors"],
        )

    def test_stale_source_snapshot_is_not_overridden_by_tool(self) -> None:
        (self.root / "vllm/requirements/ascend.txt").write_text(
            "transformers>=4.57.4,<5\n"
        )
        report = PREFLIGHT.collect(self.root)
        self.assertFalse(report["passed"])
        self.assertIn(
            "Missing or mismatched: transformers>=4.57.4,<5; installed=5.2.0",
            report["errors"],
        )

    def test_python_architecture_and_cann_checks_remain_enforced(self) -> None:
        with patch.object(PREFLIGHT.platform, "machine", return_value="x86_64"):
            self.assertFalse(PREFLIGHT.collect(self.root)["passed"])
        with patch.object(PREFLIGHT.sys, "version_info", (3, 12, 3)):
            self.assertFalse(PREFLIGHT.collect(self.root)["passed"])
        self.cann_info.write_text("version=8.5.0\n")
        report = PREFLIGHT.collect(self.root)
        self.assertFalse(report["passed"])
        self.assertIn(
            "ASCEND_HOME_PATH does not identify the approved CANN 8.5.1",
            report["errors"],
        )

    def test_missing_build_tool_still_fails(self) -> None:
        with patch.object(PREFLIGHT.shutil, "which", return_value=None):
            report = PREFLIGHT.collect(self.root)
        self.assertFalse(report["passed"])
        self.assertIn("Missing tool: cmake", report["errors"])


if __name__ == "__main__":
    unittest.main()
