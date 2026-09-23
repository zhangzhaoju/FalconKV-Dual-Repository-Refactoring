# SPDX-License-Identifier: Apache-2.0
"""Exact-hash source update contracts; no frameworks, compiler or Git needed."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from importlib import util
from pathlib import Path
from unittest.mock import patch


SPEC = util.spec_from_file_location(
    "p1_source_audit", Path(__file__).resolve().parents[1] / "tools/audit_sources.py"
)
AUDIT = util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SourceUpdates(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "vllm"
        self.runtime = self.repo / "ascend/vllm_ascend/attention.py"
        self.runtime.parent.mkdir(parents=True)
        self.runtime.write_text("VALUE = 1\n")
        (self.repo / "pyproject.toml").write_text(
            '[project]\nname = "vllm"\nversion = "0.18.0+ascend.p1"\n'
        )
        self.base = self.root / "source-manifest.json"
        self.base.write_text(
            json.dumps(
                {
                    "imports": [
                        {
                            "repository": "vllm",
                            "donor_commit": "a" * 40,
                            "files": [
                                {
                                    "destination": "ascend/vllm_ascend/attention.py",
                                    "mode": "100644",
                                    "sha256_before_adaptation": AUDIT.sha256(
                                        self.runtime
                                    ),
                                }
                            ],
                        }
                    ]
                }
            )
        )
        self.runtime.write_text("VALUE = 2\n")
        self.added = self.repo / "ascend/vllm_ascend/new.py"
        self.added.write_text("NEW = True\n")
        self.updates = self.root / "updates.json"
        self.payload = {
            "schema_version": 1,
            "base_manifest_sha256": AUDIT.sha256(self.base),
            "files": [
                {
                    "repository": "vllm",
                    "destination": str(path.relative_to(self.repo)),
                    "sha256_after_integration": AUDIT.sha256(path),
                }
                for path in (self.runtime, self.added)
            ],
        }

    def run_audit(self):
        self.updates.write_text(json.dumps(self.payload))
        return AUDIT.audit(self.root, self.base, self.updates)

    def test_unregistered_production_change_still_fails(self):
        self.assertTrue(AUDIT.audit(self.root, self.base)["errors"])

    def test_exact_recorded_changes_and_additions_pass(self):
        original = self.base.read_bytes()
        report = self.run_audit()
        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["source_updates"]["verified_files"]), 2)
        self.assertEqual(self.base.read_bytes(), original)
        # Git checkouts must validate dirty primary sources too; unlisted edits fail.
        primary = self.repo / "vllm/scheduler.py"
        primary.parent.mkdir()
        primary.write_text("READY = False\n")
        self.payload["files"].append(
            {
                "repository": "vllm",
                "destination": "vllm/scheduler.py",
                "sha256_after_integration": AUDIT.sha256(primary),
            }
        )
        (self.repo / ".git").mkdir()
        with patch.object(
            AUDIT.subprocess, "check_output", return_value="vllm/scheduler.py\n"
        ):
            self.assertEqual(self.run_audit()["errors"], [])
        with patch.object(
            AUDIT.subprocess, "check_output", return_value="vllm/unlisted.py\n"
        ):
            self.assertTrue(self.run_audit()["errors"])

    def test_changed_recorded_file_fails_even_for_standalone_tests(self):
        for path in (self.runtime, self.repo / "ascend/tests/standalone/test_new.py"):
            with self.subTest(path=path):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 3\n")
                self.payload["files"][0]["destination"] = str(
                    path.relative_to(self.repo)
                )
                self.assertIn(
                    f"Updated file hash mismatch: vllm/{path.relative_to(self.repo)}",
                    self.run_audit()["errors"],
                )

    def test_missing_added_file_fails(self):
        self.added.unlink()
        self.assertTrue(self.run_audit()["errors"])

    def test_wrong_base_manifest_fails(self):
        self.payload["base_manifest_sha256"] = "0" * 64
        self.assertTrue(self.run_audit()["errors"])

    def test_invalid_manifest_records_fail_closed(self):
        valid = json.loads(json.dumps(self.payload))
        variants = [
            {**valid, "schema_version": 2},
            {**valid, "files": []},
            {**valid, "files": [*valid["files"], valid["files"][0]]},
            {**valid, "files": [None]},
        ]
        for field, value in (
            ("repository", "unknown"),
            ("repository", []),
            ("destination", "../outside.py"),
            ("destination", "/tmp/outside.py"),
            ("destination", "ascend/../outside.py"),
            ("destination", "."),
            ("sha256_after_integration", "bad"),
        ):
            variants.append({**valid, "files": [{**valid["files"][0], field: value}]})
        for payload in variants:
            with self.subTest(payload=payload):
                self.payload = payload
                self.assertTrue(self.run_audit()["errors"])

    def test_approval_does_not_whitelist_unlisted_runtime_changes(self):
        unlisted = self.repo / "ascend/vllm_ascend/unlisted.py"
        unlisted.write_text("VALUE = 0\n")
        manifest = json.loads(self.base.read_text())
        manifest["imports"][0]["files"].append(
            {
                "destination": str(unlisted.relative_to(self.repo)),
                "mode": "100644",
                "sha256_before_adaptation": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
            }
        )
        self.base.write_text(json.dumps(manifest))
        self.payload["base_manifest_sha256"] = AUDIT.sha256(self.base)
        self.assertTrue(self.run_audit()["errors"])


if __name__ == "__main__":
    unittest.main()
