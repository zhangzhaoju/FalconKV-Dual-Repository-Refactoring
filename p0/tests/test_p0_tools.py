# SPDX-License-Identifier: Apache-2.0
"""Standard-library-only regression checks for P0 tooling and scope guards."""

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch


P0 = Path(__file__).resolve().parents[1]
WORKSPACE = P0.parents[1]
sys.path.insert(0, str(P0 / "tools"))

# This is an unpackaged audit toolkit, loaded only after adding its local path.
from build_review import class_contracts, disposition  # noqa: E402
from collect_intranet_env import (  # noqa: E402
    MAX_CONFIG_BYTES,
    collect,
    evaluate_versions,
    model_metadata,
)
from run_host_checks import junit_counts  # noqa: E402
from source_inventory import PythonInventory, model_registry, write_json  # noqa: E402


class StaticInventoryTests(unittest.TestCase):
    def test_import_conditions_and_order_are_preserved(self) -> None:
        node = ast.parse(
            "if enabled:\n import example.first\nelse:\n from example import second as alias\n"
        )
        visitor = PythonInventory()
        visitor.visit(node)
        self.assertEqual(visitor.imports[0]["lexical_guards"], ["enabled"])
        self.assertEqual(visitor.imports[1]["lexical_guards"], ["not (enabled)"])
        self.assertEqual(visitor.imports[1]["names"][0]["asname"], "alias")

    def test_patch_candidates_are_not_marked_as_proven(self) -> None:
        visitor = PythonInventory(True)
        visitor.visit(
            ast.parse(
                "def _patch_ops():\n module.op = replacement\n setattr(module, 'x', impl)\n"
            )
        )
        self.assertEqual(len(visitor.patch_candidates), 2)
        self.assertTrue(
            all(
                x["review_status"] == "needs_semantic_review"
                for x in visitor.patch_candidates
            )
        )
        self.assertEqual(visitor.patch_candidates[0]["scope"], "_patch_ops")

    def test_no_assignments_indexed_as_patches_outside_patch_files(self) -> None:
        visitor = PythonInventory(False)
        visitor.visit(ast.parse("obj.a = value"))
        self.assertEqual(visitor.patch_candidates, [])

    def test_dynamic_import_is_flagged_not_executed(self) -> None:
        visitor = PythonInventory()
        visitor.visit(
            ast.parse("importlib.import_module(user_value)\n__import__('known.module')")
        )
        self.assertEqual([x["literal"] for x in visitor.dynamic_imports], [False, True])

    def test_registry_extraction_does_not_evaluate_code(self) -> None:
        rows = model_registry(
            ast.parse(
                "r = {'GlmMoeDsaForCausalLM': ('deepseek_v2', 'GlmMoeDsaForCausalLM'), 'unsafe': factory()}"
            )
        )
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["runtime_verified"])

    def test_generated_artifacts_never_overwrite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0-unit-") as directory:
            path = Path(directory) / "artifact.json"
            write_json(path, {"original": True})
            with self.assertRaises(FileExistsError):
                write_json(path, {"original": False})
            self.assertEqual(json.loads(path.read_text()), {"original": True})


class EnvironmentTests(unittest.TestCase):
    def test_version_metadata_accepts_local_suffix_without_claiming_abi(self) -> None:
        rows = evaluate_versions({"torch": ["2.9.0+local"], "torch-npu": ["2.9.0"]})
        self.assertTrue(all(x["status"] == "metadata_matches" for x in rows))
        self.assertTrue(all(not x["runtime_abi_verified"] for x in rows))

    def test_version_mismatch_and_absence_require_review(self) -> None:
        rows = evaluate_versions({"torch": ["2.10.0"]})
        self.assertTrue(all(x["status"] == "needs_review" for x in rows))

    def test_proxy_secrets_not_collected_and_policy_not_assumed(self) -> None:
        args = argparse.Namespace(
            cann_root=Path("/nonexistent-p0-toolkit"),
            model=[],
            probe_runtime=False,
            probe_npu=False,
        )
        with (
            patch.dict(
                "os.environ",
                {
                    "HTTPS_PROXY": "https://secret-user:secret-password@private-proxy:8443"
                },
            ),
            patch("collect_intranet_env.package_inventory", return_value={}),
            patch("collect_intranet_env.shutil.which", return_value=None),
        ):
            result = collect(args)
        encoded = json.dumps(result)
        self.assertNotIn("secret-password", encoded)
        self.assertNotIn("private-proxy", encoded)
        self.assertTrue(result["proxy_configuration_presence_only"]["HTTPS_PROXY"])
        self.assertFalse(
            result["network_policy"]["external_llm_access_verified_disabled"]
        )
        self.assertEqual(result["runtime_probe"], {"status": "not_requested"})

    def test_model_metadata_hashes_local_config_without_loading_weights(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0-model-test-") as directory:
            root = Path(directory)
            data = b'{"architectures": ["GlmMoeDsaForCausalLM"], "model_type": "glm_moe_dsa"}'
            (root / "config.json").write_bytes(data)
            result = model_metadata("glm52=" + directory)
            self.assertEqual(
                result["metadata_files"][0]["sha256"], hashlib.sha256(data).hexdigest()
            )
            self.assertFalse(result["runtime_verified"])

    def test_model_alias_requires_local_directory(self) -> None:
        with self.assertRaises(ValueError):
            model_metadata("glm52=https://example.invalid/model")

    def test_large_quantization_and_index_metadata_are_hashed_without_parsing(
        self,
    ) -> None:
        # Match the 17,224,106-byte quantization description reported in the intranet.
        data = b"{}" + b" " * (17_224_106 - 2)
        expected_hash = hashlib.sha256(data).hexdigest()
        config = b'{"architectures": ["GlmMoeDsaForCausalLM"]}'
        for name in ("quant_model_description.json", "model.safetensors.index.json"):
            with (
                self.subTest(name=name),
                tempfile.TemporaryDirectory(prefix="p0-large-metadata-") as directory,
            ):
                root = Path(directory)
                (root / "config.json").write_bytes(config)
                (root / name).write_bytes(data)
                with (
                    patch.object(
                        Path,
                        "read_bytes",
                        side_effect=AssertionError("Metadata must be streamed"),
                    ),
                    patch("collect_intranet_env.json.loads", wraps=json.loads) as loads,
                ):
                    result = model_metadata("glm52=" + directory)
                loads.assert_called_once_with(config)
                self.assertEqual(result["architectures"], ["GlmMoeDsaForCausalLM"])
                self.assertEqual(
                    result["metadata_files"][1],
                    {"file": name, "size_bytes": len(data), "sha256": expected_hash},
                )
                self.assertFalse(result["runtime_verified"])

    def test_metadata_reads_are_bounded_and_include_final_partial_chunk(self) -> None:
        data = b'{"synthetic": 42}'
        read_sizes = []
        test = self

        class BoundedReader(io.BytesIO):
            def read(self, size: int = -1) -> bytes:
                test.assertGreater(size, 0)
                test.assertLessEqual(size, 8)
                read_sizes.append(size)
                return super().read(size)

        with tempfile.TemporaryDirectory(prefix="p0-chunked-metadata-") as directory:
            (Path(directory) / "quant_model_description.json").touch()
            with (
                patch("collect_intranet_env.HASH_CHUNK_BYTES", 8),
                patch.object(Path, "open", return_value=BoundedReader(data)),
            ):
                result = model_metadata("glm52=" + directory)
        self.assertGreater(len(read_sizes), 2)
        self.assertEqual(result["metadata_files"][0]["size_bytes"], len(data))
        self.assertEqual(
            result["metadata_files"][0]["sha256"], hashlib.sha256(data).hexdigest()
        )

    def test_config_json_parsing_limit_is_still_enforced(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p0-large-config-") as directory:
            with (Path(directory) / "config.json").open("wb") as stream:
                stream.truncate(MAX_CONFIG_BYTES + 1)
            with patch("collect_intranet_env.json.loads") as loads:
                with self.assertRaisesRegex(
                    ValueError, r"Unexpectedly large config file: glm52/config\.json"
                ):
                    model_metadata("glm52=" + directory)
            loads.assert_not_called()

    def test_config_json_at_parsing_limit_is_accepted(self) -> None:
        data = b"{}      "
        with tempfile.TemporaryDirectory(prefix="p0-config-limit-") as directory:
            (Path(directory) / "config.json").write_bytes(data)
            with patch("collect_intranet_env.MAX_CONFIG_BYTES", len(data)):
                result = model_metadata("glm52=" + directory)
        self.assertEqual(result["metadata_files"][0]["size_bytes"], len(data))
        self.assertEqual(
            result["metadata_files"][0]["sha256"], hashlib.sha256(data).hexdigest()
        )


class ScopeAndBuildTests(unittest.TestCase):
    def test_new_scope_and_topology_are_explicit(self) -> None:
        scope = json.loads((P0 / "support-matrix.json").read_text())
        self.assertEqual(scope["model"]["artifact_label"], "GLM-5.2-w4a8c8")
        self.assertEqual(scope["deployment"]["total_cards"], 32)
        self.assertEqual(sum(x["cards"] for x in scope["deployment"]["roles"]), 32)
        self.assertFalse(scope["future_extension"]["implemented_or_certified"])
        self.assertFalse(scope["p0_exit_passed"])

    def test_shared_deepseek_code_is_not_bulk_deleted(self) -> None:
        result = disposition("vllm", "vllm/model_executor/models/deepseek_v2.py")
        self.assertEqual(result["action"], "retain_glm52_implementation_components")
        self.assertFalse(result["ready_to_execute"])
        self.assertEqual(
            disposition("vllm", "vllm/model_executor/models/glm4.py")["action"],
            "non_target_model_candidate_remove_after_closure",
        )

    def test_all_build_metadata_agree_with_installed_target(self) -> None:
        for repo in ("vllm", "vllm-ascend", "LMCache", "LMCache-Ascend"):
            with self.subTest(repo=repo):
                data = tomllib.loads((WORKSPACE / repo / "pyproject.toml").read_text())
                requirements = [
                    line.replace(" ", "") for line in data["build-system"]["requires"]
                ]
                self.assertIn("torch==2.9.0", requirements)
                if "Ascend" in repo or "ascend" in repo:
                    self.assertIn("torch-npu==2.9.0", requirements)
        self.assertIn(
            "torch==2.9.0", (WORKSPACE / "vllm/requirements/build.txt").read_text()
        )

    def test_subclass_comparison_resolves_actual_v2_parent(self) -> None:
        index = json.loads((P0 / "baseline/python-dependencies.json").read_text())[
            "files"
        ]
        contracts = class_contracts(index)
        self.assertEqual(len(contracts), 4)
        self.assertTrue(
            all(x["status"].startswith("static_comparison") for x in contracts)
        )
        self.assertEqual(
            contracts[1]["parent"][1], "vllm/v1/worker/gpu/model_runner.py"
        )

    def test_missing_test_report_is_not_success(self) -> None:
        self.assertTrue(
            junit_counts(Path("/nonexistent-p0-report.xml"))["report_missing"]
        )

    def test_source_restore_verifications_recorded_and_submodule_gap_visible(
        self,
    ) -> None:
        source = json.loads((P0 / "baseline/source-manifest.json").read_text())
        for repo in source["repos"]:
            self.assertEqual(repo["head"], repo["backup"]["restored_head"])
            self.assertEqual(repo["tree"], repo["backup"]["restored_tree"])
            self.assertEqual(repo["backup"]["fsck"], "passed")
        self.assertEqual(sum(len(x["gitlinks"]) for x in source["repos"]), 2)


if __name__ == "__main__":
    unittest.main()
