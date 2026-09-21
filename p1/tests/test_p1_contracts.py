# SPDX-License-Identifier: Apache-2.0
"""Source/build-planning contracts only; no backend, compiler or framework import."""

from __future__ import annotations

import ast
import json
import os
import runpy
import tempfile
import unittest
import zipfile
from importlib import util
from pathlib import Path
from unittest.mock import patch

import tomllib

P1 = Path(__file__).resolve().parents[1]
WORKSPACE = Path(os.environ.get("P1_SOURCE_WORKSPACE", P1.parents[1] / "p1-repos"))


def load(path: Path, name: str):
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class P1Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builders = {
            name: load(WORKSPACE / name / "p1_build.py", "p1_" + name)
            for name in ("vllm", "LMCache")
        }
        cls.material = load(P1 / "tools/materialize_submodules.py", "material")
        cls.exporter = load(P1 / "tools/export_sources.py", "exporter")
        cls.inspector = load(P1 / "tools/inspect_wheels.py", "inspector")
        cls.host = load(P1 / "tools/run_host_checks.py", "host")

    def test_independent_build_helpers_are_identical(self):
        self.assertEqual(
            (WORKSPACE / "vllm/p1_build.py").read_bytes(),
            (WORKSPACE / "LMCache/p1_build.py").read_bytes(),
        )

    def test_packaging_owns_both_namespaces_without_test_packages(self):
        for repo, module in self.builders.items():
            primary = repo.lower()
            arguments = module.setup_arguments(primary)
            self.assertIn(primary, arguments["packages"])
            self.assertIn(primary + "_ascend", arguments["packages"])
            self.assertTrue(
                all(
                    n == primary or n.startswith((primary + ".", primary + "_ascend"))
                    for n in arguments["packages"]
                )
            )
            self.assertEqual(
                arguments["package_dir"][primary + "_ascend"],
                f"ascend/{primary}_ascend",
            )
            self.assertEqual(len(arguments["ext_modules"]), 1)

    def test_build_requirements_match_declared_profile(self):
        for repo, module in self.builders.items():
            root = WORKSPACE / repo
            config = tomllib.loads((root / "pyproject.toml").read_text())
            requirements = {
                line
                for line in (root / "requirements/build.txt").read_text().splitlines()
                if line and not line.startswith("#")
            }
            self.assertEqual(set(config["build-system"]["requires"]), requirements)
            self.assertEqual(config["project"]["requires-python"], ">=3.11,<3.12")
            self.assertNotIn("setuptools_scm", config.get("tool", {}))
            self.assertFalse((root / "ascend/setup.py").exists())
            self.assertFalse((root / "ascend/pyproject.toml").exists())
            runtime = module.runtime_requirements()
            self.assertIn("torch==2.9.0", runtime)
            self.assertIn("torch-npu==2.9.0.post1+gitee7ba04", runtime)
            self.assertFalse(
                any(
                    n.startswith(("cupy", "cufile", "nvtx", "nixl", "nvidia-"))
                    for n in runtime
                )
            )

    def test_ascend_entry_point_is_owned_by_vllm(self):
        config = tomllib.loads((WORKSPACE / "vllm/pyproject.toml").read_text())
        self.assertEqual(
            config["project"]["entry-points"]["vllm.platform_plugins"]["ascend"],
            "vllm_ascend:register",
        )

    def test_setup_does_not_probe_devices_or_import_frameworks(self):
        for repo in self.builders:
            root = WORKSPACE / repo
            for name in ("setup.py", "p1_build.py"):
                tree = ast.parse((root / name).read_text())
                for node in tree.body:
                    if isinstance(node, ast.Import):
                        self.assertFalse(
                            any(
                                n.name.startswith(("torch", "vllm", "lmcache"))
                                for n in node.names
                            )
                        )
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse(
                            (node.module or "").startswith(("torch", "vllm", "lmcache"))
                        )

    def test_metadata_uses_one_version_and_valid_cann_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for primary, version in self.inspector.VERSIONS.items():
                self.builders["vllm"].write_build_metadata(
                    root,
                    primary,
                    primary + "_ascend",
                    version,
                    {"cann_version": "8.5.1"},
                )
                for name in (primary, primary + "_ascend"):
                    self.assertEqual(
                        runpy.run_path(str(root / name / "_version.py"))["__version__"],
                        version,
                    )
            info = runpy.run_path(str(root / "lmcache_ascend/_build_info.py"))
            self.assertEqual(info["cann_version_tuple"](), (8, 5, 1))
            self.assertEqual(info["__framework_name__"], "pytorch")

    def test_non_ascend_build_switches_fail_before_subprocess(self):
        builder = self.builders["vllm"]
        for key, value in (
            ("VLLM_TARGET_DEVICE", "empty"),
            ("VLLM_TARGET_DEVICE", "cuda"),
            ("USE_MINDSPORE", "1"),
            ("BUILD_WITH_HIP", "1"),
            ("VLLM_USE_PRECOMPILED", "1"),
            ("COMPILE_CUSTOM_KERNELS", "0"),
        ):
            with (
                self.subTest(key=key, value=value),
                patch.dict(
                    os.environ, {"SOC_VERSION": "ascend910b3", key: value}, clear=True
                ),
                patch.object(builder.subprocess, "check_output") as process,
            ):
                with self.assertRaises(RuntimeError):
                    builder.check_environment()
                process.assert_not_called()

    def test_material_requires_provenance(self):
        builder = self.builders["vllm"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(builder, "ROOT", Path(directory)),
        ):
            with self.assertRaisesRegex(RuntimeError, "materialize_submodules"):
                builder.verify_materials("vllm")

    def test_material_hashes_reject_changes_additions_and_wrong_commit(self):
        builder = self.builders["vllm"]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(builder, "ROOT", Path(directory)),
        ):
            root = Path(directory)
            relative, commit = builder.MATERIALS["vllm"]
            material = root / "ascend" / relative
            material.mkdir(parents=True)
            header = material / "example.h"
            header.write_text("// test fixture\n")
            manifest = root / "ascend/submodule-materials.json"
            record = {
                "path": relative,
                "commit": commit,
                "files": self.material.inventory(material),
            }
            manifest.write_text(json.dumps(record))
            self.assertEqual(builder.verify_materials("vllm")["files"], 1)
            header.write_text("// altered test fixture\n")
            with self.assertRaises(RuntimeError):
                builder.verify_materials("vllm")
            header.write_text("// test fixture\n")
            (material / "unexpected").write_text("test")
            with self.assertRaises(RuntimeError):
                builder.verify_materials("vllm")
            record["commit"] = "0" * 40
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, "pinned commit"):
                builder.verify_materials("vllm")

    def test_runtime_profile_includes_both_32_card_layouts(self):
        profile = json.loads((P1 / "profile.json").read_text())
        runtime = profile["runtime"]
        self.assertTrue(
            runtime["dsa_two_groups"] and runtime["dsa_unbundle"] and runtime["mtp"]
        )
        self.assertFalse(runtime["c8"] or runtime["enable_sparse_c8"])
        for layout in profile["parallel_profiles"]:
            self.assertEqual(layout["tp"] * layout["dp_per_role"] * 2, 32)
            self.assertEqual(layout["tp"] * layout["instances_per_node"], 8)
        self.assertTrue(profile["authorization"]["p0_exit_is_not_assumed_passed"])

    def test_export_keeps_source_directories_named_build_but_not_git(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "vllm/x.py",
                ".git/config",
                "build/generated.cpp",
                "docs/build/guide.md",
                "ascend/upstream-build/setup.py.txt",
                "ascend/csrc/build/generated.cpp",
                "vllm/__pycache__/x.pyc",
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test")
            actual = {
                str(p.relative_to(root)) for p in self.exporter.source_files(root)
            }
            self.assertEqual(
                actual,
                {
                    "vllm/x.py",
                    "docs/build/guide.md",
                    "ascend/upstream-build/setup.py.txt",
                },
            )

    def test_export_refuses_output_inside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                self.exporter.export(root, root / "delivery")

    def test_host_suite_count_is_112_no_missing_reports_pass(self):
        self.assertEqual(sum(self.host.EXPECTED_COUNTS.values()), 112)
        self.assertEqual(self.host.DIRECTORIES["vllm-ascend"], "vllm/ascend")
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(
                self.host.junit_counts(Path(directory) / "absent")["report_missing"]
            )

    def test_wheel_inspection_accepts_complete_synthetic_zip_not_abi(self):
        # Synthetic ZIP fixture only: no wheel backend or native compilation.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            self.make_zip(path)
            result = self.inspector.inspect(path, "vllm")
            self.assertTrue(result["wheel_contents_passed"])
            self.assertFalse(result["abi_tested"])

    def test_wheel_inspection_rejects_missing_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            self.make_zip(path, omit="vllm_ascend/libvllm_ascend_kernels.so")
            with self.assertRaisesRegex(ValueError, "Missing required"):
                self.inspector.inspect(path, "vllm")

    def test_wheel_inspection_rejects_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            self.make_zip(path, tag="cp311-cp311-linux_x86_64")
            with self.assertRaisesRegex(ValueError, "aarch64"):
                self.inspector.inspect(path, "vllm")

    def test_wheel_inspection_rejects_gpu_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            self.make_zip(path, dependency="cupy-cuda12x")
            with self.assertRaisesRegex(ValueError, "device/plugin"):
                self.inspector.inspect(path, "vllm")

    def make_zip(
        self,
        path,
        omit=None,
        tag="cp311-cp311-linux_aarch64",
        dependency="torch==2.9.0",
    ):
        prefix = "vllm-0.18.0+ascend.p1.dist-info/"
        with zipfile.ZipFile(path, "w") as archive:
            for pattern in self.inspector.REQUIRED["vllm"]:
                if pattern != omit:
                    archive.writestr(pattern.replace("*", "fixture"), "test fixture")
            archive.writestr(
                prefix + "METADATA",
                "Metadata-Version: 2.4\nName: vllm\nVersion: 0.18.0+ascend.p1\n"
                f"Requires-Dist: {dependency}\n",
            )
            archive.writestr(
                prefix + "WHEEL", "Root-Is-Purelib: false\nTag: " + tag + "\n"
            )
            archive.writestr(
                prefix + "entry_points.txt",
                "[vllm.platform_plugins]\nascend = vllm_ascend:register\n",
            )


if __name__ == "__main__":
    unittest.main()
