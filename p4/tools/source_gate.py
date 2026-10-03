#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free P4 source gate; not a build, ABI or NPU acceptance test."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import tomllib

try:
    from check_p4_exports import check
except ModuleNotFoundError as exc:
    if exc.name != "check_p4_exports":
        raise
    from check_python_exports import check

VERSIONS = {"vllm": "0.18.0+ascend.p4", "lmcache": "0.4.3+ascend.p4"}
FORBIDDEN_IMPORTS = (
    "vllm_ascend",
    "lmcache_ascend",
    "flash_attn",
    "flashinfer",
    "pynvml",
    "torch_npu.contrib.transfer_to_npu",
    "cupy",
    "mistral_common",
    "sglang",
    "mindspore",
    "triton.language.extra.cuda",
    "vllm._custom_ops",
    "vllm._xpu_ops",
    "vllm._aiter_ops",
    "vllm._oink_ops",
)
REMOVED = {
    "vllm": (
        "vllm/multimodal",
        "vllm/platforms/ascend_310p",
        "vllm/platforms/cuda.py",
        "vllm/platforms/rocm.py",
        "vllm/platforms/xpu.py",
        "vllm/platforms/cpu.py",
        "vllm/v1/worker/gpu",
        "vllm/entrypoints/pooling",
        "vllm/distributed/weight_transfer",
        "vllm/_xpu_ops.py",
        "vllm/v1/pool/late_interaction.py",
        "vllm/v1/worker/mamba_utils.py",
        "ascend/csrc/causal_conv1d",
        "vllm/model_executor/layers/ascend/triton/layernorm_gated.py",
        "vllm/model_executor/layers/ascend/triton/linearnorm/split_qkv_rmsnorm_mrope.py",
    ),
    "lmcache": (
        "lmcache/v0",
        "lmcache/integration/sglang",
        "lmcache/v1/compute",
        "lmcache/v1/transfer_channel/nixl_channel.py",
        "ascend/csrc/mindspore",
    ),
}


def audit(root: Path) -> dict:
    """Audit current Python sources and package identity without importing them."""
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    package = project["name"].lower()
    report = check(root, package)
    errors = report["errors"]
    if project["version"] != VERSIONS[package]:
        errors.append({"version": project["version"], "expected": VERSIONS[package]})
    for relative in REMOVED[package]:
        path = root / relative
        if path.is_file() or (
            path.is_dir() and any(p.is_file() for p in path.rglob("*"))
        ):
            errors.append({"retired_tree": relative})
    files = list((root / package).rglob("*.py"))
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), "exec")
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in modules:
                if any(
                    module == bad or module.startswith(bad + ".")
                    for bad in FORBIDDEN_IMPORTS
                ):
                    errors.append(
                        {
                            "file": str(path.relative_to(root)),
                            "line": node.lineno,
                            "forbidden_import": module,
                        }
                    )
            if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(
                ("torch.cuda.", "torch.xpu.", "torch.ops._C.")
            ):
                errors.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": node.lineno,
                        "retired_device_api": ast.unparse(node),
                    }
                )
    report.update(
        scope="P4_static_source_not_build_ABI_or_runtime",
        package=package,
        python_files=len(files),
        version=project["version"],
        passed=not errors,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
