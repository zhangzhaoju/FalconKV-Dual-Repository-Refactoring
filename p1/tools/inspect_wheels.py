# SPDX-License-Identifier: Apache-2.0
"""Inspect paired P1 wheels without installing/importing them (not an ABI test)."""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import hashlib
import json
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

REQUIRED = {
    "vllm": [
        "vllm/__init__.py",
        "vllm/_version.py",
        "vllm_ascend/__init__.py",
        "vllm_ascend/_version.py",
        "vllm_ascend/_build_info.py",
        "vllm_ascend/p1_build_info.json",
        "vllm_ascend/vllm_ascend_C*.so",
        "vllm_ascend/libvllm_ascend_kernels.so",
        "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/*.so",
    ],
    "lmcache": [
        "lmcache/__init__.py",
        "lmcache/_version.py",
        "lmcache_ascend/__init__.py",
        "lmcache_ascend/_version.py",
        "lmcache_ascend/_build_info.py",
        "lmcache_ascend/p1_build_info.json",
        "lmcache_ascend/c_ops*.so",
        "lmcache_ascend/libcache_kernels.so",
        "lmcache_ascend/hixl_npu_comms*.so",
        "lmcache_ascend/hcomm_onesided*.so",
        "lmcache/native_storage_ops*.so",
        "lmcache/lmcache_fs*.so",
        "lmcache/lmcache_redis*.so",
    ],
}
VERSIONS = {"vllm": "0.18.0+ascend.p1", "lmcache": "0.4.3+ascend.p1"}
FORBIDDEN = {
    "cufile-python",
    "cupy-cuda12x",
    "nvtx",
    "nixl",
    "triton",
    "vllm-ascend",
    "lmcache-ascend",
    "vllm-flash-attn",
    "flash-attn",
}


def inspect(path: Path, name: str) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)) or wheel.testzip() is not None:
            raise ValueError(f"Invalid or duplicate wheel entries: {path.name}")
        if any(
            PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts
            for n in names
        ):
            raise ValueError("Unsafe wheel member")
        metadata_files = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("Each wheel must own exactly one distribution")
        project = BytesParser().parsebytes(wheel.read(metadata_files[0]))
        if project["Name"].lower() != name or project["Version"] != VERSIONS[name]:
            raise ValueError("Unexpected P1 distribution identity")
        info_dir = metadata_files[0].rsplit("/", 1)[0]
        wheel_info = BytesParser().parsebytes(wheel.read(info_dir + "/WHEEL"))
        if wheel_info["Root-Is-Purelib"].lower() != "false":
            raise ValueError("Ascend native wheel must not be pure Python")
        tags = wheel_info.get_all("Tag", [])
        if not tags or any(
            not tag.startswith("cp311-") or "aarch64" not in tag for tag in tags
        ):
            raise ValueError("P1 candidate must be a CPython 3.11 aarch64 wheel")
        for pattern in REQUIRED[name]:
            if not any(fnmatch.fnmatchcase(n, pattern) for n in names):
                raise ValueError(f"Missing required wheel resource: {pattern}")
        deps = project.get_all("Requires-Dist", [])
        for dep in deps:
            dependency = dep.split(";", 1)[0].split("[", 1)[0]
            dependency = dependency.split(" ", 1)[0].split("(", 1)[0]
            for operator in "=<>!~":
                dependency = dependency.split(operator, 1)[0]
            dependency = dependency.lower().replace("_", "-")
            if dependency in FORBIDDEN or dependency.startswith("nvidia-"):
                raise ValueError(f"Unexpected device/plugin dependency: {dep}")
        if name == "vllm":
            entries = configparser.ConfigParser()
            entries.read_string(wheel.read(info_dir + "/entry_points.txt").decode())
            if entries["vllm.platform_plugins"]["ascend"] != "vllm_ascend:register":
                raise ValueError(
                    "The vllm distribution must own the Ascend entry point"
                )
    return {
        "distribution": name,
        "version": VERSIONS[name],
        "wheel": path.name,
        "sha256": digest,
        "wheel_contents_passed": True,
        "abi_tested": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-wheel", required=True, type=Path)
    parser.add_argument("--lmcache-wheel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = [inspect(args.vllm_wheel, "vllm"), inspect(args.lmcache_wheel, "lmcache")]
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
