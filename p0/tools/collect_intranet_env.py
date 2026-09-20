# SPDX-License-Identifier: Apache-2.0
"""Collect LOCAL P0 evidence inside the intranet; never contact any service.

Requires Python 3.10+ and only the standard library. No installation, build,
model execution or uploads. Runtime imports and npu-smi require explicit flags.
The output is an intranet original, NOT an automatically sanitized export.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig


EXPECTED = {"torch": "2.9.0", "torch-npu": "2.9.0"}
PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
RUNTIME_PROBE = """
import json
import torch
import torch_npu
print(json.dumps({
    'torch_version': str(torch.__version__),
    'torch_file': torch.__file__,
    'torch_npu_version': str(torch_npu.__version__),
    'torch_npu_file': torch_npu.__file__,
    'cxx11_abi': bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    'npu_available': bool(torch.npu.is_available()),
    'npu_count': int(torch.npu.device_count()),
}))
"""


def run_probe(argv: list[str], timeout: int = 30) -> dict:
    """Run a fixed local diagnostic with timeout; keep errors as evidence."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        return {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error_type": type(exc).__name__}


def package_inventory() -> dict[str, list[str]]:
    """Read installed distribution metadata, without importing any framework."""
    packages: dict[str, list[str]] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            normalized = name.lower().replace("_", "-").replace(".", "-")
            packages.setdefault(normalized, []).append(dist.version)
    return {name: sorted(set(versions)) for name, versions in sorted(packages.items())}


def evaluate_versions(packages: dict[str, list[str]]) -> list[dict]:
    """Report missing/mismatching base versions; never install or change them."""
    result = []
    for name, expected in EXPECTED.items():
        actual = packages.get(name, [])
        okay = len(actual) == 1 and actual[0].split("+")[0] == expected
        result.append(
            {
                "package": name,
                "expected": expected,
                "actual": actual,
                "status": "metadata_matches" if okay else "needs_review",
                "runtime_abi_verified": False,
            }
        )
    return result


def toolkit_inventory(root: Path) -> dict:
    """Read version fields from known CANN metadata locations, not environment dumps."""
    candidates = [
        root / f"{platform.machine()}-linux/ascend_toolkit_install.info",
        root / "ascend_toolkit_install.info",
        root / "version.cfg",
    ]
    found = []
    for path in candidates:
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            versions = [
                line.strip()
                for line in lines
                if line.strip()
                .lower()
                .startswith(("version=", "version =", "version_version="))
            ]
            found.append(
                {
                    "path": str(path),
                    "version_fields": versions,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return {
        "requested_version": "8.5.1",
        "root": str(root),
        "metadata": found,
        "hixl_hcomm_hccl_sdk_abi_status": "requires_intranet_verification",
        "driver_firmware_status": "requires_operator_record",
    }


def model_metadata(spec: str) -> dict:
    """Hash local small model metadata; never load weights, tokenizers or remote code."""
    alias, separator, raw_path = spec.partition("=")
    if not separator or not alias or not raw_path:
        raise ValueError("--model must be alias=/local/model/path")
    root = Path(raw_path).resolve()
    if not root.is_dir():
        raise ValueError(f"Model directory is unavailable for alias: {alias}")
    files = []
    config = {}
    for name in (
        "config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "quant_model_description.json",
    ):
        path = root / name
        if not path.is_file():
            continue
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError(f"Unexpectedly large metadata file: {alias}/{name}")
        data = path.read_bytes()
        files.append(
            {
                "file": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        if name == "config.json":
            config = json.loads(data)
    return {
        "alias": alias,
        "local_path": str(root),
        "metadata_files": files,
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "dtype": config.get("dtype", config.get("torch_dtype")),
        "quantization_config_present": "quantization_config" in config,
        "num_hidden_layers": config.get("num_hidden_layers"),
        "indexer_types_present": "indexer_types" in config,
        "weight_revision_and_hashes": "must_be_filled_from_approved_artifact_manifest",
        "runtime_verified": False,
    }


def collect(args: argparse.Namespace) -> dict:
    """Create local evidence, explicitly distinguishing metadata from runtime probes."""
    packages = package_inventory()
    result = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "intranet_local_original_requires_manual_review_before_export",
        "network_calls": 0,
        "builds_or_installs": 0,
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "host": {
            "os": platform.system(),
            "machine": platform.machine(),
            "kernel": platform.release(),
            "libc": platform.libc_ver(),
        },
        "packages": packages,
        "version_checks": evaluate_versions(packages),
        "proxy_configuration_presence_only": {
            key: bool(os.getenv(key)) for key in PROXY_KEYS
        },
        "ca_configuration_presence_only": {
            key: bool(os.getenv(key))
            for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE")
        },
        "network_policy": {
            "proxy_routing_verified": False,
            "external_llm_access_verified_disabled": False,
            "note": "Operator must review clients, CI, egress policy and NO_PROXY; no connection probes performed",
        },
        "cann": toolkit_inventory(args.cann_root),
        "tools": {
            name: run_probe([name, "--version"])
            if shutil.which(name)
            else {"available": False}
            for name in ("cmake", "ninja", "c++", "git")
        },
        "runtime_probe": {"status": "not_requested"},
        "npu_smi": {"status": "not_requested", "requested_device": "Ascend 910B3"},
        "models": [model_metadata(spec) for spec in args.model],
        "operator_fields_pending": [
            "node_count",
            "cards_per_node",
            "parallelism",
            "driver",
            "firmware",
            "communication_libraries",
            "container_digest",
            "approved_dependency_sources",
            "network_policy_review",
            "checkpoint_revision",
            "quantization_profile",
        ],
        "p0_gate": "pending_intranet_review_and_baseline_tests",
    }
    if args.probe_runtime:
        probe = run_probe([sys.executable, "-B", "-c", RUNTIME_PROBE], timeout=45)
        if probe.get("returncode") == 0:
            try:
                probe["parsed"] = json.loads(probe["stdout"].strip().splitlines()[-1])
            except (ValueError, IndexError):
                probe["parse_status"] = "manual_review_required"
        result["runtime_probe"] = probe
    if args.probe_npu:
        result["npu_smi"] = run_probe(["npu-smi", "info"], timeout=30)
    return result


def main() -> None:
    """Only write the explicitly requested new local report, with private permissions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cann-root",
        type=Path,
        default=Path(
            os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        ),
    )
    parser.add_argument(
        "--model", action="append", default=[], metavar="ALIAS=LOCAL_PATH"
    )
    parser.add_argument("--probe-runtime", action="store_true")
    parser.add_argument("--probe-npu", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new batch/report name")
    result = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        "Local report saved. Do not upload the raw report; review and sanitize permitted fields first."
    )


if __name__ == "__main__":
    main()
