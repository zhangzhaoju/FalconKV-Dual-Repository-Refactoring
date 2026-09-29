# SPDX-License-Identifier: Apache-2.0
"""One-shot, source-preserving P3 relocation from the frozen P3-01 input.

This is a migration record, not an installation hook. It never runs from setup.
Method bodies retain their comments; an explicit method map records overridden
common methods retained for former super() calls. Hand-reviewed semantic edits
are applied after this mechanical step and must not be overwritten by rerunning.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[3] / "p1-repos/LMCache"
INPUT = "470fd4dc6c566bc7b878ba425bc181a155e945ec"
OUTPUT: dict[str, str] = {}
METHODS: list[dict] = []
PATHS: dict[str, str] = {}


def original(path: str) -> str:
    """Read only committed migration inputs, independent of output order."""
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", f"{INPUT}:{path}"], text=True
    )


def span(node: ast.AST) -> int:
    """Include decorators when copying a definition."""
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def piece(source: str, node: ast.AST) -> str:
    """Copy one complete definition with its original indentation."""
    return "".join(source.splitlines(keepends=True)[span(node) - 1 : node.end_lineno])


def replace_node(source: str, node: ast.AST, value: str) -> str:
    """Replace complete definition lines, not the surrounding source."""
    lines = source.splitlines(keepends=True)
    return "".join(lines[: span(node) - 1]) + value + "".join(lines[node.end_lineno :])


def class_node(source: str, name: str) -> ast.ClassDef:
    return next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)


def definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def combine_class(base_source: str, base_name: str, child_source: str, child_name: str, owner: str) -> str:
    """Flatten exactly one known class, retaining explicit common delegation."""
    base = class_node(base_source, base_name)
    child = class_node(child_source, child_name)
    base_members = {definition_name(n): n for n in base.body if definition_name(n)}
    child_members = {definition_name(n): n for n in child.body if definition_name(n)}
    child_text = piece(child_source, child)
    delegates = set(re.findall(r"super\(\)\.([A-Za-z_][A-Za-z_0-9]*)", child_text))
    delegates.update(re.findall(r'getattr\(super\(\), "([A-Za-z_][A-Za-z_0-9]*)", None\)', child_text))
    if delegates - base_members.keys():
        raise RuntimeError(f"Unresolved parent delegation: {owner}: {delegates - base_members.keys()}")
    result = f"class {base_name}"
    if base.bases:
        result += "(" + ", ".join(ast.unparse(b) for b in base.bases) + ")"
    result += ":\n"
    retained = {}
    for n in base.body:
        name = definition_name(n)
        if name in child_members:
            if name not in delegates:
                continue
            common = "_common_" + name.strip("_")
            retained[name] = common
            body = piece(base_source, n)
            body = re.sub(rf"\bdef {re.escape(name)}\(", f"def {common}(", body, count=1)
        else:
            body = piece(base_source, n)
        result += body + "\n"
    for n in child.body:
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            continue
        body = piece(child_source, n)
        for name in delegates:
            target = retained.get(name, name)
            body = body.replace(f"super().{name}", f"self.{target}")
            body = body.replace(f'getattr(super(), "{name}", None)', f"self.{target}")
        result += body + "\n"
    # Bare parent references in inherited classmethods refer to the canonical owner.
    result = re.sub(rf"\b{child_name}\b", base_name, result)
    METHODS.append({"owner": owner, "base": base_name, "donor": child_name,
                    "base_methods": sorted(n for n in base_members if isinstance(base_members[n], ast.FunctionDef)),
                    "donor_methods": sorted(n for n in child_members if isinstance(child_members[n], ast.FunctionDef)),
                    "common_delegates": retained})
    return result


def donor_prelude(source: str, skip_classes: set[str], self_module: str) -> str:
    """Copy donor imports and helpers, excluding self-imports and duplicate logger."""
    result = ""
    for n in ast.parse(source).body:
        if isinstance(n, ast.ClassDef) and n.name in skip_classes:
            continue
        if isinstance(n, ast.ImportFrom) and n.module == self_module:
            continue
        if definition_name(n) == "logger":
            continue
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant):
            continue
        result += piece(source, n) + "\n"
    return result


def merge(base_path: str, base_name: str, donor_path: str, donor_name: str) -> None:
    base, donor = original(base_path), original(donor_path)
    cls = class_node(base, base_name)
    owner = base_path.removesuffix(".py").replace("/", ".")
    merged = combine_class(base, base_name, donor, donor_name, owner)
    prelude = donor_prelude(donor, {donor_name}, owner)
    # Self imports in donor are gone; insert helpers after base's shared DTOs.
    OUTPUT[base_path] = replace_node(base, cls, prelude + "\n" + merged)
    PATHS[donor_path] = base_path


def inject_method(base_path: str, class_name: str, method: str, donor_path: str, function: str, decorator: str = "") -> None:
    source = OUTPUT.get(base_path, original(base_path))
    cls = class_node(source, class_name)
    old = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    donor = original(donor_path)
    func = next(n for n in ast.parse(donor).body if isinstance(n, ast.FunctionDef) and n.name == function)
    value = piece(donor, func).replace(f"def {function}(", f"def {method}(", 1)
    OUTPUT[base_path] = replace_node(source, old, textwrap.indent(decorator + value, "    "))
    PATHS[donor_path] = base_path


def main() -> None:
    """Apply only to the clean paired P3-01 LMCache checkout."""
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
    if head != INPUT or dirty:
        raise SystemExit("One-shot migration requires the clean P3-01 input; do not overwrite subsequent edits")
    merge("lmcache/v1/cache_engine.py", "LMCacheEngine", "ascend/lmcache_ascend/v1/cache_engine.py", "AscendLMCacheEngine")
    merge("lmcache/integration/vllm/vllm_v1_adapter.py", "LMCacheConnectorV1Impl", "ascend/lmcache_ascend/integration/vllm/vllm_v1_adapter.py", "LMCacheAscendConnectorV1Impl")
    merge("lmcache/integration/vllm/lmcache_connector_v1.py", "LMCacheConnectorV1Dynamic", "ascend/lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py", "LMCacheAscendConnectorV1Dynamic")
    merge("lmcache/v1/multiprocess/custom_types.py", "CudaIPCWrapper", "ascend/lmcache_ascend/v1/multiprocess/custom_types.py", "AscendIPCWrapper")

    gpu_path = "lmcache/v1/gpu_connector/gpu_connectors.py"
    gpu = original(gpu_path)
    donor_path = "ascend/lmcache_ascend/v1/npu_connector/npu_connectors.py"
    donor = original(donor_path)
    names = [("VLLMBufferLayerwiseGPUConnector", "VLLMBufferLayerwiseNPUConnector"),
             ("VLLMPagedMemGPUConnectorV2", "VLLMPagedMemNPUConnectorV2"),
             ("VLLMPagedMemLayerwiseGPUConnector", "VLLMPagedMemLayerwiseNPUConnector"),
             ("SGLangGPUConnector", "SGLangNPUConnector"),
             ("SGLangLayerwiseGPUConnector", "SGLangLayerwiseNPUConnector")]
    # Preserve donor helper/class ordering (including dataclasses used in annotations).
    native = donor
    for base_name, child_name in reversed(names):
        native = replace_node(native, class_node(native, child_name),
                              combine_class(gpu, base_name, donor, child_name, "lmcache.v1.npu_connector.npu_connectors").replace(base_name, child_name))
    gpu_imports = "".join(piece(gpu, n) + "\n" for n in ast.parse(gpu).body if isinstance(n, (ast.Import, ast.ImportFrom)))
    native = gpu_imports + "\n" + native
    # Remove original GPU parent import. Concrete NPU classes directly implement the interface.
    for n in reversed(ast.parse(native).body):
        if isinstance(n, ast.ImportFrom) and n.module == "lmcache.v1.gpu_connector.gpu_connectors":
            native = replace_node(native, n, "")
    native = '# SPDX-License-Identifier: Apache-2.0\nfrom lmcache.v1.device_connector import DeviceConnectorInterface\n' + native
    OUTPUT["lmcache/v1/npu_connector/npu_connectors.py"] = native
    PATHS[donor_path] = "lmcache/v1/npu_connector/npu_connectors.py"
    interface = piece(gpu, class_node(gpu, "GPUConnectorInterface"))
    OUTPUT["lmcache/v1/device_connector/__init__.py"] = (
        '# SPDX-License-Identifier: Apache-2.0\n"""Device-neutral KV transfer contract; no accelerator implementation import."""\n'
        'import abc\nfrom typing import List, Optional\nimport torch\n'
        'from lmcache.v1.memory_management import MemoryObj\n'
        'from lmcache.v1.device_connector.utils import permute_kv_caches_to_contiguous\n\n' + interface)
    for path in (ROOT / "lmcache/v1/gpu_connector").glob("*.py"):
        if path.name in {"utils.py", "sparse.py", "mock_gpu_connector.py"}:
            OUTPUT[f"lmcache/v1/device_connector/{path.name}"] = original(str(path.relative_to(ROOT)))

    # Helpers whose effective patched implementations now belong to their owner.
    inject_method("lmcache/v1/token_database.py", "TokenDatabase", "_hash_tokens", "ascend/lmcache_ascend/v1/tokens_hash.py", "_hash_tokens")
    inject_method("lmcache/v1/token_database.py", "SegmentTokenDatabase", "process_tokens", "ascend/lmcache_ascend/v1/token_database.py", "TokenDatabase_process_tokens")
    inject_method("lmcache/v1/kv_layer_groups.py", "KVLayerGroupsManager", "build_kv_layer_groups", "ascend/lmcache_ascend/v1/kv_layer_groups.py", "build_kv_layer_groups")
    inject_method("lmcache/v1/kv_layer_groups.py", "KVLayerGroupInfo", "hidden_dim_size", "ascend/lmcache_ascend/v1/kv_layer_groups.py", "patched_hidden_dim_size", "@property\n")
    kv_source = original("ascend/lmcache_ascend/v1/kv_layer_groups.py")
    helpers = "\n".join(piece(kv_source, n) for n in ast.parse(kv_source).body if isinstance(n, ast.FunctionDef) and n.name.startswith("_get_"))
    OUTPUT["lmcache/v1/kv_layer_groups.py"] += "\n\n" + helpers
    inject_method("lmcache/v1/system_detection.py", "NUMADetector", "_read_from_sys", "ascend/lmcache_ascend/v1/system_detection.py", "_read_from_sys", "@staticmethod\n")

    skip = set(PATHS) | {"ascend/lmcache_ascend/__init__.py", "ascend/lmcache_ascend/integration/vllm/utils.py", "ascend/lmcache_ascend/v1/memory_management.py", "ascend/lmcache_ascend/v1/rpc_utils.py", "ascend/lmcache_ascend/v1/lookup_client/lmcache_lookup_client.py"}
    module_map = {
        "lmcache_ascend.v1.remote_fill": "lmcache.v1.remote_fill.npu_transport",
        "lmcache_ascend.v1.blend": "lmcache.v1.compute.npu_blend",
        "lmcache_ascend.v1.storage_backend.p2p_backend": "lmcache.v1.storage_backend.npu_p2p_backend",
        "lmcache_ascend.v1.storage_backend.utils": "lmcache.v1.storage_backend.npu_utils",
    }
    for path in sorted((ROOT / "ascend/lmcache_ascend").rglob("*.py")):
        rel = str(path.relative_to(ROOT))
        if rel in skip or any(s in rel for s in ("/mindspore/", "/integration/patch/", "/integration/sglang/")):
            continue
        suffix = rel.removeprefix("ascend/lmcache_ascend/")
        if suffix in {"v1/storage_backend/connector/__init__.py", "v1/multiprocess/server.py", "v1/multiprocess/__init__.py", "v1/__init__.py", "integration/vllm/__init__.py"}:
            continue
        target = "lmcache/" + suffix
        for before, after in module_map.items():
            old = before.replace("lmcache_ascend.", "lmcache.").replace(".", "/")
            if target == old + ".py" or target.startswith(old + "/"):
                target = target.replace(old, after.replace(".", "/"), 1)
                break
        if (ROOT / target).exists() and suffix not in {"v1/storage_backend/__init__.py", "v1/transfer_channel/__init__.py"}:
            raise RuntimeError(f"Unreviewed relocation collision: {rel} -> {target}")
        value = original(rel)
        if suffix == "v1/storage_backend/__init__.py":
            old = original(target)
            fn = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == "storage_plugin_launcher")
            value += "\n" + piece(old, fn)
            value = value.replace("from lmcache.v1.storage_backend import storage_plugin_launcher\n", "")
        OUTPUT[target] = value
        PATHS[rel] = target

    # Rewrite all first-party references, never the preserved event-handoff wire string.
    for path in (ROOT / "lmcache").rglob("*.py"):
        rel = str(path.relative_to(ROOT))
        if "/gpu_connector/" not in rel:
            OUTPUT.setdefault(rel, original(rel))
    for rel, source in list(OUTPUT.items()):
        for old, new in sorted(module_map.items(), key=lambda item: -len(item[0])):
            # Exact module prefix: do not rewrite remote_fill_coordinator/producer.
            source = re.sub(re.escape(old) + r"(?=[.\s\"']|$)", new, source)
        source = source.replace("lmcache_ascend.", "lmcache.")
        source = source.replace("from lmcache_ascend import", "from lmcache import")
        source = source.replace("lmcache.v1.gpu_connector.gpu_connectors", "lmcache.v1.npu_connector.npu_connectors")
        source = source.replace("lmcache.v1.gpu_connector", "lmcache.v1.device_connector")
        # The interface has its own neutral owner; do not import concrete NPU classes for types.
        source = source.replace("from lmcache.v1.npu_connector.npu_connectors import GPUConnectorInterface", "from lmcache.v1.device_connector import DeviceConnectorInterface")
        for before, after in [("GPUConnectorInterface", "DeviceConnectorInterface"), ("CudaIPCWrapper", "NPUIPCWrapper"), ("AscendIPCWrapper", "NPUIPCWrapper"), ("AscendLMCacheEngine", "LMCacheEngine"), ("LMCacheAscendConnectorV1Impl", "LMCacheConnectorV1Impl")]:
            source = re.sub(rf"\b{before}\b", after, source)
        source = source.replace("from lmcache.v1.device_connector import CreateGPUConnector", "from lmcache.v1.npu_connector import CreateNPUConnector")
        source = re.sub(r"\bCreateGPUConnector\b", "CreateNPUConnector", source)
        # Device API conversion is limited to modules reachable through native owners.
        if "torch.cuda" in source and (rel.startswith("lmcache/v1/npu_connector/") or rel in {
            "lmcache/v1/cache_engine.py", "lmcache/v1/memory_management.py", "lmcache/v1/storage_backend/storage_manager.py", "lmcache/v1/storage_backend/local_cpu_backend.py", "lmcache/v1/storage_backend/local_disk_backend.py", "lmcache/v1/storage_backend/remote_backend.py", "lmcache/v1/storage_backend/abstract_backend.py", "lmcache/v1/storage_backend/pd_backend.py", "lmcache/v1/storage_backend/p2p_backend.py", "lmcache/v1/storage_backend/connector/mooncakestore_connector.py", "lmcache/v1/system_detection.py", "lmcache/integration/vllm/vllm_v1_adapter.py", "lmcache/integration/vllm/multi_process_adapter.py", "lmcache/v1/device_connector/utils.py"}):
            source = source.replace("torch.cuda", "torch.npu")
            source = source.replace('"cuda"', '"npu"').replace('"cuda:', '"npu:').replace("'cuda'", "'npu'")
            source = source.replace(".is_cuda", '.is_npu')
        if "torch.npu" in source and "import torch_npu" not in source:
            source = source.replace("import torch\n", "import torch\nimport torch_npu  # noqa: F401\n", 1)
        if not source.startswith("# SPDX-License-Identifier:"):
            source = "# SPDX-License-Identifier: Apache-2.0\n" + source
        ast.parse(source)
        OUTPUT[rel] = source
    for rel, value in OUTPUT.items():
        path = ROOT / rel
        if not path.exists() or path.read_text() != value:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value)
    # Keep original plugin source as reference-only provenance, excluded by packaging.
    archive = ROOT / "ascend/legacy-p3"
    archive.mkdir(exist_ok=True)
    (ROOT / "ascend/lmcache_ascend").rename(archive / "lmcache_ascend")
    (ROOT / "lmcache/v1/gpu_connector").rename(archive / "gpu_connector")
    record = ROOT / "docs/p3-native-migration.json"
    record.write_text(json.dumps({"input_commit": INPUT, "relocations": PATHS, "method_merges": METHODS}, indent=2) + "\n")
    print(f"Materialized {len(PATHS)} owners and {len(METHODS)} method-level merges; semantic review follows")


if __name__ == "__main__":
    main()
