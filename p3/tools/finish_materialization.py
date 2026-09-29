# SPDX-License-Identifier: Apache-2.0
"""Mechanical follow-up: shared build helpers, layout owner and wrapper methods.

One-time migration tooling, never invoked by the distributed build backend.
"""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
LMC = ROOT / "p1-repos/LMCache"
VLLM = ROOT / "p1-repos/vllm"


def start(n):
    return min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])


def chunk(s, n):
    return "".join(s.splitlines(keepends=True)[start(n)-1:n.end_lineno])


def replace(s, n, value):
    lines = s.splitlines(keepends=True)
    return "".join(lines[:start(n)-1]) + value + "".join(lines[n.end_lineno:])


def main():
    # Both paired helpers intentionally implement the same distribution layout.
    build = (VLLM / "p1_build.py").read_text()
    build = build.replace('return "vllm" if primary == "vllm" else primary + "_ascend"', 'return primary')
    build = build.replace('"lmcache_ascend": ["c_ops*.so", "libcache_kernels.so", *channels],\n        "lmcache": [', '"lmcache": ["c_ops*.so", "libcache_kernels.so", *channels,')
    build = build.replace('"""P2 native Ascend', '"""P3 native Ascend')
    dev = (VLLM / "p1_dev.py").read_text()
    dev = dev.replace('if primary == "vllm" and any(', 'if any(')
    dev = dev.replace('("vllm_ascend/", "ascend/legacy_patches/", "ascend/legacy_plugin/")', '("vllm_ascend/", "lmcache_ascend/", "ascend/legacy_patches/", "ascend/legacy_plugin/", "ascend/legacy-p3/")')
    dev = dev.replace('P2 wheel contains', 'P3 wheel contains')
    for repo in (LMC, VLLM):
        (repo / "p1_build.py").write_text(build)
        (repo / "p1_dev.py").write_text(dev)

    # The generic layout utility owns tuple handling; avoid an import cycle
    # between the neutral interface and concrete NPU kernels.
    npu_path = LMC / "lmcache/v1/npu_connector/utils.py"
    npu = npu_path.read_text()
    fn = next(n for n in ast.parse(npu).body if isinstance(n, ast.FunctionDef) and n.name == "permute_kv_caches_to_contiguous")
    value = chunk(npu, fn).replace("_KVLayer", "Union[torch.Tensor, Tuple[torch.Tensor, ...]]")
    layout_path = LMC / "lmcache/v1/device_connector/utils.py"
    layout = layout_path.read_text()
    old = next(n for n in ast.parse(layout).body if isinstance(n, ast.FunctionDef) and n.name == fn.name)
    layout_path.write_text(replace(layout, old, value))
    npu_path.write_text(replace(npu, fn, "").replace('from lmcache.v1.device_connector.utils import permute_to_contiguous', 'from lmcache.v1.device_connector.utils import permute_kv_caches_to_contiguous  # noqa: F401'))
    # Append the port utility, not the decorator/monkey patch.
    old_path = LMC / "ascend/legacy-p3/lmcache_ascend/v1/rpc_utils.py"
    rpc = old_path.read_text()
    fn = next(n for n in ast.parse(rpc).body if isinstance(n, ast.FunctionDef) and n.name == "_find_free_port")
    path = LMC / "lmcache/v1/rpc_utils.py"
    path.write_text(path.read_text() + "\n\n" + chunk(rpc, fn))

    # Formal vLLM connector lifecycle uses the effective dynamic wrapper
    # implementation. Keep vLLM's event aggregation and constructor ownership.
    wrapper = (LMC / "lmcache/integration/vllm/lmcache_connector_v1.py").read_text()
    donor = next(n for n in ast.parse(wrapper).body if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1Dynamic")
    path = VLLM / "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"
    source = path.read_text()
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "LMCacheConnectorV1")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    keep = {"__init__", "_common_init", "update_connector_output"}
    additions = []
    replacements = []
    for n in donor.body:
        if isinstance(n, ast.FunctionDef) and n.name not in keep:
            if n.name in methods:
                replacements.append((methods[n.name], chunk(wrapper, n)))
            else:
                additions.append(chunk(wrapper, n))
        elif isinstance(n, ast.Assign):
            additions.append(chunk(wrapper, n))
    # Insert additions first, at the original end of the class.
    lines = source.splitlines(keepends=True)
    source = "".join(lines[:cls.end_lineno]) + "\n\n" + "\n".join(additions) + "".join(lines[cls.end_lineno:])
    for old, value in sorted(replacements, key=lambda item: item[0].lineno, reverse=True):
        source = replace(source, old, value)
    path.write_text(source)
    # The old plugin module is not a compatibility shim in the native package.
    pool = VLLM / "vllm/distributed/kv_transfer/ascend/kv_pool/lmcache_ascend_connector.py"
    archive = VLLM / "ascend/legacy_plugin/lmcache_ascend_connector_p3.py"
    pool.rename(archive)
    factory = VLLM / "vllm/distributed/kv_transfer/kv_connector/factory.py"
    factory.write_text(factory.read_text().replace('vllm.distributed.kv_transfer.ascend.kv_pool.lmcache_ascend_connector', 'vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector'))
    for path in (VLLM / "vllm").rglob("*.py"):
        s = path.read_text()
        new = re.sub(r"\bCudaIPCWrapper\b", "NPUIPCWrapper", s)
        if new != s:
            path.write_text(new)
    # Mechanical factory rename across native package, preserving config keys.
    for path in (LMC / "lmcache").rglob("*.py"):
        s = path.read_text()
        new = re.sub(r"\bis_cuda_worker\b", "is_npu_worker", s)
        if new != s:
            path.write_text(new)


if __name__ == "__main__":
    main()
