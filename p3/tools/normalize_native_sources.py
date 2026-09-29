# SPDX-License-Identifier: Apache-2.0
"""One-time mechanical import consolidation and test namespace relocation."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
LMC = ROOT / "p1-repos/LMCache"


def main():
    for rel in ("lmcache/v1/cache_engine.py", "lmcache/integration/vllm/vllm_v1_adapter.py", "lmcache/integration/vllm/lmcache_connector_v1.py", "lmcache/v1/npu_connector/npu_connectors.py", "lmcache/v1/multiprocess/custom_types.py"):
        path = LMC / rel
        s = path.read_text()
        lines = s.splitlines(keepends=True)
        imports = [n for n in ast.parse(s).body if isinstance(n, (ast.Import, ast.ImportFrom))]
        collected = "".join("".join(lines[n.lineno - 1:n.end_lineno]) for n in imports)
        for n in reversed(imports):
            lines[n.lineno - 1:n.end_lineno] = []
        # Consolidate before eager type annotations, not after common DTOs.
        s = "".join(lines)
        s = s.split("\n", 1)[0] + "\n" + collected + "\n" + s.split("\n", 1)[1]
        ast.parse(s)
        path.write_text(s)
    # Native package directories are ordinary packages for setuptools/PEP 660.
    for sub in ("lmcache/v1/npu_connector", "lmcache/v1/device_connector", "lmcache/v1/compute/npu_blend", "lmcache/v1/storage_backend/pd"):
        root = LMC / sub
        for directory in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
            init = directory / "__init__.py"
            if any(directory.glob("*.py")) and not init.exists():
                init.write_text('# SPDX-License-Identifier: Apache-2.0\n')
    # Keep paired development contracts identical; each tests its own checkout.
    (LMC / "tests/standalone/test_p1_development.py").write_text((ROOT / "p1-repos/vllm/tests/standalone/test_p1_development.py").read_text())
    maps = {
        'lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1': 'lmcache.integration.vllm.lmcache_connector_v1',
        'lmcache_ascend.v1.remote_fill': 'lmcache.v1.remote_fill.npu_transport',
        'lmcache_ascend.v1.blend': 'lmcache.v1.compute.npu_blend',
        'lmcache_ascend.v1.storage_backend.p2p_backend': 'lmcache.v1.storage_backend.npu_p2p_backend',
        'lmcache_ascend.v1.storage_backend.utils': 'lmcache.v1.storage_backend.npu_utils',
        'vllm_ascend.live_source_handoff': 'vllm.distributed.kv_transfer.live_source_handoff',
        'vllm_ascend.lmcache_diagnostics': 'vllm.distributed.kv_transfer.lmcache_diagnostics',
    }
    for sub in ("tests", "ascend/tests", "ascend/benchmark"):
        for path in (LMC / sub).rglob("*.py"):
            if path.name == "bootstrap.py":
                continue
            s = path.read_text()
            for before, after in sorted(maps.items(), key=lambda item: -len(item[0])):
                s = re.sub(re.escape(before) + r'(?=[.\s\"\']|$)', after, s)
            s = s.replace('lmcache_ascend.', 'lmcache.')
            s = s.replace('from lmcache_ascend import', 'from lmcache import')
            s = re.sub(r'^import lmcache_ascend\b.*$', 'import lmcache  # native owner, no import patch', s, flags=re.M)
            s = s.replace('pytest.importorskip("vllm_ascend")', 'pytest.importorskip("vllm")')
            s = s.replace('"lmcache_ascend"', '"lmcache"')
            for before, after in [('AscendLMCacheEngine','LMCacheEngine'), ('LMCacheAscendConnectorV1Impl','LMCacheConnectorV1Impl'), ('LMCacheAscendConnectorV1Dynamic','LMCacheConnectorV1Dynamic'), ('CudaIPCWrapper','NPUIPCWrapper'), ('AscendIPCWrapper','NPUIPCWrapper')]:
                s = re.sub(rf'\b{before}\b', after, s)
            s = s.replace('lmcache.v1.gpu_connector.utils', 'lmcache.v1.device_connector.utils')
            s = s.replace('lmcache.v1.gpu_connector.sparse', 'lmcache.v1.device_connector.sparse')
            # File-based CPU harnesses now locate the canonical product root.
            s = s.replace('Path(__file__).resolve().parents[2] / "lmcache_ascend/', 'Path(__file__).resolve().parents[3] / "lmcache/')
            if s != path.read_text():
                path.write_text(s)
    # Multiprocess adapter event type annotations also need explicit NPU ownership.
    path = LMC / 'lmcache/integration/vllm/vllm_multi_process_adapter.py'
    s = path.read_text().replace('torch.cuda.Event', 'torch.npu.Event')
    s = s.replace('import torch\n', 'import torch\nimport torch_npu  # noqa: F401\n')
    path.write_text(s)


if __name__ == '__main__':
    main()
