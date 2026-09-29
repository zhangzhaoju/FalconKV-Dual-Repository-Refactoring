# SPDX-License-Identifier: Apache-2.0
"""Compare effective P3 methods with P3-01 bodies after explicit relocation rules."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3] / "p1-repos/LMCache"


def class_methods(source, name):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def check():
    manifest = json.loads((ROOT / "docs/p3-native-migration.json").read_text())
    commit = manifest["input_commit"]
    original = lambda path: subprocess.check_output(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], text=True)
    errors, changed, checked = [], [], 0
    for entry in manifest["method_merges"]:
        path = entry["owner"].replace(".", "/") + ".py"
        npu = path.endswith("npu_connectors.py")
        basepath = "lmcache/v1/gpu_connector/gpu_connectors.py" if npu else path
        donorpath = "ascend/lmcache_ascend/v1/npu_connector/npu_connectors.py" if npu else next(old for old, new in manifest["relocations"].items() if new == path)
        base = class_methods(original(basepath), entry["base"])
        donor = class_methods(original(donorpath), entry["donor"])
        name = entry["donor"] if npu else entry["base"]
        name = "NPUIPCWrapper" if name == "CudaIPCWrapper" else name
        current = class_methods((ROOT / path).read_text(), name)
        desired = {key: (node, True) for key, node in donor.items()}
        for key, node in base.items():
            if key not in donor:
                desired[key] = (node, False)
            if key in entry["common_delegates"]:
                desired[entry["common_delegates"][key]] = (node, False)
        for target, (node, is_donor) in desired.items():
            source = ast.unparse(node)
            source = re.sub(rf"\bdef {node.name}\(", f"def {target}(", source, count=1)
            if is_donor:
                for old, new in entry["common_delegates"].items():
                    source = source.replace(f"super().{old}", f"self.{new}")
                    source = source.replace(f"getattr(super(), '{old}', None)", f"self.{new}")
            replacements = {
                "lmcache_ascend.v1.remote_fill": "lmcache.v1.remote_fill.npu_transport",
                "lmcache_ascend.v1.storage_backend.p2p_backend": "lmcache.v1.storage_backend.npu_p2p_backend",
                "lmcache_ascend.v1.storage_backend.utils": "lmcache.v1.storage_backend.npu_utils",
            }
            for old, new in replacements.items():
                source = re.sub(re.escape(old) + r'(?=[.\s\"\']|$)', new, source)
            source = source.replace("lmcache_ascend.", "lmcache.")
            source = source.replace("from lmcache_ascend import", "from lmcache import")
            source = source.replace("lmcache.v1.gpu_connector.utils", "lmcache.v1.device_connector.utils")
            source = source.replace("lmcache.v1.gpu_connector.sparse", "lmcache.v1.device_connector.sparse")
            symbols = {"AscendLMCacheEngine": "LMCacheEngine", "LMCacheAscendConnectorV1Impl": "LMCacheConnectorV1Impl", "LMCacheAscendConnectorV1Dynamic": "LMCacheConnectorV1Dynamic", "CudaIPCWrapper": "NPUIPCWrapper", "AscendIPCWrapper": "NPUIPCWrapper", "GPUConnectorInterface": "DeviceConnectorInterface"}
            if npu:
                symbols[entry["base"]] = entry["donor"]
            for old, new in symbols.items():
                source = re.sub(rf"\b{old}\b", new, source)
            source = source.replace("torch.cuda", "torch.npu").replace(".is_cuda", ".is_npu")
            source = source.replace("'cuda'", "'npu'").replace("'cuda:", "'npu:")
            expected = ast.parse(source).body[0]
            if ast.dump(expected) != ast.dump(current[target]):
                label = f"{name}.{target}"
                # Explicit-device allocation replaces the old global tensor shim.
                if label == "NPUIPCWrapper.to_tensor":
                    changed.append(label)
                else:
                    errors.append(label)
            checked += 1
    return {"scope": "normalized_method_AST_not_runtime", "input": commit, "checked": checked,
            "reviewed_explicit_device_change": changed, "unexpected_changes": errors, "passed": not errors}


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
