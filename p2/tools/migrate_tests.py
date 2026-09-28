"""Update test imports and host source locations to the native owners."""

import re

from migrate_namespace import PREFIXES, mapped, rewrite
from source_edit import ROOT, read, write, replace, replace_text

for area in (ROOT / "ascend/tests", ROOT / "tests/standalone"):
    for p in area.rglob("*.py"):
        source = p.read_text()
        # Source readers in standalone tests are rooted at ascend/.
        if "ascend/tests/standalone" in str(p):
            for old in sorted(PREFIXES, key=len, reverse=True):
                if old == "vllm_ascend":
                    continue
                old_path = old.replace(".", "/")
                new_path = PREFIXES[old].replace(".", "/")
                source = source.replace(old_path, "../" + new_path)
            source = source.replace('"vllm_ascend/_cann_ops_custom"', '"vllm/_cann_ops_custom"')
            source = source.replace('"vllm/ascend/../vllm/', '"vllm/vllm/')
        if "vllm_ascend" in source:
            source = rewrite(source)
        p.write_text(source)

path = "tests/standalone/test_p1_development.py"
replace_text(path, '        self.assertEqual(len(mapping), len(set(resources)))', '''        expected = set(resources) | {
            f"{namespace}/{name}"
            for namespace in {self.primary, self.addon}
            for name in ("__init__.py", "_version.py")
        } | {f"{self.addon}/_build_info.py", f"{self.addon}/p1_build_info.json"}
        self.assertEqual(
            {str(Path(path).relative_to(self.command.build_lib)) for path in mapping},
            expected,
        )''')
path = "vllm/distributed/parallel_state.py"
replace_text(path, 'self.device = torch.npu.current_device()', 'self.device = torch.device(f"npu:{local_rank}")')

path = "tests/standalone/test_p2_platform.py"
replace(path, "component_stubs", '''
def component_stubs(self) -> None:
    name = "vllm.utils.ascend_profiling_config"
    sys.modules[name] = module(name, generate_service_profiling_config=lambda: self.calls.append("profiling"))
''', "NativePlatformTests")
source = read(path).replace('["connector", "netloader", "rfork", "profiling", "optional"]', '["profiling", "optional"]')
source = source.replace('["connector", "netloader", "rfork", "profiling"]', '["profiling"]')
source = source.replace("forward_oot", "forward_npu")
source = source.replace('enum.Enum("Backend", "OOT CUDA CPU TPU XPU TRITON AITER")', 'enum.Enum("Backend", "NPU OOT CUDA CPU TPU XPU TRITON AITER")')
source = source.replace("backend.OOT", "backend.NPU")
write(path, source)
print("Migrated native module imports and standalone source fixtures")
