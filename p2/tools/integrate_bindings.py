"""Declare native NPU factories and dispatch without mutable OOT registrations."""

import ast
import re
import textwrap

from source_edit import ROOT, add_imports, append_class, definition, find, prepend_body, read, replace, replace_text, write

utils = "vllm/utils/ascend.py"
registration = definition(utils, "register_ascend_customop")
tree = ast.parse(registration)
imports = {a.asname or a.name: (n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
dictionaries = [n for n in ast.walk(tree) if isinstance(n, ast.Dict)]
bindings, overrides = [{k.value: imports[v.id] for k, v in zip(n.keys, n.values)} for n in dictionaries]
path = "vllm/model_executor/layers/ascend/registry.py"
source = '''# SPDX-License-Identifier: Apache-2.0
"""Built-in NPU layer selection. Entries are lazy and never modify OOT state."""
from importlib import import_module
from types import MappingProxyType

'''
source += "NPU_LAYERS = MappingProxyType(" + repr(bindings) + ")\n"
source += "NPU_310P_LAYERS = MappingProxyType(" + repr(overrides) + ")\n"
source += '''
def get_npu_layer_class(name: str) -> type | None:
    from vllm.utils.ascend import is_310p

    entry = NPU_LAYERS.get(name)
    if name == "GateLinear":
        from vllm.config import get_current_vllm_config
        config = get_current_vllm_config().model_config.hf_text_config
        if (getattr(config, "model_type", None) == "glm_moe_dsa"
                or getattr(config, "moe_router_dtype", None) == "float32"):
            entry = ("vllm.model_executor.layers.ascend.fused_moe.gate_linear", "AscendGateLinear")
    if entry is None:
        return None
    if is_310p():
        if name == "MRotaryEmbedding":
            return None
        entry = NPU_310P_LAYERS.get(name, entry)
    return getattr(import_module(entry[0]), entry[1])
'''
write(path, source)
# The former decorator also set the enabled-op name. Declare it on each class.
for name, (module, cls) in {**bindings, "GateLinear": ("vllm.model_executor.layers.ascend.fused_moe.gate_linear", "AscendGateLinear")}.items():
    append_class(module.replace(".", "/") + ".py", cls, f"name = {name!r}")
for name, (module, cls) in overrides.items():
    append_class(module.replace(".", "/") + ".py", cls, f"name = {name!r}")
replace(utils, "register_ascend_customop", "")
replace(utils, "adapt_patch", "")
source = read(utils)
source = re.sub(r"^(_ASCEND_CUSTOMOP_IS_REIGISTERED|REGISTERED_ASCEND_OPS) = .*\n", "", source, flags=re.M)
write(utils, source)

path = "vllm/model_executor/custom_op.py"
for cls in ("CustomOp", "PluggableLayer"):
    prepend_body(path, "__new__", '''
if current_platform.is_npu():
    from vllm.model_executor.layers.ascend.registry import get_npu_layer_class
    return super().__new__(get_npu_layer_class(cls.__name__) or cls)
''', cls)
append_class(path, "CustomOp", '''
def forward_npu(self, *args, **kwargs):
    return self.forward_native(*args, **kwargs)
''')
replace_text(path, '''elif current_platform.is_npu() or current_platform.is_out_of_tree():
            # Preserve Ascend's current op bindings until their P2 migration.
            return self.forward_oot''', '''elif current_platform.is_npu():
            return self.forward_npu
        elif current_platform.is_out_of_tree():
            return self.forward_oot''')
# Static class methods, such as MM attention shape queries, use the same selector.
write(path, read(path) + '''
def get_platform_class_by_name(class_name: str) -> type | None:
    if current_platform.is_npu():
        from vllm.model_executor.layers.ascend.registry import get_npu_layer_class
        return get_npu_layer_class(class_name)
    return get_oot_class_by_name(class_name)
''')
path = "vllm/model_executor/layers/attention/mm_encoder_attention.py"
write(path, read(path).replace("get_oot_class_by_name", "get_platform_class_by_name"))

for folder in ("vllm/model_executor/layers/ascend", "vllm/platforms/ascend_310p", "vllm/model_executor/layers/quantization/ascend"):
    for p in (ROOT / folder).rglob("*.py"):
        write(str(p.relative_to(ROOT)), p.read_text().replace("forward_oot", "forward_npu"))

path = "vllm/model_executor/layers/ascend/__init__.py"
# Import side effects only register real torch operators, at worker initialization.
source = read(path)
for name in ("dummyFusionOp", "register_dummy_fusion_op"):
    replace(path, name, "")
source = read(path)
tree = ast.parse(source)
initializers = [ast.get_source_segment(source, n) for n in tree.body if not isinstance(n, ast.Assign)]
write(path, '''# SPDX-License-Identifier: Apache-2.0
"""Native Ascend operators, initialized explicitly by the NPU worker."""
def initialize_native_ops() -> None:
''' + textwrap.indent("\n".join(initializers), "    ") + "\n")

path = "vllm/v1/worker/npu_worker.py"
source = read(path).replace("from vllm.utils.ascend import register_ascend_customop\n", "")
source = source.replace('''        # register patch for vllm
        from vllm.utils.ascend import adapt_patch

        adapt_patch()

''', "")
source = source.replace("ops.register_dummy_fusion_op()", "ops.initialize_native_ops()")
source = source.replace("        register_ascend_customop(vllm_config)\n", "")
write(path, source)

path = "vllm/platforms/npu.py"
replace(path, "pre_register_and_update", '''
@classmethod
def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
    # Platform, quantization and component factories are declared in-tree.
    config_deprecated_logging()
''', "NPUPlatform")
replace(path, "register_builtin_components", '''
@classmethod
def register_builtin_components(cls) -> None:
    from vllm.utils.ascend_profiling_config import generate_service_profiling_config
    generate_service_profiling_config()
''', "NPUPlatform")

path = "vllm/model_executor/layers/quantization/__init__.py"
replace_text(path, 'QuantizationMethods = Literal[', 'QuantizationMethods = Literal[\n    "ascend",')
prepend_body(path, "get_quantization_config", '''
if current_platform.is_npu() and quantization in ("ascend", "compressed-tensors"):
    from vllm.utils.ascend import is_310p
    if quantization == "ascend":
        if is_310p():
            from vllm.platforms.ascend_310p.quantization.modelslim_config import AscendModelSlimConfig310
            return AscendModelSlimConfig310
        from vllm.model_executor.layers.quantization.ascend.modelslim_config import AscendModelSlimConfig
        return AscendModelSlimConfig
    from vllm.model_executor.layers.quantization.ascend.compressed_tensors_config import AscendCompressedTensorsConfig
    return AscendCompressedTensorsConfig
''')
for folder in ("vllm/model_executor/layers/quantization/ascend", "vllm/platforms/ascend_310p/quantization"):
    for p in (ROOT / folder).rglob("*.py"):
        write(str(p.relative_to(ROOT)), re.sub(r"^@register_quantization_config\([^\n]+\)\n", "", p.read_text(), flags=re.M))

path = "vllm/model_executor/layers/fused_moe/oracle/unquantized.py"
source = read(path).replace('    OOT = "OOT"', '    NPU = "NPU"\n    OOT = "OOT"').replace("    UnquantizedMoeBackend.OOT,", "    UnquantizedMoeBackend.NPU,\n    UnquantizedMoeBackend.OOT,")
source = source.replace("if current_platform.is_npu() or current_platform.is_out_of_tree():", "if current_platform.is_npu():\n        return UnquantizedMoeBackend.NPU, None\n    if current_platform.is_out_of_tree():")
write(path, source)

# Keep connector loading lazy; the factory owns its complete static registration table.
source = read("vllm/distributed/kv_transfer/ascend/__init__.py")
calls = [ast.get_source_segment(source, n) for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "register_connector"]
path = "vllm/distributed/kv_transfer/kv_connector/factory.py"
source = read(path)
tree = ast.parse(source)
lines = source.splitlines(True)
for node in reversed(tree.body):
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.args and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == "MultiConnector":
        lines[node.lineno - 1:node.end_lineno] = []
write(path, "".join(lines) + "\n\n# Built-in NPU connectors. No connector module is imported by registration.\n" + "\n\n".join(calls) + "\n")
write("vllm/distributed/kv_transfer/ascend/__init__.py", '# SPDX-License-Identifier: Apache-2.0\n"""Native NPU KV transfer implementations; factory registration is in kv_connector.factory."""\n')

loaders = {}
for p in (ROOT / "vllm/model_executor/model_loader/ascend").rglob("*.py"):
    source = p.read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id == "register_model_loader":
                    loaders[dec.args[0].value] = (str(p.relative_to(ROOT)).removesuffix(".py").replace("/", "."), node.name)
    write(str(p.relative_to(ROOT)), re.sub(r"^@register_model_loader\([^\n]+\)\n", "", source, flags=re.M))
path = "vllm/model_executor/model_loader/__init__.py"
prepend_body(path, "get_model_loader", f'''
native_loaders = {loaders!r}
if load_config.load_format in native_loaders:
    from importlib import import_module
    module, name = native_loaders[load_config.load_format]
    return getattr(import_module(module), name)(load_config)
''')

# GPU-only matcher constants must not force CUDA kernels to exist during native compiler import.
# Native passes declare torch_npu/_C_ascend patterns; unsupported GPU matchers remain explicit errors.
path = "vllm/compilation/passes/fusion/matcher_utils.py"
source = read(path)
begin = source.index("RMS_OP =")
end = source.index("\n\nclass MatcherCustomOp")
block = source[begin:end]
source = source[:begin] + 'if not current_platform.is_npu():\n' + textwrap.indent(block, '    ') + source[end:]
write(path, source)
prepend_body(path, "__init__", '''
if current_platform.is_npu():
    raise RuntimeError("CUDA fusion matchers are not used by the native NPU compiler")
''', "MatcherCustomOp")

# Fold the sampling helpers at their call sites instead of replacing module functions.
path = "vllm/v1/sample/rejection_sampler.py"
for name in ("apply_sampling_constraints", "expand_batch_to_tokens", "rejection_sample"):
    source = read(path)
    node = find(source, name)
    args = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
    prepend_body(path, name, "if current_platform.is_npu():\n    from vllm.v1.sample.ascend.rejection_sampler import " + name + " as native_impl\n    return native_impl(" + ", ".join(f"{a}={a}" for a in args) + ")\n")
if "from vllm.platforms import current_platform" not in read(path):
    add_imports(path, "from vllm.platforms import current_platform")
path = "vllm/v1/sample/ops/logprobs.py"
prepend_body(path, "batched_count_greater_than", "if current_platform.is_npu():\n    return (x >= values).sum(-1)\n")
if "from vllm.platforms import current_platform" not in read(path):
    add_imports(path, "from vllm.platforms import current_platform")

print("Integrated native layer, quantization, loader, connector and sampling factories")
