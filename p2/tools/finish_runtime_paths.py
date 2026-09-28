"""Complete native stream/collective ownership and source migration checks."""

import ast
import json
from pathlib import Path

from source_edit import ROOT, add_imports, definition, find, prepend_body, read, replace, replace_text, write

path = "vllm/compilation/graph_types.py"
replace_text(path, "@dataclasses.dataclass", "@dataclass")
path = "vllm/tool_parsers/glm4_moe_tool_parser.py"
replace_text(path, "self: Glm4MoeModelToolParser,", "self,")
path = "vllm/v1/worker/npu_model_runner.py"
target = "vllm/distributed/ascend/parallel_state.py"
add_imports(target, "from contextlib import contextmanager, nullcontext\nfrom dataclasses import dataclass")
write(target, read(target) + "\n\n" + definition(path, "GraphCaptureContext") + "\n\n" + definition(path, "graph_capture"))
replace(path, "GraphCaptureContext", "")
replace(path, "graph_capture", "")

# No process-wide CUDA hook: use the actual selected device stream.
path = "vllm/utils/torch_utils.py"
replace(path, "_patched_set_stream", "")
source = read(path).replace("prev_set_stream = torch.cuda.set_stream\n", "").replace("torch.cuda.set_stream = _patched_set_stream\n", "")
write(path, source)
replace(path, "current_stream", '''
def current_stream():
    """Return the active platform stream without replacing framework functions."""
    from vllm.platforms import current_platform
    if current_platform.is_npu():
        return torch.npu.current_stream()
    if current_platform.is_cuda_alike():
        if not getattr(_current_stream_tls, "initialized", False):
            torch.cuda.set_stream(torch.cuda.Stream())
            _current_stream_tls.initialized = True
        return torch.cuda.current_stream()
    if current_platform.is_cpu():
        return _StreamPlaceholder()
    stream = current_platform.current_stream
    if stream is None:
        raise ValueError("The selected platform does not provide a current stream")
    return stream()
''')
path = "vllm/distributed/parallel_state.py"
prepend_body(path, "graph_capture", '''
if current_platform.is_npu():
    stream = graph_capture_context.stream if graph_capture_context else torch.npu.Stream()
    context = graph_capture_context or GraphCaptureContext(stream)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        yield context
    return
''', "GroupCoordinator")
prepend_body(path, "graph_capture", '''
if current_platform.is_npu():
    from vllm.distributed.ascend.parallel_state import graph_capture as npu_graph_capture
    with npu_graph_capture(device) as context:
        yield context
    return
''')

# The 310P adaptation is called explicitly. Torch's module attributes are never assigned.
patch = read("ascend/legacy_patches/platform/patch_distributed.py")
nodes = list(ast.walk(ast.parse(patch)))
from source_edit import segment
broadcast = segment(patch, next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "broadcast310p"))
all_reduce = segment(patch, next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "all_reduce"))
broadcast = broadcast.replace("def broadcast310p(", "def broadcast(")
broadcast = broadcast.replace("fn(", "torch.distributed.broadcast(")
all_reduce = all_reduce.replace("fn(", "torch.distributed.all_reduce(")
path = "vllm/distributed/ascend/collectives.py"
write(path, '''# SPDX-License-Identifier: Apache-2.0
"""Explicit NPU collective adaptation; CPU and non-310P use torch directly."""
import torch

def _uses_310p():
    from vllm.platforms import current_platform
    if not current_platform.is_npu():
        return False
    from vllm.utils.ascend import is_310p
    return is_310p()

''' + definition("ascend/legacy_patches/platform/patch_distributed.py", "NullHandle") + "\n\n" + broadcast + "\n\n" + all_reduce)
prepend_body(path, "broadcast", '''
if not _uses_310p():
    kwargs = {"src": src, "group": group, "async_op": async_op}
    if group_src is not None:
        kwargs.pop("src")
        kwargs["group_src"] = group_src
    return torch.distributed.broadcast(tensor, **kwargs)
''')
prepend_body(path, "all_reduce", '''
if not _uses_310p():
    return torch.distributed.all_reduce(tensor, op, group, async_op)
''')
native_files = set(json.loads((ROOT.parents[1] / "design/p2/baseline/namespace-migration.json").read_text())["files"][i]["destination"] for i in range(355))
for p in (ROOT / "vllm").rglob("*.py"):
    relative = str(p.relative_to(ROOT))
    if relative == path or not (relative in native_files or relative.startswith("vllm/distributed/") or "/npu/" in relative):
        continue
    source = p.read_text()
    dist_aliases = {"torch.distributed"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            dist_aliases.update(a.asname for a in node.names if a.name == "torch.distributed" and a.asname)
    changed = source
    for alias in dist_aliases:
        for name in ("broadcast", "all_reduce"):
            changed = changed.replace(alias + "." + name + "(", "npu_" + name + "(")
    if changed != source:
        write(relative, changed)
        used = [name for name in ("broadcast", "all_reduce") if "npu_" + name + "(" in changed]
        add_imports(relative, "from vllm.distributed.ascend.collectives import " + ", ".join(name + " as npu_" + name for name in used))

for p in (ROOT / "vllm").rglob("*.py"):
    source = p.read_text()
    updated = source.replace("vllm_C", "_ascend_C")
    if "/ascend/" in str(p):
        updated = updated.replace("vllm.v1.worker.gpu_input_batch", "vllm.v1.worker.input_batch")
    if updated != source:
        write(str(p.relative_to(ROOT)), updated)
path = "tests/standalone/test_p1_development.py"
replace_text(path, '        self.populate(self.staging, "strict-editable")', '        resources = self.populate(self.staging, "strict-editable")')
replace_text(path, '        self.assertGreaterEqual(len(mapping), 8)', '        self.assertEqual(len(mapping), len(set(resources)))')
print("Completed native graph contexts, stream ownership and collective adapters")
