"""One-shot extraction of the inherited P1 Runner contract into native owners."""

import ast
import hashlib
import json
import re
import textwrap

from source_edit import ROOT, add_imports, definition, find, read, replace, segment, start, write

RECORDS = []
HEADER = "# SPDX-License-Identifier: Apache-2.0\n# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n"


def native(source):
    return (source.replace("torch.cuda.CUDAGraph", "torch.npu.NPUGraph")
            .replace("torch.cuda", "torch.npu")
            .replace("torch.accelerator", "torch.npu")
            .replace("torch.Event", "torch.npu.Event"))


def unwrap(path, names):
    source = read(path)
    lines = source.splitlines(True)
    edits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.With) and all(
            isinstance(i.context_expr, ast.Call)
            and isinstance(i.context_expr.func, ast.Name)
            and i.context_expr.func.id in names for i in node.items
        ):
            body = "".join(lines[node.body[0].lineno - 1:node.end_lineno])
            edits.append((node.lineno - 1, node.end_lineno,
                          textwrap.indent(textwrap.dedent(body), " " * node.col_offset)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            edits.append((start(node), node.end_lineno, ""))
    for first, last, replacement in sorted(edits, reverse=True):
        lines[first:last] = [replacement]
    write(path, "".join(lines))


def extract(parent_path, child_path, dest, parent_name, new_name, explicit):
    source = read(parent_path)
    parent = find(source, parent_name)
    child = find(read(child_path), "NPUModelRunner")
    overridden = {n.name for n in child.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    selected = [n for n in parent.body if getattr(n, "name", None) not in overridden or getattr(n, "name", None) in explicit]
    bases = ", ".join(ast.unparse(b) for b in parent.bases)
    # Keep preamble/helper declarations; prune unused imports with Ruff after extraction.
    preamble = "".join(source.splitlines(True)[:start(parent)])
    body = "\n\n".join(segment(source, n) for n in selected)
    output = preamble + f"class {new_name}({bases}):\n" + textwrap.indent(body, "    ") + "\n"
    write(dest, native(re.sub(r"\b" + parent_name + r"\b", new_name, output)))
    RECORDS.append({"source": parent_path, "destination": dest,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "methods": [n.name for n in selected if isinstance(n, ast.FunctionDef)],
                    "explicit_parent_calls": sorted(explicit)})


# Device-independent request bookkeeping is shared by both first-generation Runners.
if "class InputBatch" in read("vllm/v1/worker/gpu_input_batch.py"):
    write("vllm/v1/worker/input_batch.py", read("vllm/v1/worker/gpu_input_batch.py"))
write("vllm/v1/worker/gpu_input_batch.py", HEADER +
      '"""Compatibility exports; request state belongs to worker.input_batch."""\n'
      'from vllm.v1.worker.input_batch import *  # noqa: F403\n')

output = HEADER + "from collections.abc import Callable\nimport torch\nfrom vllm.sequence import IntermediateTensors\nfrom vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput, LogprobsTensors, PoolerOutput\n"
for name in ("AsyncGPUModelRunnerOutput", "_copy_pooler_output_to_cpu", "AsyncGPUPoolingModelRunnerOutput"):
    output += "\n\n" + definition("vllm/v1/worker/gpu_model_runner.py", name)
output += "\n\n" + definition("vllm/v1/worker/gpu_worker.py", "AsyncIntermediateTensors")
output = native(output).replace("AsyncGPU", "AsyncNPU")
write("vllm/v1/worker/runner_output.py", output)

write("vllm/compilation/graph_types.py", HEADER + "from dataclasses import dataclass\n\n" +
      definition("vllm/compilation/cuda_graph.py", "CUDAGraphStat"))

v1 = "vllm/v1/worker/npu_model_runner.py"
state1 = "vllm/v1/worker/npu_runner_state.py"
extract("vllm/v1/worker/gpu_model_runner.py", v1, state1, "GPUModelRunner", "NPUModelRunnerState",
        {"__init__", "_update_states", "profile_run", "_check_and_update_cudagraph_mode", "capture_model", "profile_cudagraph_memory"})
# Helpers have a single native definition, separate from execution state.
for name in ("AsyncGPUModelRunnerOutput", "_copy_pooler_output_to_cpu", "AsyncGPUPoolingModelRunnerOutput"):
    replace(state1, name, "")
add_imports(state1, "from vllm.v1.worker.runner_output import AsyncNPUModelRunnerOutput, AsyncNPUPoolingModelRunnerOutput\nfrom vllm.distributed.ascend.parallel_state import graph_capture\nfrom vllm.compilation.ascend.acl_graph import ACLGraphWrapper\nfrom vllm.compilation.graph_types import CUDAGraphStat")
source = read(state1).replace("AsyncGPU", "AsyncNPU").replace("CUDAGraphWrapper", "ACLGraphWrapper")
source = source.replace("from vllm.compilation.cuda_graph import CUDAGraphStat, ACLGraphWrapper\n", "")
source = source.replace("    graph_capture,\n", "")
# The native execution path already owns microbatching; unwrapping only needs its ACL wrapper.
source = source.replace("from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper\n", "")
source = source.replace("(ACLGraphWrapper, UBatchWrapper)", "ACLGraphWrapper")
write(state1, source)
unwrap(v1, {"_torch_cuda_wrapper", "_replace_gpu_model_runner_function_wrapper"})
source = read(v1).replace("from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput, GPUModelRunner",
                         "from vllm.v1.worker.runner_output import AsyncNPUModelRunnerOutput\nfrom vllm.v1.worker.npu_runner_state import NPUModelRunnerState")
source = source.replace("GPUModelRunner", "NPUModelRunnerState").replace("AsyncNPUModelRunnerStateOutput", "AsyncNPUModelRunnerOutput")
write(v1, source)

v2 = "vllm/v1/worker/npu/v2/model_runner.py"
state2 = "vllm/v1/worker/npu/v2/runner_state.py"
extract("vllm/v1/worker/gpu/model_runner.py", v2, state2, "GPUModelRunner", "NPUModelRunnerState", {"__init__", "postprocess"})
unwrap(v2, {"torch_cuda_wrapper"})
source = read(v2).replace("from vllm.v1.worker.gpu.model_runner import GPUModelRunner", "from vllm.v1.worker.npu.v2.runner_state import NPUModelRunnerState")
source = source.replace("GPUModelRunner", "NPUModelRunnerState")
source = source.replace("from vllm.v1.worker.npu.v2.utils import torch_cuda_wrapper\n", "")
write(v2, source)

# Copy only the transitive component dependencies, never the GPU Runner/worker.
old = "vllm.v1.worker.gpu"
new = "vllm.v1.worker.npu.v2.common"
queue = [ROOT / state1, ROOT / state2, *(ROOT / "vllm/v1/worker/npu/v2").rglob("*.py")]
done = set()
while queue:
    p = queue.pop()
    for node in ast.walk(ast.parse(p.read_text())):
        modules = []
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
            modules.extend(node.module + "." + a.name for a in node.names)
        elif isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        for mod in modules:
            if not mod.startswith(old + ".") or mod in done:
                continue
            rel = mod.replace(".", "/")
            candidate = ROOT / (rel + ".py")
            if not candidate.is_file():
                candidate = ROOT / rel / "__init__.py"
            if not candidate.is_file():
                continue
            assert not mod.endswith((".model_runner", ".worker")), mod
            done.add(mod)
            dest = str(candidate.relative_to(ROOT)).replace("vllm/v1/worker/gpu/", "vllm/v1/worker/npu/v2/common/")
            write(dest, native(candidate.read_text()))
            RECORDS.append({"source": str(candidate.relative_to(ROOT)), "destination": dest,
                            "source_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest()})
            queue.append(candidate)

for p in (ROOT / "vllm/v1/worker/npu/v2/common").rglob("*.py"):
    for directory in p.parents:
        if directory == ROOT / "vllm":
            break
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text(HEADER)

for p in (ROOT / "vllm").rglob("*.py"):
    # Restrict changes to native owners and their extracted dependencies.
    s = p.read_text()
    if p in {ROOT / v1, ROOT / state1, ROOT / "vllm/v1/worker/npu_worker.py", ROOT / "vllm/v1/worker/npu_input_batch.py"} or "/ascend/" in str(p) or "/npu/" in str(p):
        s = s.replace(old + ".", new + ".")
        s = s.replace("vllm.v1.worker.gpu_input_batch", "vllm.v1.worker.input_batch")
        s = s.replace("from vllm.v1.worker.gpu_worker import AsyncIntermediateTensors", "from vllm.v1.worker.runner_output import AsyncIntermediateTensors")
        s = s.replace("from vllm.compilation.cuda_graph import CUDAGraphStat", "from vllm.compilation.graph_types import CUDAGraphStat")
        s = s.replace("ModelCudaGraphManager", "ModelGraphManager")
        write(str(p.relative_to(ROOT)), s)

# Native class construction replaces the former assignments into GPU module globals.
source = read(state2)
source = source.replace(f"from {new}.block_table import BlockTables", "from vllm.v1.worker.npu.v2.block_table import AscendBlockTables as BlockTables")
source = source.replace(f"from {new}.model_states import init_model_state", "from vllm.v1.worker.npu.v2.model_states import init_asecnd_model_state as init_model_state")
source = source.replace("    InputBatch,\n", "")
write(state2, source)
add_imports(state2, "from vllm.v1.worker.npu.v2.input_batch import AscendInputBatch as InputBatch")
graph = "vllm/v1/worker/npu/v2/common/cudagraph_utils.py"
source = read(graph).replace(f"from {new}.input_batch import InputBatch, InputBuffers", f"from {new}.input_batch import InputBuffers\nfrom vllm.v1.worker.npu.v2.input_batch import AscendInputBatch as InputBatch")
write(graph, source)

buffer = "vllm/v1/worker/npu/v2/common/buffer_utils.py"
patch = "ascend/legacy_patches/worker/patch_v2/patch_uva.py"
replace(buffer, "UvaBuffer", definition(patch, "UvaBufferWrapper").replace("UvaBufferWrapper", "UvaBuffer"))
add_imports(buffer, "from collections.abc import Callable")
write(buffer, read(buffer) + "\n\n" + "\n\n".join(definition(patch, n) for n in ("get_row_indices_from_key", "MonitoredNumPyArray", "MonitoredTorchTensor")))
for mod, fn in (("logprob", "compute_token_logprobs"), ("penalties", "apply_penalties"), ("gumbel", "gumbel_sample")):
    path = f"vllm/v1/worker/npu/v2/common/sample/{mod}.py"
    # Kernels and helpers reside together in the native implementation.
    source = read(path)
    node = find(source, fn)
    args = [a.arg for a in node.args.args]
    signature = segment(source, node).split("\n")
    body_start = node.body[0].lineno - start(node) - 1
    signature = "\n".join(signature[:body_start])
    replacement = signature + f"\n    from vllm.v1.worker.npu.v2.sample.{mod} import {fn} as native_impl\n    return native_impl(" + ", ".join(f"{a}={a}" for a in args) + ")\n"
    replace(path, fn, replacement)

path = "vllm/v1/worker/npu/v2/common/input_batch.py"
source = read(path)
node = find(source, "post_update")
signature = "\n".join(segment(source, node).split("\n")[:node.body[0].lineno - start(node) - 1])
replace(path, "post_update", signature + "\n    from vllm.v1.worker.npu.v2.input_batch import post_update as native_impl\n    return native_impl(" + ", ".join(f"{a.arg}={a.arg}" for a in node.args.args) + ")\n")
path = "vllm/v1/worker/npu/v2/common/spec_decode/eagle/speculator.py"
replace(path, "propose", definition("ascend/legacy_patches/worker/patch_v2/patch_eagle.py", "propose"), "EagleSpeculator")
source = read(path).replace("    build_attn_metadata,\n", "")
write(path, source)
add_imports(path, "from vllm.v1.worker.npu.v2.attn_utils import build_attn_metadata")
unwrap("vllm/v1/worker/npu/v2/spec_decode/eagle.py", {"build_attn_metadata_wrapper"})
# No runtime API replacement utility remains.
(ROOT / "vllm/v1/worker/npu/v2/utils.py").unlink()

(ROOT.parents[1] / "design/p2/baseline/runner-extraction.json").write_text(json.dumps(RECORDS, indent=2) + "\n")
print(f"Extracted {len(RECORDS)} Runner and component contracts")
