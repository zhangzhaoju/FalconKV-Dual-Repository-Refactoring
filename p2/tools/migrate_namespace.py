#!/usr/bin/env python3
"""One-shot P2 namespace migration from the preserved platform batch.

Moves sources by semantic owner and rewrites absolute/relative Python imports.
Runtime patch integration is a separate reviewed step, not a rename operation.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = WORKSPACE / "p1-repos/vllm"
PREFIXES = {
    "vllm_ascend.worker.model_runner_v1": "vllm.v1.worker.npu_model_runner",
    "vllm_ascend.worker.worker": "vllm.v1.worker.npu_worker",
    "vllm_ascend.worker.npu_input_batch": "vllm.v1.worker.npu_input_batch",
    "vllm_ascend.worker.block_table": "vllm.v1.worker.npu_block_table",
    "vllm_ascend.worker.pcp_utils": "vllm.v1.worker.npu_pcp_utils",
    "vllm_ascend.worker.dsa_shared_pool": "vllm.v1.worker.dsa_shared_pool",
    "vllm_ascend.worker.serving_perf": "vllm.v1.worker.npu_serving_perf",
    "vllm_ascend.worker": "vllm.v1.worker.npu",
    "vllm_ascend.attention": "vllm.v1.attention.backends.ascend",
    "vllm_ascend.ops": "vllm.model_executor.layers.ascend",
    "vllm_ascend.quantization": "vllm.model_executor.layers.quantization.ascend",
    "vllm_ascend.compilation": "vllm.compilation.ascend",
    "vllm_ascend.distributed.kv_transfer": "vllm.distributed.kv_transfer.ascend",
    "vllm_ascend.distributed.device_communicators.npu_communicator":
        "vllm.distributed.device_communicators.npu_communicator",
    "vllm_ascend.distributed.device_communicators":
        "vllm.distributed.device_communicators.ascend",
    "vllm_ascend.distributed": "vllm.distributed.ascend",
    "vllm_ascend.eplb": "vllm.distributed.eplb.ascend",
    "vllm_ascend.core.recompute_scheduler": "vllm.v1.core.sched.recompute_scheduler",
    "vllm_ascend.core.scheduler_dynamic_batch": "vllm.v1.core.sched.dynamic_batch_scheduler",
    "vllm_ascend.core.mc2_recovery": "vllm.v1.core.mc2_recovery",
    "vllm_ascend.core": "vllm.v1.core.ascend",
    "vllm_ascend.sample": "vllm.v1.sample.ascend",
    "vllm_ascend.spec_decode": "vllm.v1.spec_decode.ascend",
    "vllm_ascend.kv_offload": "vllm.v1.kv_offload.ascend",
    "vllm_ascend.lora": "vllm.lora.ascend",
    "vllm_ascend.model_loader": "vllm.model_executor.model_loader.ascend",
    "vllm_ascend.device_allocator": "vllm.device_allocator.ascend",
    "vllm_ascend.device": "vllm.platforms.ascend_device",
    "vllm_ascend._310p": "vllm.platforms.ascend_310p",
    "vllm_ascend.xlite": "vllm.compilation.xlite",
    "vllm_ascend.ascend_config": "vllm.config.ascend",
    "vllm_ascend.envs": "vllm.envs_ascend",
    "vllm_ascend.utils": "vllm.utils.ascend",
    "vllm_ascend.cpu_binding": "vllm.utils.npu_cpu_binding",
    "vllm_ascend.diagnostic_utils": "vllm.utils.diagnostic_utils",
    "vllm_ascend.serving_perf": "vllm.utils.serving_perf",
    "vllm_ascend.lmcache_diagnostics": "vllm.distributed.kv_transfer.lmcache_diagnostics",
    "vllm_ascend.live_source_handoff": "vllm.distributed.kv_transfer.live_source_handoff",
    "vllm_ascend.profiling_config": "vllm.utils.ascend_profiling_config",
    "vllm_ascend.ascend_forward_context": "vllm.ascend_forward_context",
    "vllm_ascend.flash_common3_context": "vllm.compilation.flash_common3_context",
    "vllm_ascend.batch_invariant": "vllm.model_executor.layers.ascend_batch_invariant",
    "vllm_ascend.meta_registration": "vllm.model_executor.layers.ascend.meta_registration",
    "vllm_ascend.platform": "vllm.platforms.npu",
    "vllm_ascend.vllm_ascend_C": "vllm._ascend_C",
    "vllm_ascend._build_info": "vllm._build_info",
    "vllm_ascend._version": "vllm._version",
    "vllm_ascend": "vllm",
}


def mapped(name: str) -> str:
    for old in sorted(PREFIXES, key=len, reverse=True):
        if name == old or name.startswith(old + "."):
            return PREFIXES[old] + name[len(old):]
    return name


def rewrite(source: str, old_module: str | None = None, package=False) -> str:
    """Preserve non-import source text while resolving moved import bindings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        replacement = None
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level and old_module:
                parent = old_module if package else old_module.rpartition(".")[0]
                module = importlib.util.resolve_name("." * node.level + module, parent)
            if module.startswith("vllm_ascend"):
                statements = []
                for alias in node.names:
                    qualified = mapped(module + "." + alias.name)
                    parent, _, leaf = qualified.rpartition(".")
                    local = alias.asname or alias.name
                    target = leaf + (f" as {local}" if local != leaf else "")
                    statements.append(f"from {parent} import {target}")
                replacement = ("\n" + " " * node.col_offset).join(statements)
        else:
            aliases = []
            changed = False
            for alias in node.names:
                name = mapped(alias.name)
                changed |= name != alias.name
                aliases.append(name + (f" as {alias.asname}" if alias.asname else ""))
            if changed:
                replacement = "import " + ", ".join(aliases)
        if replacement is not None:
            edits.append((offsets[node.lineno - 1] + node.col_offset,
                          offsets[node.end_lineno - 1] + node.end_col_offset,
                          replacement))
    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    # Qualified references and lazy class-name strings; leave torch operator ABI
    # namespaces (torch.ops.vllm_ascend) untouched.
    pattern = r"(?<![\w.])vllm_ascend(?:\.[A-Za-z_]\w*)*(?![\w:])"
    source = re.sub(pattern, lambda match: mapped(match.group()), source)
    return source


def main():
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT,
                                   text=True).strip() == "p2"
    origin = ROOT / "ascend/vllm_ascend"
    assert origin.is_dir(), "One-shot migration has already run"
    records = []
    old_sources = {}
    tracked = subprocess.check_output(["git", "ls-files", "ascend/vllm_ascend"],
                                      cwd=ROOT, text=True).splitlines()
    for relative in tracked:
        path = ROOT / relative
        tail = path.relative_to(origin)
        content = path.read_bytes()
        module = None
        if tail.parts[0] == "patch":
            target = ROOT / "ascend/legacy_patches" / tail.relative_to("patch")
        elif str(tail) in ("__init__.py", "platform.py"):
            target = ROOT / "ascend/legacy_plugin" / tail
        elif tail.parts[0] == "_cann_ops_custom":
            target = ROOT / "vllm" / tail
        else:
            module = "vllm_ascend." + str(tail.with_suffix("")).replace("/", ".")
            is_package = tail.name == "__init__.py"
            if is_package:
                module = module.removesuffix(".__init__")
            if tail.suffix == ".py":
                dest = mapped(module).replace(".", "/")
                target = ROOT / (dest + ("/__init__.py" if is_package else ".py"))
            else:
                parent = "vllm_ascend." + str(tail.parent).replace("/", ".")
                target = ROOT / mapped(parent).replace(".", "/") / tail.name
            old_sources[target] = (module, is_package)
        assert not target.exists(), target
        records.append({"source": relative, "destination": str(target.relative_to(ROOT)),
                        "source_sha256": hashlib.sha256(content).hexdigest(),
                        "disposition": "archived" if "legacy_" in str(target) else "native"})
    for record in records:
        source, target = ROOT / record["source"], ROOT / record["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        if record["disposition"] == "native":
            parent = target.parent
            while parent != ROOT and parent.is_relative_to(ROOT / "vllm"):
                init = parent / "__init__.py"
                if not init.exists():
                    init.write_text("# SPDX-License-Identifier: Apache-2.0\n")
                parent = parent.parent
    for path in sorted((ROOT / "vllm").rglob("*.py")):
        source = path.read_text()
        if "vllm_ascend" not in source and path not in old_sources:
            continue
        module, package = old_sources.get(path, (None, False))
        updated = rewrite(source, module, package)
        if updated != source:
            path.write_text(updated)
    # Paired repository consumers must use the same public source/event contract.
    paired = WORKSPACE / "p1-repos/LMCache"
    for area in [paired / "lmcache", paired / "ascend/lmcache_ascend"]:
        for path in area.rglob("*.py"):
            if "/integration/patch/" in str(path):
                continue  # P3 owns legacy third-party patch tooling.
            source = path.read_text()
            if "vllm_ascend" in source:
                path.write_text(rewrite(source))
    output = WORKSPACE / "design/p2/baseline/namespace-migration.json"
    output.write_text(json.dumps({"prefixes": PREFIXES, "files": records}, indent=2) + "\n")
    print(f"Moved {len(records)} files; {sum(r['disposition']=='archived' for r in records)} archival")


if __name__ == "__main__":
    main()
