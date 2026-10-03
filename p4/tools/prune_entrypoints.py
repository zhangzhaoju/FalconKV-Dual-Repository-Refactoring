#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit the audited P4 API/training deletion patch without writing sources."""

import ast
import difflib
from pathlib import Path

ROOT = Path("p1-repos/vllm")


def remove(source, predicate):
    """Remove whole statements, including decorators, never overlapping edits."""
    lines = source.splitlines(True)
    edits = []

    def visit(node):
        if isinstance(node, ast.stmt) and predicate(node):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            edits.append((start - 1, node.end_lineno))
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    for start, end in sorted(edits, reverse=True):
        del lines[start:end]
    result = "".join(lines)
    ast.parse(result)
    return result


def prune(path, defs=(), imports=(), statements=()):
    source = path.read_text()

    def matches(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name in defs
        if isinstance(node, ast.ImportFrom):
            return any((node.module or "").startswith(prefix) for prefix in imports)
        if isinstance(node, (ast.If, ast.Expr, ast.Assign, ast.AnnAssign)):
            test = node.test if isinstance(node, ast.If) else node
            text = ast.get_source_segment(source, test) or ""
            return any(fragment in text for fragment in statements)
        return False

    return source, remove(source, matches)


SPECS = {
    "vllm/entrypoints/llm.py": dict(
        defs=("encode", "embed", "classify", "reward", "score", "_embedding_score",
              "_late_interaction_score", "_cross_encoding_score", "init_weight_transfer_engine", "update_weights"),
        imports=("vllm.entrypoints.pooling", "vllm.distributed.weight_transfer"),
        statements=("self.pooling_io_processors =",)),
    "vllm/v1/engine/async_llm.py": dict(defs=("encode", "init_weight_transfer_engine", "update_weights"),
                                      imports=("vllm.distributed.weight_transfer",)),
    "vllm/engine/protocol.py": dict(defs=("encode", "init_weight_transfer_engine", "update_weights"),
                                   imports=("vllm.distributed.weight_transfer",)),
    "vllm/entrypoints/openai/api_server.py": dict(
        imports=("vllm.entrypoints.sagemaker", "vllm.entrypoints.serve.rlhf"),
        statements=('"transcription" in supported_tasks', '"realtime" in supported_tasks',
                    "any(task in POOLING_TASKS", "attach_rlhf_router(app)",
                    "register_sagemaker_api_router(app", "app = sagemaker_standards_bootstrap(app)",
                    "default_mm_loras =", "lora_modules = process_lora_modules",
                    "await state.openai_serving_models.init_static_loras()")),
    "vllm/entrypoints/serve/__init__.py": dict(imports=("vllm.entrypoints.serve.lora",),
                                              statements=("attach_lora_router(app)",)),
    "vllm/entrypoints/openai/engine/serving.py": dict(
        defs=("_create_pooling_params", "_prepare_generators", "_collect_batch", "_pipeline", "handle",
              "_get_active_default_mm_loras", "_get_message_types"),
        imports=("vllm.entrypoints.pooling", "vllm.entrypoints.openai.speech_to_text"),
        statements=("ScoreDataRequest,",)),
    "vllm/entrypoints/cli/main.py": dict(statements=('sys.argv[1] == "bench"',)),
    "vllm/config/vllm.py": dict(imports=("weight_transfer",), statements=("weight_transfer_config:",)),
    "vllm/config/__init__.py": dict(imports=("vllm.config.weight_transfer",)),
    "vllm/engine/arg_utils.py": dict(statements=("weight_transfer_config:", "isinstance(self.weight_transfer_config, dict)",
                                               '"--weight-transfer-config"')),
}


def main():
    print("*** Begin Patch")
    for relative, spec in SPECS.items():
        path = ROOT / relative
        old, new = prune(path, **spec)
        if relative == "vllm/entrypoints/openai/api_server.py":
            new = new.replace("        lora_modules=lora_modules,\n", "")
            new = new.replace("    process_lora_modules,\n", "")
            new = new.replace("from vllm.tasks import POOLING_TASKS, SupportedTask", "from vllm.tasks import SupportedTask")
        if relative == "vllm/engine/arg_utils.py":
            new = new.replace("    WeightTransferConfig,\n", "")
            new = new.replace("            weight_transfer_config=self.weight_transfer_config,\n", "")
        if relative == "vllm/config/__init__.py":
            new = new.replace('    "WeightTransferConfig",\n', "")
        if old == new:
            continue
        ast.parse(new)
        print(f"*** Update File: {path}")
        for line in list(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))[2:]:
            print("@@" if line.startswith("@@") else line.rstrip("\n"))
    print("*** End Patch")


if __name__ == "__main__":
    main()
