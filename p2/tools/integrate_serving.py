"""Integrate GLM streaming/usage and process diagnostics as declared methods."""

import ast
from source_edit import add_imports, append_class, definition, find, read, replace, segment, write
from migrate_namespace import rewrite

PATCH = "ascend/legacy_patches/platform/"


def member(path, owner, name, source):
    cls = find(read(path), owner)
    if any(getattr(node, "name", None) == name for node in cls.body):
        replace(path, name, source, owner)
    else:
        append_class(path, owner, source)


def main():
    target = "vllm/entrypoints/openai/engine/protocol.py"
    donor = PATCH + "patch_minimax_usage_accounting.py"
    info = definition(donor, "CompletionTokenUsageInfo").replace("engine_protocol.OpenAIBaseModel", "OpenAIBaseModel")
    old = definition(target, "UsageInfo")
    replace(target, "UsageInfo", info + "\n\n" + old)
    append_class(target, "UsageInfo", "completion_tokens_details: CompletionTokenUsageInfo | None = None")

    target = "vllm/entrypoints/openai/chat_completion/serving.py"
    add_imports(target, "from dataclasses import dataclass\nfrom collections.abc import Sequence\n"
                       "from vllm.entrypoints.openai.engine.protocol import CompletionTokenUsageInfo")
    for name, decorator in [("_count_reasoning_tokens_for_usage", "@staticmethod\n"), ("_make_usage_info", "")]:
        text = definition(donor, name).replace("chat_serving.", "")
        member(target, "OpenAIServingChat", name, decorator + text)
    helper_names = ["_UsageTrackingState", "_create_usage_tracking_state", "_update_usage_tracking_state",
                    "_tracked_result_generator", "_sum_reasoning_tokens_for_usage", "_make_full_response_usage"]
    helpers = "\n\n".join(definition(donor, name) for name in helper_names)
    helpers = helpers.replace("chat_serving.", "")
    helpers = helpers.replace("_count_reasoning_tokens_for_usage(token_ids,", "OpenAIServingChat._count_reasoning_tokens_for_usage(token_ids,")
    write(target, read(target) + "\n\n" + helpers)
    original = definition(target, "chat_completion_full_generator", "OpenAIServingChat")
    original = original.replace("def chat_completion_full_generator(", "def _chat_completion_full_response(", 1)
    replacement = definition(donor, "_wrapped_chat_completion_full_generator")
    replacement = replacement.replace("_wrapped_chat_completion_full_generator", "chat_completion_full_generator")
    replacement = replacement.replace("self._ascend_original_chat_completion_full_generator", "self._chat_completion_full_response")
    replacement = replacement.replace("chat_protocol.", "").replace("engine_protocol.", "")
    replace(target, "chat_completion_full_generator", replacement, "OpenAIServingChat")
    append_class(target, "OpenAIServingChat", original)

    donor = PATCH + "patch_glm_tool_call_parser.py"
    for name, decorator in [
        ("_create_remaining_args_delta", "@staticmethod\n"),
        ("_record_streamed_tool_args", "@staticmethod\n"),
        ("_compact_json_fragment", "@staticmethod\n"),
        ("_compute_remaining_tool_args", "@classmethod\n"),
    ]:
        member(target, "OpenAIServingChat", name, decorator + definition(donor, name))
    replacement = definition(donor, "_patched_chat_completion_stream_generator")
    replacement = replacement.replace("_patched_chat_completion_stream_generator", "chat_completion_stream_generator")
    replace(target, "chat_completion_stream_generator", replacement, "OpenAIServingChat")
    replacement = definition(donor, "_patched_extract_tool_calls_streaming")
    replacement = replacement.replace("_patched_extract_tool_calls_streaming", "extract_tool_calls_streaming")
    replacement = replacement.replace("glm4_parser.", "")
    replace("vllm/tool_parsers/glm4_moe_tool_parser.py", "extract_tool_calls_streaming", replacement,
            "Glm4MoeModelToolParser")

    # Keep common process/queue behavior as explicit methods; no import-time
    # mutation of WorkerProc, FutureWrapper or executor classes.
    target = "vllm/v1/executor/multiproc_executor.py"
    donor = PATCH + "patch_multiproc_executor.py"
    add_imports(target, "from vllm.utils.serving_perf import cold_perf_enabled, log_cold_perf_event, log_cold_perf_process_event")
    source = read(donor)
    constants = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            name = ast.unparse(node.targets[0])
            if name.startswith(("_COLD_PERF_", "_SLOW_ASYNC_")) or name == "_worker_hook_reported":
                constants.append(segment(source, node))
    write(target, read(target) + "\n\n" + "\n".join(constants))
    for owner, name, donor_name, original_name in [
        ("WorkerProc", "handle_output", "_handle_output", "_handle_output_response"),
        ("WorkerProc", "enqueue_output", "_enqueue_output", "_enqueue_output_response"),
        ("FutureWrapper", "wait_for_response", "_wait_for_response", "_wait_for_response_base"),
    ]:
        original = definition(target, name, owner).replace(f"def {name}(", f"def {original_name}(", 1)
        replacement = definition(donor, donor_name).replace(f"def {donor_name}(", f"def {name}(", 1)
        replacement = replacement.replace("_worker_handle_output(self, output)", "self._handle_output_response(output)")
        replacement = replacement.replace("_worker_enqueue_output(self, output)", "self._enqueue_output_response(output)")
        replacement = replacement.replace("self: WorkerProc", "self").replace("self: FutureWrapper", "self")
        # A function signature ends at the first line ending in ':'.
        lines = replacement.splitlines(True)
        end = next(i for i, line in enumerate(lines) if line.rstrip().endswith(":")) + 1
        args = "get_response" if owner == "FutureWrapper" else "output"
        lines[end:end] = [f"    if not cold_perf_enabled():\n        return self.{original_name}({args})\n"]
        replace(target, name, "".join(lines), owner)
        append_class(target, owner, original)

    # Dynamic EPLB's non-daemon worker configuration is selected by the
    # executor factory instead of replacing its imported class.
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.ClassDef))]
    text = "# SPDX-License-Identifier: Apache-2.0\n# Copyright (c) 2025 Huawei Technologies Co., Ltd.\n"
    text += "\n".join(segment(source, node) for node in nodes)
    write("vllm/v1/executor/ascend_multiproc_executor.py", rewrite(text))
    target = "vllm/v1/executor/abstract.py"
    source = read(target)
    needle = "            executor_class = MultiprocExecutor\n"
    assert source.count(needle) == 1
    source = source.replace(needle, needle +
        '            import os\n'
        '            if (os.getenv("DYNAMIC_EPLB", "false").lower() in ("true", "1")\n'
        '                    or os.getenv("EXPERT_MAP_RECORD", "false") == "true"):\n'
        '                from vllm.v1.executor.ascend_multiproc_executor import AscendMultiprocExecutor\n'
        '                executor_class = AscendMultiprocExecutor\n')
    write(target, source)
    print("Integrated GLM tool streaming, usage accounting and process diagnostics")


if __name__ == "__main__":
    main()
