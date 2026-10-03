#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a reviewable patch for constant profile branches; never edit files.

Only explicit hardware predicates and disabled feature branches are folded.
No identifier substring deletion is performed. Comments and retained statement
text are preserved. Run the P4 closure and contract checks after applying.
"""

import argparse
import ast
import difflib
from pathlib import Path
import textwrap

UNKNOWN = object()
PLATFORMS = {"is_npu": True, "is_cuda": False, "is_rocm": False,
             "is_cuda_alike": False, "is_cpu": False, "is_xpu": False,
             "is_tpu": False, "is_out_of_tree": False, "is_zen_cpu": False}
DISABLED = {"pcp_use_hybrid_attn", "lora_config", "is_pooling_model", "is_multimodal_model",
            "enable_lora", "enable_blending", "supports_multimodal",
            "is_aiter_triton_fp4_bmm_enabled", "is_aiter_triton_fp8_bmm_enabled",
            "rocm_aiter_fmoe_enabled", "aiter_fmoe_shared_expert_enabled",
            "is_rocm_aiter_moe_enabled", "is_fusion_moe_shared_experts_enabled",
            "use_aiter", "use_rocm_aiter", "_has_gdn", "_use_cudnn_prefill",
            "_use_fi_prefill", "_use_trtllm_ragged_prefill", "use_harmony",
            "supports_mm_inputs", "is_multimodal", "use_unified_vision_chunk",
            "allow_dsv3_router_gemm", "allow_cublas_router_gemm",
            "is_encoder_decoder", "uses_mrope", "uses_xdrope",
            "is_multimodal_raw_input_only_model", "enable_prompt_embeds", "has_encoder_inputs"}


def value(node: ast.AST):
    """Evaluate only the closed profile predicates, not arbitrary Python."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id == "dual_chunk_attention_config":
        return None
    if isinstance(node, ast.Attribute) and node.attr == "uses_xdrope_dim":
        return 0
    if isinstance(node, ast.Name) and node.id in ("use_aiter", "use_rocm_aiter", "supports_mm_inputs", "is_encoder_decoder", "encoder_inputs_to_schedule", "external_encoder_inputs", "preempted_encoder_inputs", "is_ngram_gpu"):
        return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "is_mistral_tokenizer":
        return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "supports_multimodal":
        return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance" and len(node.args) == 2 and isinstance(node.args[1], ast.Name) and node.args[1].id in ("MistralTokenizer", "MistralToolCall"):
        return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "isinstance" and len(node.args) == 2:
        types = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
        if all(isinstance(t, ast.Name) and t.id in {"AscendNgramProposer", "AscendSuffixDecodingProposer", "AscendMedusaProposer", "NgramProposer", "NgramDeviceProposer", "MedusaProposer", "SuffixDecodingProposer", "ExtractHiddenStatesProposer"} for t in types):
            return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "supports_multimodal_inputs":
        return False
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in ("use_ngram_gpu", "uses_draft_model"):
            return False
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            if fn.value.id == "current_platform" and fn.attr in PLATFORMS:
                return PLATFORMS[fn.attr]
            if fn.value.id == "rocm_aiter_ops" and fn.attr.startswith("is_"):
                return False
        if isinstance(fn, ast.Name) and fn.id == "is_310p":
            return False
        if isinstance(fn, ast.Name) and fn.id == "get_ascend_device_type":
            return "AscendDeviceType.A2"
        if isinstance(fn, ast.Name) and fn.id in {
            "has_flashinfer", "has_nvidia_artifactory", "is_deep_gemm_supported",
            "use_flashinfer_prefill", "use_cudnn_prefill", "has_triton_kernels",
            "use_trtllm_ragged_deepseek_prefill", "backend_supports_prefill_query_quantization",
        }:
            return False
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "AscendDeviceType":
            return f"AscendDeviceType.{node.attr}"
    if isinstance(node, ast.Attribute) and node.attr in DISABLED:
        # Config objects or runner properties, not free-standing user mappings.
        if isinstance(node.value, (ast.Name, ast.Attribute)):
            return None if node.attr in ("lora_config", "ec_connector", "ec_transfer_config") else False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        operand = value(node.operand)
        return UNKNOWN if operand is UNKNOWN else not operand
    if isinstance(node, ast.BoolOp):
        operands = [value(n) for n in node.values]
        if isinstance(node.op, ast.And):
            if any(v is not UNKNOWN and not v for v in operands):
                return False
            if all(v is not UNKNOWN for v in operands):
                return all(operands)
        else:
            if any(v is not UNKNOWN and v for v in operands):
                return True
            if all(v is not UNKNOWN for v in operands):
                return any(operands)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = value(node.left), value(node.comparators[0])
        if isinstance(node.left, ast.Name) and node.left.id == "scaling_type" and right in ("llama3", "mllama4", "ntk", "dynamic", "xdrope", "longrope", "openpangu"):
            if isinstance(node.ops[0], ast.Eq):
                return False
        if isinstance(node.ops[0], ast.In) and left in ("mrope_section", "use_fope") and isinstance(node.comparators[0], ast.Name) and node.comparators[0].id == "rope_parameters":
            return False
        if isinstance(node.left, ast.Attribute) and node.left.attr == "method" and right in ("ngram", "ngram_gpu", "suffix", "medusa", "extract_hidden_states", "eagle", "eagle3", "draft_model"):
            if isinstance(node.ops[0], ast.Eq):
                return False
            if isinstance(node.ops[0], ast.NotEq):
                return True
        if isinstance(node.left, ast.Subscript) and isinstance(node.left.slice, ast.Constant) and node.left.slice.value == "type" and right in ("multimodal", "enc_dec", "embeds"):
            if isinstance(node.ops[0], ast.Eq):
                return False
            if isinstance(node.ops[0], ast.NotEq):
                return True
        if isinstance(node.left, ast.Attribute) and node.left.attr == "quantization":
            if isinstance(right, str) and right not in ("ascend", "compressed-tensors"):
                if isinstance(node.ops[0], ast.Eq):
                    return False
                if isinstance(node.ops[0], ast.NotEq):
                    return True
        if isinstance(node.left, (ast.Attribute, ast.Name)):
            name = node.left.attr if isinstance(node.left, ast.Attribute) else node.left.id
            if name in ("load_format", "safetensors_load_strategy") and right in (
                    "tensorizer", "gguf", "bitsandbytes", "mistral", "runai_streamer", "runai_streamer_sharded", "torchao", "fastsafetensors", "instanttensor"):
                if isinstance(node.ops[0], ast.Eq):
                    return False
        if left is not UNKNOWN and right is not UNKNOWN:
            op = node.ops[0]
            if isinstance(op, ast.Is):
                return left is right
            if isinstance(op, ast.IsNot):
                return left is not right
            if isinstance(op, ast.Eq):
                return left == right
            if isinstance(op, ast.NotEq):
                return left != right
            if isinstance(op, ast.Gt):
                return left > right
        # Pooling is excluded for every model; MTP uses the draft runner.
        if isinstance(node.left, ast.Attribute) and node.left.attr == "runner_type":
            if right == "pooling":
                if isinstance(node.ops[0], ast.Eq):
                    return False
                if isinstance(node.ops[0], ast.NotEq):
                    return True
    return UNKNOWN


def specialize(source: str) -> str:
    """Reduce known if-statements, preserving exact retained source text."""
    tree = ast.parse(source)
    predicates = []
    source_lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in source_lines:
        offsets.append(offsets[-1] + len(line))

    def offset(line, column):
        return offsets[line - 1] + len(source_lines[line - 1].encode()[:column].decode())

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and value(node) in (True, False):
            # Replace explicit device probes also in assignments/ternaries.
            predicates.append((offset(node.lineno, node.col_offset),
                               offset(node.end_lineno, node.end_col_offset),
                               repr(value(node))))
    # Profile-only ternaries (e.g. absent encoder budgets). Do this in a
    # separate pass to avoid overlapping edits with the call replacements.
    for start, end, constant in sorted(predicates, reverse=True):
        source = source[:start] + constant + source[end:]
    for _ in range(20):
        lines = source.splitlines(True)
        positions = [0]
        for line in lines:
            positions.append(positions[-1] + len(line))
        edits = []
        def visit_expr(node):
            if isinstance(node, ast.IfExp) and value(node.test) is not UNKNOWN:
                branch = node.body if value(node.test) else node.orelse
                begin = positions[node.lineno - 1] + len(lines[node.lineno - 1].encode()[:node.col_offset].decode())
                end = positions[node.end_lineno - 1] + len(lines[node.end_lineno - 1].encode()[:node.end_col_offset].decode())
                edits.append((begin, end, '(' + ast.get_source_segment(source, branch) + ')'))
                return
            for child in ast.iter_child_nodes(node):
                visit_expr(child)
        visit_expr(ast.parse(source))
        if not edits:
            break
        for begin, end, replacement in sorted(edits, reverse=True):
            source = source[:begin] + replacement + source[end:]
    for _ in range(40):
        tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        changes = []

        def visit(node: ast.AST) -> None:
            # LoRA contexts are no-ops for the approved profile. Preserve their
            # complete model/graph body without executing adapter machinery.
            if isinstance(node, ast.With) and len(node.items) == 1:
                expr = node.items[0].context_expr
                if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                        and expr.func.attr in ("maybe_dummy_run_with_lora", "maybe_setup_dummy_loras")):
                    branch = node.body
                    selected = "".join(lines[branch[0].lineno - 1:branch[-1].end_lineno])
                    changes.append((node.lineno - 1, node.end_lineno,
                                    textwrap.indent(textwrap.dedent(selected), " " * node.col_offset)))
                    return
            if isinstance(node, ast.If):
                condition = value(node.test)
                if condition is not UNKNOWN:
                    branch = node.body if condition else node.orelse
                    if branch:
                        start = branch[0].lineno - 1
                        end = branch[-1].end_lineno
                        selected = "".join(lines[start:end])
                        if selected.lstrip().startswith("elif "):
                            selected = selected.replace("elif ", "if ", 1)
                        new = textwrap.indent(textwrap.dedent(selected), " " * node.col_offset)
                    else:
                        new = " " * node.col_offset + "pass  # Unsupported P4 branch removed.\n"
                    changes.append((node.lineno - 1, node.end_lineno, new))
                    return
            for child in ast.iter_child_nodes(node):
                visit(child)

        visit(tree)
        if not changes:
            return source
        for start, end, replacement in sorted(changes, reverse=True):
            lines[start:end] = [replacement]
        source = "".join(lines)
        ast.parse(source)
    raise RuntimeError("branch specialization did not converge")


def main() -> None:
    """Write apply_patch input to stdout; caller reviews and applies it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print("*** Begin Patch")
    for root in args.paths:
        for path in ([root] if root.is_file() else sorted(root.rglob("*.py"))):
            # Platform/interface methods are protocol definitions, not detection.
            if path.name in ("inference_profile.py", "interface.py"):
                continue
            old = path.read_text(encoding="utf-8")
            new = specialize(old)
            if new == old:
                continue
            print(f"*** Update File: {path}")
            for line in list(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))[2:]:
                print("@@" if line.startswith("@@") else line.rstrip("\n"))
    print("*** End Patch")


if __name__ == "__main__":
    main()
