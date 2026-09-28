"""Fold the first set of effective Ascend patches into their source owners."""

import ast
import textwrap

from migrate_namespace import rewrite
from source_edit import (
    add_imports, append_class, definition, find, prepend_body, read,
    replace, replace_text, segment, write,
)

PATCH = "ascend/legacy_patches/"


def main():
    target = "vllm/v1/kv_cache_interface.py"
    donor = PATCH + "platform/patch_kv_cache_interface.py"
    child = find(read(donor), "AscendMLAAttentionSpec")
    for item in child.body:
        if isinstance(item, ast.FunctionDef):
            body = segment(read(donor), item)
            if item.name == "merge":
                replace(target, item.name, body, "MLAAttentionSpec")
            else:
                append_class(target, "MLAAttentionSpec", body)
        elif isinstance(item, ast.AnnAssign):
            append_class(target, "MLAAttentionSpec", segment(read(donor), item))
    replace_text("vllm/v1/attention/backends/mla/indexer.py",
                 "return [1 if current_platform.is_rocm() else 64]", "return [128]")

    target = "vllm/v1/core/sched/scheduler.py"
    source = definition(PATCH + "platform/patch_kv_connector_worker_metadata.py", "_update_from_output")
    function = ast.parse(source).body[0]
    prelude = "\n".join(segment(source, n) for n in function.body[:-1])
    prepend_body(target, "update_from_output", prelude, "Scheduler")

    target = "vllm/distributed/parallel_state.py"
    donor = PATCH + "worker/patch_distributed.py"
    body = definition(donor, "__init__", "GroupCoordinatorPatch")
    body = body.replace('    group_name = group_name or "anonymous"',
        '    from vllm.distributed.device_communicators.npu_communicator import NPUCommunicator\n'
        '    from vllm.utils.ascend import create_hccl_pg_options\n\n'
        '    group_name = group_name or "anonymous"')
    replace(target, "__init__", body, "GroupCoordinator")
    body = definition(donor, "all_to_all", "GroupCoordinatorPatch")
    append_class(target, "GroupCoordinator", body)

    replace("vllm/v1/cudagraph_dispatcher.py", "_create_padded_batch_descriptor",
            definition(PATCH + "worker/patch_cudagraph.py", "_create_padded_batch_descriptor"),
            "CudagraphDispatcher")

    target = "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"
    replace(target, "init_buffer", definition(PATCH + "worker/patch_routed_experts_capturer.py", "init_buffer"),
            "RoutedExpertsCapturer")
    if "from vllm.platforms import current_platform" not in read(target):
        add_imports(target, "from vllm.platforms import current_platform")

    target = "vllm/model_executor/layers/utils.py"
    donor = PATCH + "worker/patch_unquantized_gemm.py"
    replace(target, "default_unquantized_gemm", definition(donor, "default_unquantized_gemm"))
    extra = definition(donor, "unquantized_gemm") + "\n" + definition(donor, "unquantized_gemm_fake")
    extra += '\ndirect_register_custom_op(op_name="unquantized_gemm", op_func=unquantized_gemm, '
    extra += 'fake_impl=unquantized_gemm_fake, mutates_args=[], dispatch_key="PrivateUse1")\n'
    write(target, read(target) + "\n\n" + extra)
    if "import direct_register_custom_op" not in read(target):
        add_imports(target, "from vllm.utils.torch_utils import direct_register_custom_op")

    target = "vllm/model_executor/model_loader/weight_utils.py"
    old = definition(target, "maybe_remap_kv_scale_name")
    renamed = old.replace("def maybe_remap_kv_scale_name(", "def _remap_common_kv_scale_name(", 1)
    donor = definition(PATCH + "worker/patch_weight_utils.py", "patch_deepseek")
    nested = next(n for n in ast.walk(ast.parse(donor)) if isinstance(n, ast.FunctionDef) and n.name == "new_remap")
    native = segment(donor, nested).replace("def new_remap(", "def maybe_remap_kv_scale_name(", 1)
    native = native.replace("ori_maybe_remap_kv_scale_name(", "_remap_common_kv_scale_name(")
    replace(target, "maybe_remap_kv_scale_name", renamed + "\n\n" + native)

    donor = PATCH + "worker/patch_deepseek_mtp.py"
    for target in ["vllm/model_executor/models/deepseek_v2.py", "vllm/model_executor/models/deepseek_mtp.py"]:
        source = read(target)
        if any(getattr(n, "name", None) == "get_spec_layer_idx_from_weight_name" for n in ast.parse(source).body):
            replace(target, "get_spec_layer_idx_from_weight_name", definition(donor, "get_spec_layer_idx_from_weight_name"))
            add_imports(target, 'MTP_ROT_WEIGHT_NAME = "rot.weight"')
    target = "vllm/model_executor/models/deepseek_mtp.py"
    source = definition(donor, "__init__", "AscendDeepSeekMultiTokenPredictorLayer")
    function = ast.parse(source).body[0]
    extra = "\n".join(segment(source, n) for n in function.body[1:])
    original = definition(target, "__init__", "DeepSeekMultiTokenPredictorLayer")
    replace(target, "__init__", original.rstrip() + "\n" + textwrap.indent(extra, "    "),
            "DeepSeekMultiTokenPredictorLayer")
    replace(target, "forward", definition(donor, "forward", "AscendDeepSeekMultiTokenPredictorLayer"),
            "DeepSeekMultiTokenPredictorLayer")
    prepend_body(target, "_rewrite_spec_layer_name",
                 'if name == "rot.weight":\n    return f"model.layers.{spec_layer}.rot.weight"', "DeepSeekMTP")
    print("Integrated KV spec, metadata, HCCL coordinator, graph descriptors, GEMM, routing, weights and MTP")


if __name__ == "__main__":
    main()
