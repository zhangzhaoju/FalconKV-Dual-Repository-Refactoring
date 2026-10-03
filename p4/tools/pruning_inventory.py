#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Inventory explicitly removed P4 trees from the frozen P3 Git index.

Read-only by default. --patch emits text-file deletions for apply_patch, and a
recoverable SHA-256 manifest. Binary files and gitlinks are reported separately.
The immutable P3 tag is the recovery source, not another installed code archive.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

WORKSPACE = Path(__file__).resolve().parents[3]
COMMON = {
    ".github/workflows/": "upstream CI/build/publish pipelines; replaced by source-only P4 gate",
    "benchmarks/": "upstream multi-model/device benchmarks; paired GLM client retained",
    "ascend/legacy-p3/": "P3 plugin reference archive (frozen in Git)",
    "ascend/legacy_plugin/": "P2 plugin reference archive (frozen in Git)",
    "ascend/legacy_patches/": "P2 patch reference archive (frozen in Git)",
    "ascend/legacy_tests/": "tests of removed plugin owners",
    "ascend/upstream-build/": "superseded standalone build entries",
    "ascend/.github/": "standalone donor CI",
    "ascend/docker/": "standalone donor images",
    "docker/": "upstream other-device images",
    ".buildkite/": "upstream other-device CI",
}
VLLM = {
    "ascend/csrc/causal_conv1d/": "off-profile Mamba/GDN native operator; no GLM callers",
    "vllm/v1/pool/late_interaction.py": "retired pooling routing and MaxSim execution",
    "vllm/v1/worker/mamba_utils.py": "retired recurrent-state execution; GLM DSA/MTP retained",
    "vllm/model_executor/layers/mamba/abstract.py": "retired Mamba model provider interface",
    "vllm/compilation/passes/utility/": "GPU-only functionalization passes; native Ascend passes retained",
    "vllm/distributed/ec_transfer/": "removed multimodal encoder transfer",
    "vllm/entrypoints/mcp/": "removed server-executed Harmony/MCP tools; GLM tool-call text protocol retained",
    "vllm/transformers_utils/configs/": "other-model custom checkpoint configurations",
    "ascend/csrc/moe_gating_top_k/op_kernel/arch35/": "A5-only kernel implementation",
    "vllm/vllm_flash_attn/": "CUDA FlashAttention extension interfaces",
    "vllm/kernels/helion/": "GPU Helion kernels",
    "vllm/entrypoints/openai/responses/": "multipurpose agent/media Responses API; text chat/completions retained",
    "vllm/entrypoints/openai/parser/": "GPT-OSS Harmony protocol",
    "vllm/model_executor/layers/pooler/": "pooling execution (retired request slots reject payloads)",
    "vllm/multimodal/": "removed media registry, processors, encoders and serializers",
    "vllm/transformers_utils/processors/": "multimodal checkpoint processors",
    "vllm/benchmarks/": "upstream multipurpose benchmark suite; use frozen paired text client",
    "vllm/entrypoints/cli/benchmark/": "off-profile benchmark CLI",
    "vllm/v1/worker/npu/v2/common/mm/": "multimodal encoder execution",
    "vllm/model_executor/layers/ascend/triton/fla/": "non-GLM linear attention implementation",
    "vllm/model_executor/layers/ascend/triton/mamba/": "non-GLM Mamba implementation",
    "csrc/": "unused upstream CUDA/HIP/CPU compute build tree",
    "cmake/": "unused upstream device build definitions (native uses ascend/cmake)",
    "vllm/platforms/ascend_310p/": "non-910B3 platform",
    "vllm/v1/worker/gpu/": "GPU worker implementation",
    "vllm/compilation/xlite/": "other-device compiler backend",
    "vllm/entrypoints/serve/rlhf/": "training weight-update API",
    "vllm/entrypoints/openai/speech_to_text/": "multimodal audio serving",
    "vllm/entrypoints/openai/realtime/": "multimodal realtime serving",
    "vllm/entrypoints/pooling/": "pooling serving",
    "examples/": "upstream off-profile examples; replaced by P4 recipe",
    "vllm/distributed/weight_transfer/": "training weight transfer",
    "vllm/entrypoints/serve/lora/": "LoRA management API",
    "vllm/entrypoints/sagemaker/": "off-profile deployment and pooling routes",
    "vllm/plugins/lora_resolvers/": "LoRA adapter resolution",
    "vllm/model_executor/warmup/": "GPU kernel warmup (no NPU callers)",
    "vllm/v1/worker/npu/v2/common/pool/": "pooling worker implementation",
    "vllm/model_executor/kernels/linear/": "unused CUDA/HIP/XPU/CPU linear kernels",
    "vllm/model_executor/layers/fused_moe/configs/": "other-device MoE tuning data",
    "vllm/model_executor/layers/fused_moe/experts/": "TRTLLM MoE experts",
    "vllm/compilation/passes/fusion/": "GPU fusion passes; native Ascend passes retained",
    "vllm/v1/attention/ops/": "GPU attention execution; native Ascend attention retained",
    "vllm/model_executor/layers/fla/": "off-profile linear attention models",
    "vllm/model_executor/layers/pooler/seqwise/": "sequence pooling implementations",
    "vllm/model_executor/layers/pooler/tokwise/": "token pooling implementations",
}
LMCACHE = {
    "ascend/csrc/mindspore/": "MindSpore native integration",
    "ascend/tests/mindspore/": "MindSpore tests",
    "lmcache/server/": "legacy v0 server; native v1 server retained",
    "lmcache/storage_backend/": "legacy CUDA CacheGen/serde; native v1 storage retained",
    "lmcache/v1/distributed/": "separate GPU multiprocess server storage stack; native remote-fill/controller retained",
    "lmcache/integration/sglang/": "SGLang integration",
    "lmcache/v1/compute/": "CacheBlend compute/models/other-device attention",
    "lmcache/v0/": "legacy v0 engine",
    "examples/": "upstream off-profile examples; replaced by P4 recipe",
}
MODEL_KEEP = {
    "__init__.py", "registry.py", "interfaces.py", "interfaces_base.py",
    "utils.py", "adapters.py", "config.py", "deepseek_v2.py", "deepseek_mtp.py",
    "module_mapping.py",
}


def reason(name: str, path: str) -> str | None:
    """Select exact audited directories/files, never substring-match model layers."""
    if name == "vllm" and path.startswith("ascend/tests/e2e/"):
        if not (path.startswith("ascend/tests/e2e/nightly/single_node/ops/")
                or path.startswith("ascend/tests/e2e/singlecard/compile/")
                or path in {"ascend/tests/e2e/__init__.py", "ascend/tests/e2e/conftest.py",
                            "ascend/tests/e2e/singlecard/__init__.py",
                            "ascend/tests/e2e/singlecard/test_sfa_event_handoff.py"}):
            return "off-profile model launchers/fixtures; native operator and SFA handoff tests retained"
        if Path(path).name in {"test_fused_sigmoid_gating_delta_rule.py",
                               "test_fused_qkvzba_split_reshape_cat.py", "test_l2norm.py"}:
            return "tests of removed Qwen linear-attention implementations"
    if name == "vllm" and path.startswith("ascend/benchmarks/") and not path.startswith("ascend/benchmarks/ops/"):
        return "upstream multi-model benchmark launcher; native sparse-cache operator benchmarks retained"
    if name == "vllm" and path in {
        "vllm/_xpu_ops.py",
        "vllm/model_executor/layers/ascend/triton/fused_gdn_gating.py",
        "vllm/model_executor/layers/ascend/triton/layernorm_gated.py",
        "vllm/model_executor/layers/ascend/triton/linearnorm/split_qkv_rmsnorm_mrope.py",
    }:
        return "off-profile device, gated linear-attention or multimodal rotary kernels"
    for prefix, why in {**COMMON, **(VLLM if name == "vllm" else LMCACHE)}.items():
        if path.startswith(prefix):
            return why
    if path.startswith("ascend/Dockerfile") or path in ("ascend/requirement_ms.txt", "ascend/requirements.txt", "ascend/requirement.txt", "ascend/collect_env.py"):
        return "superseded donor install/device environment instructions"
    if path.startswith("requirements/") and Path(path).name not in {"ascend.txt", "build.txt", "lint.txt", "docs.txt", "test.txt"}:
        return "off-profile dependency install entry"
    if name == "vllm" and path.startswith("docs/") and path not in {"docs/design/ascend_p2.md", "docs/design/ascend_p3.md", "docs/design/npu_bootstrap_fix.md"}:
        return "upstream multi-device/model documentation; phase history retained, P4 guide replaces it"
    if name == "vllm" and path.startswith("ascend/docs/source/"):
        return "standalone donor multi-model/device documentation"
    if name == "vllm" and path.startswith("tests/") and not path.startswith("tests/standalone/") and path != "tests/__init__.py":
        return "upstream non-profile test suite; native Ascend and paired standalone regressions retained"
    if name == "vllm" and path.startswith("ascend/examples/") and not path.startswith("ascend/examples/disaggregated_prefill_v1/"):
        return "other-model/device donor examples"
    if name == "vllm" and path.startswith("tools/") and (path.startswith(("tools/ep_kernels/", "tools/vllm-rocm/", "tools/vllm-tpu/")) or Path(path).name in {"flashinfer-build.sh", "install_deepgemm.sh", "install_gdrcopy.sh", "install_nixl_from_source_ubuntu.py", "install_torchcodec_rocm.sh", "generate_cmake_presets.py"}):
        return "other-vendor build/install tooling"
    if name == "vllm" and path.startswith(("ascend/tests/ut/_310p/", "ascend/tests/ut/lora/", "ascend/tests/ut/patch/", "ascend/tests/e2e/nightly/single_node/models/")):
        return "tests of deleted donor patch/model/hardware/LoRA implementations"
    if name == "vllm" and path.startswith("ascend/tests/e2e/") and any(word in path.lower() for word in ("a3", "310p", "lora", "multimodal", "qwen", "mrope", "bgmv", "sgmv", "gated_delta", "gdn", "conv1d", "mamba")):
        return "removed model/device operator tests"
    if name == "LMCache":
        if path.startswith("docs/source/") and path not in {"docs/source/index.rst", "docs/source/conf.py", "docs/source/getting_started/ascend_p1.rst", "docs/source/getting_started/ascend_p2.rst", "docs/source/getting_started/ascend_p3.rst"}:
            return "upstream multi-provider documentation; replaced by P4 native guide"
        if path.startswith("docs/design/l2_adapters/"):
            return "removed multiprocess GPU L2 adapter documentation"
        if path.startswith(("docs/source/api/", "docs/source/_static/", "ascend/docs/cachegen/", "ascend/tests/v1/blend/", "ascend/benchmark/")):
            return "removed plugin/CacheBlend/cachegen documentation or benchmark assets"
        if path.startswith(("tests/v1/multiprocess/", "tests/v1/distributed/", "tests/v1/compute/", "tests/v1/blend/", "tests/sglang/", "tests/v0/")):
            return "tests of retired GPU multiprocess/CacheBlend/legacy server"
        if path.startswith("ascend/examples/"):
            return "legacy donor deployment examples; formal paired P4 instructions replace them"
    if name == "vllm":
        if path.startswith("vllm/model_executor/layers/rotary_embedding/") and Path(path).name not in {"__init__.py", "base.py", "common.py", "deepseek_scaling_rope.py", "yarn_scaling_rope.py", "linear_scaling_rope.py"}:
            return "non-GLM rotary variants and multimodal positional encodings"
        if path in ("ascend/tools/install_flash_infer_attention_score_ops_a3.sh", "ascend/tools/send_mm_request.py"):
            return "other-hardware/multimodal helper"
        if path in ("vllm/utils/nvtx_pytorch_hooks.py", "vllm/model_executor/layers/lightning_attn.py", "vllm/compilation/passes/pass_manager.py",
                    "vllm/model_executor/offloader/uva.py", "vllm/model_executor/offloader/prefetch.py", "vllm/model_executor/offloader/prefetch_ops.py"):
            return "unused other-device execution/profiling/weight-offload provider"
        if path.startswith("vllm/transformers_utils/chat_templates/") and path.endswith(".jinja"):
            return "other-model fallback templates; use the GLM checkpoint template"
        if path in {f"vllm/assets/{name}.py" for name in ("audio", "image", "video")}:
            return "multimodal test asset loaders"
        if path in ("vllm/_oink_ops.py", "vllm/transformers_utils/gguf_utils.py"):
            return "removed GPU provider or unsupported checkpoint format"
        if path in {f"vllm/v1/spec_decode/{name}.py" for name in (
                "draft_model", "medusa", "extract_hidden_states", "ngram_proposer", "ngram_proposer_gpu", "suffix_decoding")}:
            return "non-MTP speculative decoding implementations"
        if path in {f"vllm/v1/spec_decode/ascend/{name}.py" for name in (
                "draft_proposer", "medusa_proposer", "ngram_proposer", "ngram_device_proposer", "suffix_proposer")}:
            return "non-MTP Ascend speculative decoding implementations"
        if path.startswith("vllm/distributed/device_communicators/") and Path(path).name in {
            "all_reduce_utils.py", "cpu_communicator.py", "cuda_communicator.py", "cuda_wrapper.py",
            "custom_all_reduce.py", "flashinfer_all_reduce.py", "mnnvl_compat.py", "pynccl_allocator.py",
            "pynccl.py", "pynccl_wrapper.py", "quick_all_reduce.py", "symm_mem.py", "xpu_communicator.py", "shm_object_storage.py"}:
            return "other-device collective implementation; HCCL and host IPC retained"
        if path in ("vllm/v1/core/encoder_cache_manager.py", "vllm/v1/worker/ec_connector_model_runner_mixin.py", "vllm/parser/minimax_m2_parser.py", "ascend/csrc/moe_gating_top_k/op_host/moe_gating_top_k_tiling_arch35.cpp", "ascend/csrc/moe_gating_top_k/op_kernel/moe_gating_top_k_apt.cpp", "vllm/_custom_ops.py", "vllm/utils/flashinfer.py", "vllm/utils/deep_gemm.py",
                    "vllm/utils/mistral.py", "vllm/transformers_utils/processor.py",
                    "vllm/device_allocator/cumem.py", "vllm/model_executor/layers/fused_moe/moe_align_block_size.py",
                    "vllm/v1/attention/backends/fa_utils.py", "vllm/v1/kv_offload/worker/cpu_gpu.py",
                    "vllm/v1/kv_offload/cpu.py", "vllm/entrypoints/openai/chat_completion/stream_harmony.py"):
            return "non-NPU operator/provider path"
        if path.startswith("vllm/tool_parsers/") and Path(path).name not in {
            "__init__.py", "abstract_tool_parser.py", "utils.py", "glm4_moe_tool_parser.py", "glm47_moe_tool_parser.py"}:
            return "non-GLM tool parser"
        if path.startswith("vllm/reasoning/") and Path(path).name not in {
            "__init__.py", "abs_reasoning_parsers.py", "basic_parsers.py", "identity_reasoning_parser.py", "deepseek_r1_reasoning_parser.py", "deepseek_v3_reasoning_parser.py"}:
            return "non-GLM reasoning parser; glm45 shared implementation retained"
        if path in {f"ascend/csrc/kernels/{name}.cpp" for name in ("bgmv_expand", "bgmv_shrink", "sgmv_expand", "sgmv_shrink")}:
            return "LoRA native kernels (bindings also removed)"
        if path.startswith("vllm/renderers/") and Path(path).name in ("grok2.py", "mistral.py", "deepseek_v32.py", "qwen_vl.py", "kimi_audio.py", "terratorch.py", "embed_utils.py"):
            return "non-GLM text renderer or embedding input implementation"
        if path.startswith("vllm/tokenizers/") and Path(path).name in ("grok2.py", "mistral.py", "deepseek_v32.py", "qwen_vl.py", "kimi_audio.py"):
            return "non-GLM tokenizer implementation"
        if path == "vllm/v1/worker/npu/v2/common/model_states/whisper.py":
            return "Whisper multimodal model state"
        if path.startswith("vllm/model_executor/layers/mamba/") and Path(path).name not in ("__init__.py", "mamba_utils.py", "abstract.py"):
            return "off-profile Mamba model implementations"
        if path in ("vllm/_aiter_ops.py", "vllm/model_executor/layers/kda.py",
                    "vllm/model_executor/layers/pooler/special.py", "vllm/model_executor/layers/pooler/activations.py",
                    "vllm/model_executor/layers/attention/static_sink_attention.py"):
            return "off-profile backend or operator implementation"
        quant_utils = "vllm/model_executor/layers/quantization/utils/"
        if path.startswith(quant_utils) and Path(path).name not in ("__init__.py", "quant_utils.py", "ocp_mx_utils.py", "layer_utils.py"):
            return "GPU quantization kernels; shared weight/shape helpers retained"
        if path in {f"vllm/model_executor/layers/fused_moe/{module}.py" for module in (
                "batched_deep_gemm_moe", "cpu_fused_moe", "cutlass_moe", "deep_gemm_moe", "deep_gemm_utils",
                "deepep_ht_prepare_finalize", "deepep_ll_prepare_finalize", "flashinfer_cutedsl_moe",
                "flashinfer_cutlass_moe", "flashinfer_nvlink_one_sided_prepare_finalize",
                "flashinfer_nvlink_two_sided_prepare_finalize", "flashinfer_trtllm_moe", "fused_batched_moe",
                "fused_marlin_moe", "fused_moe", "gpt_oss_triton_kernels_moe", "mori_prepare_finalize",
                "nixl_ep_prepare_finalize", "rocm_aiter_fused_moe", "triton_cutlass_moe",
                "triton_deep_gemm_moe", "trtllm_moe", "xpu_fused_moe", "zero_expert_fused_moe") }:
            return "other-device MoE implementation; common weight/router contracts retained"
        if path.startswith("vllm/model_executor/layers/fused_moe/oracle/") and Path(path).name not in ("__init__.py", "unquantized.py"):
            return "GPU MoE quantization dispatch"
        if path in ("vllm/model_executor/layers/sparse_attn_indexer.py", "vllm/model_executor/layers/quantization/utils/gptq_utils.py"):
            return "unused GPU indexer/quantization implementation"
        connector_prefix = "vllm/distributed/kv_transfer/kv_connector/v1/"
        if path.startswith(connector_prefix) and path.removeprefix(connector_prefix) not in (
                "__init__.py", "base.py", "metrics.py", "lmcache_connector.py", "multi_connector.py"):
            return "non-profile KV connector; paired native LMCache connector retained"
        backend_prefix = "vllm/v1/attention/backends/"
        if path.startswith(backend_prefix):
            relative = path.removeprefix(backend_prefix)
            if not (relative.startswith("ascend/") or relative in (
                    "__init__.py", "utils.py", "registry.py", "fa_utils.py", "mla/__init__.py", "mla/indexer.py")):
                return "non-native attention backend"
        if path.startswith("vllm/lora/") and path not in ("vllm/lora/__init__.py", "vllm/lora/request.py"):
            return "LoRA implementation (wire DTO retained only to reject legacy requests)"
        if path in ("vllm/entrypoints/openai/run_batch.py", "vllm/entrypoints/cli/run_batch.py",
                    "vllm/config/weight_transfer.py", "vllm/model_executor/models/adapters.py",
                    "vllm/v1/worker/lora_model_runner_mixin.py", "vllm/v1/worker/npu/v2/common/lora_utils.py"):
            return "removed training/LoRA/pooling/batch multipurpose entry"
        quant_prefix = "vllm/model_executor/layers/quantization/"
        if path.startswith(quant_prefix):
            relative = path.removeprefix(quant_prefix)
            if not (relative.startswith(("ascend/", "utils/")) or relative in {
                    "__init__.py", "base_config.py", "kv_cache.py", "input_quant_fp8.py",
                    "compressed_tensors/__init__.py", "compressed_tensors/utils.py"}):
                return "non-Ascend quantization provider"
        if path in {f"vllm/model_executor/model_loader/{module}.py" for module in (
                "bitsandbytes_loader", "gguf_loader", "tensorizer", "tensorizer_loader", "runai_streamer_loader") }:
            return "off-profile model/weight loader"
        if path.startswith("vllm/model_executor/models/"):
            relative = path.removeprefix("vllm/model_executor/models/")
            if relative not in MODEL_KEEP:
                return "non-GLM52 model implementation; shared GLM/MTP owners retained"
        if path in {f"vllm/platforms/{device}.py" for device in ("cpu", "cuda", "rocm", "xpu", "tpu", "zen_cpu")}:
            return "other-device platform implementation"
        if path.startswith("vllm/v1/worker/") and Path(path).name.startswith(("gpu_", "cpu_", "tpu_", "xpu_")):
            return "other-device worker implementation"
        if path in ("vllm/model_executor/layers/ascend/rel_pos_attention.py",
                    "vllm/model_executor/layers/ascend/mm_encoder_attention.py",
                    "vllm/model_executor/layers/attention/mm_encoder_attention.py"):
            return "multimodal encoder attention"
    else:
        if path.startswith("lmcache/v1/multiprocess/") and path not in ("lmcache/v1/multiprocess/__init__.py", "lmcache/v1/multiprocess/custom_types.py"):
            return "GPU multiprocess/CacheBlend server; NPU IPC ownership contract retained"
        if path in ("lmcache/v1/lazy_memory_allocator.py", "lmcache/integration/vllm/vllm_multi_process_adapter.py", "lmcache/cli/commands/server.py", "lmcache/v1/check/check_mode_test_l2_adapter.py", "lmcache/v1/transfer_channel/nixl_channel.py", "lmcache/v1/storage_backend/gds_backend.py", "lmcache/v1/storage_backend/nixl_storage_backend.py", "lmcache/v1/storage_backend/maru_backend.py", "lmcache/v1/storage_backend/connector/eic_connector.py", "lmcache/v1/storage_backend/connector/eic_adapter.py"):
            return "CUDA/HIP provider not used by native Ascend storage/transport"
        if path.startswith("lmcache/v1/storage_backend/naive_serde/") and Path(path).name.startswith(("cachegen_", "kivi_")):
            return "non-profile cache quantization providers; naive serialization retained"
        if path.startswith("csrc/") and not path.startswith(("csrc/storage_manager/", "csrc/storage_backends/")):
            return "unused upstream CUDA compute/allocator; host storage retained"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--repo", choices=("vllm", "LMCache"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    inventories = {}
    for name in ("vllm", "LMCache"):
        if args.repo and name != args.repo:
            continue
        repo = WORKSPACE / "p1-repos" / name
        entries = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "p3-frozen-20261003"], text=True).splitlines()
        files = []
        for path in entries:
            why = reason(name, path)
            file = repo / path
            if why is None or not file.is_file():
                continue
            content = file.read_bytes()
            try:
                content.decode("utf-8")
                binary = b"\0" in content
            except UnicodeDecodeError:
                binary = True
            files.append({"path": path, "reason": why, "sha256": hashlib.sha256(content).hexdigest(), "binary": binary})
        inventories[name] = files[:args.limit] if args.patch else files
    if not args.patch:
        print(json.dumps({name: {"files": len(files), "binary": [x["path"] for x in files if x["binary"]], "reasons": dict(Counter(x["reason"] for x in files))} for name, files in inventories.items()}, indent=2))
        return
    print("*** Begin Patch")
    for name, files in inventories.items():
        print(f"*** Add File: design/p4/baseline/{name.lower()}-prune-batch{args.batch}.json")
        print("\n".join("+" + line for line in json.dumps({"recovery_tag": "p3-frozen-20261003", "files": files}, indent=2).splitlines()))
        for item in files:
            if not item["binary"]:
                print(f"*** Delete File: p1-repos/{name}/{item['path']}")
    print("*** End Patch")


if __name__ == "__main__":
    main()
