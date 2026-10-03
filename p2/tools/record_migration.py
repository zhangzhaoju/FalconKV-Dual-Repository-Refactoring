#!/usr/bin/env python3
"""Record final P2 ownership from the preserved, one-shot migration inventory."""

import hashlib
import json
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "p1-repos/vllm"
OUTPUT = WORKSPACE / "design/p2/baseline"

INTEGRATED = {
    "platform/patch_balance_schedule.py": ["vllm/v1/core/sched/balance_scheduler.py", "vllm/v1/engine/balance_core.py", "vllm/config/scheduler.py"],
    "platform/patch_distributed.py": ["vllm/distributed/ascend/collectives.py"],
    "platform/patch_fusion_matcher_compat_ops.py": ["vllm/compilation/passes/fusion/matcher_utils.py"],
    "platform/patch_glm_tool_call_parser.py": ["vllm/tool_parsers/glm4_moe_tool_parser.py", "vllm/entrypoints/openai/chat_completion/serving.py"],
    "platform/patch_kv_cache_interface.py": ["vllm/v1/kv_cache_interface.py", "vllm/v1/attention/backends/mla/indexer.py"],
    "platform/patch_kv_connector_worker_metadata.py": ["vllm/v1/core/sched/scheduler.py"],
    "platform/patch_minimax_usage_accounting.py": ["vllm/entrypoints/openai/chat_completion/serving.py", "vllm/entrypoints/openai/engine/protocol.py"],
    "platform/patch_multiproc_executor.py": ["vllm/v1/executor/multiproc_executor.py", "vllm/v1/executor/ascend_multiproc_executor.py", "vllm/v1/executor/abstract.py"],
    "platform/patch_sched_yield.py": ["vllm/distributed/utils.py"],
    "platform/patch_torch_accelerator.py": ["vllm/utils/mem_utils.py", "vllm/v1/worker/npu_worker.py", "vllm/v1/worker/npu_runner_state.py"],
    "worker/patch_cudagraph.py": ["vllm/v1/cudagraph_dispatcher.py"],
    "worker/patch_deepseek_mtp.py": ["vllm/model_executor/models/deepseek_mtp.py", "vllm/model_executor/models/deepseek_v2.py"],
    "worker/patch_distributed.py": ["vllm/distributed/parallel_state.py", "vllm/distributed/device_communicators/npu_communicator.py"],
    "worker/patch_draft_quarot.py": ["vllm/model_executor/model_loader/ascend/quarot.py", "vllm/model_executor/models/llama_eagle3.py"],
    "worker/patch_logprobs.py": ["vllm/v1/sample/ops/logprobs.py"],
    "worker/patch_module.py": ["vllm/v1/attention/backends/gdn_attn.py"],
    "worker/patch_npugraph_ex_triton.py": ["vllm/compilation/ascend/torchair_backend.py", "vllm/compilation/ascend/compiler_interface.py"],
    "worker/patch_qwen3_next_mtp.py": ["vllm/v1/worker/utils.py"],
    "worker/patch_rejection_sampler.py": ["vllm/v1/sample/rejection_sampler.py", "vllm/v1/sample/ascend/rejection_sampler.py"],
    "worker/patch_routed_experts_capturer.py": ["vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"],
    "worker/patch_unquantized_gemm.py": ["vllm/model_executor/layers/utils.py"],
    "worker/patch_weight_utils.py": ["vllm/model_executor/model_loader/weight_utils.py"],
    "worker/patch_v2/patch_block_table.py": ["vllm/v1/worker/npu/v2/block_table.py", "vllm/v1/worker/npu/v2/runner_state.py"],
    "worker/patch_v2/patch_eagle.py": ["vllm/v1/worker/npu/v2/common/spec_decode/eagle/speculator.py"],
    "worker/patch_v2/patch_input_batch.py": ["vllm/v1/worker/npu/v2/input_batch.py", "vllm/v1/worker/npu/v2/runner_state.py"],
    "worker/patch_v2/patch_model_state.py": ["vllm/v1/worker/npu/v2/model_states/__init__.py", "vllm/v1/worker/npu/v2/runner_state.py"],
    "worker/patch_v2/patch_triton.py": ["vllm/v1/worker/npu/v2/common/sample/gumbel.py", "vllm/v1/worker/npu/v2/common/sample/logprob.py", "vllm/v1/worker/npu/v2/common/sample/penalties.py", "vllm/v1/worker/npu/v2/common/input_batch.py"],
    "worker/patch_v2/patch_uva.py": ["vllm/v1/worker/npu/v2/common/buffer_utils.py"],
}
NOTES = {
    "platform/patch_fusion_matcher_compat_ops.py": "NPU 不构造 GPU matcher，移除伪造的 torch.ops 属性；真实算子由原生模块注册。",
    "platform/patch_minimax_usage_accounting.py": "公共 usage/full/stream 行为合入服务层；非目标 MiniMax reasoning parser 不激活，归 P4。",
    "platform/patch_distributed.py": "310P 特化通过显式调用，不改 torch.distributed；910B3 直接调用 torch。",
    "platform/patch_torch_accelerator.py": "各使用方显式选 NPU memory API，不改 torch.accelerator。",
    "worker/patch_npugraph_ex_triton.py": "单个 compiler 的私有函数依赖绑定，不改已安装 TorchAir 模块/类；内网核验私有 API、字典元数据及图执行。",
    "worker/patch_module.py": "GDN 布尔排序直接转换 dtype，删除 torch wrapper。GDN 模型入口不增加到认证范围。",
    "worker/patch_qwen3_next_mtp.py": "2026-09-30 更正：文件含 GLM DSA 通用 KV 绑定逻辑；精确 latent/indexer 兄弟对排序及原 NPU 绑定语义已合入原生函数，非 Qwen 模型支持恢复。",
}


def main():
    source = json.loads((OUTPUT / "namespace-migration.json").read_text())
    records, patches = [], []
    for item in source["files"]:
        entry = dict(item)
        destination = REPO / entry["destination"]
        if not destination.is_file():
            assert item["source"] == "ascend/vllm_ascend/worker/v2/utils.py"
            entry["final_disposition"] = "removed_cuda_wrapper"
            entry["native_owners"] = ["vllm/v1/worker/npu/v2/runner_state.py", "vllm/v1/worker/npu/v2/common"]
        else:
            entry["destination_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
            entry["final_disposition"] = item["disposition"]
        records.append(entry)
        if item["disposition"] == "native":
            continue
        assert entry["destination_sha256"] == entry["source_sha256"], destination
        old = item["source"].removeprefix("ascend/vllm_ascend/patch/")
        if old in INTEGRATED:
            disposition, owners = "integrated", INTEGRATED[old]
            note = NOTES.get(old, "行为合入正式定义或显式 NPU 分发，不再执行导入时替换。")
        elif "legacy_plugin" in str(destination) or old.endswith("__init__.py"):
            disposition, owners = "retired_loader", ["vllm/platforms/npu.py"]
            note = "原插件/patch 聚合加载器退出活动路径；不提供旧 namespace 壳。"
        else:
            disposition, owners = "inactive_non_target", []
            note = "非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。"
        for owner in owners:
            assert (REPO / owner).is_file(), owner
        patches.append({**entry, "patch_disposition": disposition, "native_owners": owners, "note": note})
    report = {
        "scope": "P2 source ownership; runtime equivalence pending intranet",
        "counts": dict(Counter(entry["final_disposition"] for entry in records)),
        "patch_counts": dict(Counter(entry["patch_disposition"] for entry in patches)),
        "files": records,
        "patches": patches,
    }
    (OUTPUT / "native-integration-migration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    rows = [
        "# P2 补丁处置清单", "",
        "覆盖 47 个旧 patch 文件及 2 个插件入口，全部保留原始字节和版权到 `ascend/legacy_*`，但不安装进 wheel/sdist，也不执行。完整哈希与文件映射见 [迁移清单](baseline/native-integration-migration.json)。", "",
        "`integrated` 表示源码已合入，运行等价仍待内网；`inactive_non_target` 表示按 P2 计划停用范围外模型补丁，后续 P4 再删源文件；`retired_loader` 表示旧入口已退出。", "",
        "| 原相对路径 | 处置 | 原生位置 | 说明 |", "| --- | --- | --- | --- |",
    ]
    for item in patches:
        old = item["source"].removeprefix("ascend/vllm_ascend/")
        owners = "<br>".join(f"`{path}`" for path in item["native_owners"]) or "—"
        rows.append(f"| `{old}` | {item['patch_disposition']} | {owners} | {item['note']} |")
    (OUTPUT.parent / "patch-migration.md").write_text("\n".join(rows) + "\n")
    print(json.dumps({key: report[key] for key in ("counts", "patch_counts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
