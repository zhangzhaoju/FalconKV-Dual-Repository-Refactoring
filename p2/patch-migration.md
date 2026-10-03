# P2 补丁处置清单

覆盖 47 个旧 patch 文件及 2 个插件入口，全部保留原始字节和版权到 `ascend/legacy_*`，但不安装进 wheel/sdist，也不执行。完整哈希与文件映射见 [迁移清单](baseline/native-integration-migration.json)。

2026-09-30 更正：`patch_qwen3_next_mtp.py` 虽带 Qwen 名称，却提供 GLM DSA 需要的通用 KV 绑定行为，原 `inactive_non_target` 分类错误。该行为现已合入 `vllm/v1/worker/utils.py` 的 NPU 分支，对应回归位于 `tests/standalone/test_npu_kv_cache_binding.py`。当前处置计数为 integrated 28、inactive_non_target 15、retired_loader 6；上方 JSON 保留 2026-09-27 交付时的历史记录，最新更正见 [修复清单](baseline/kv-cache-binding-fix-20260930.json)。不要重新运行初始测试搬迁脚本将新回归归档。

`integrated` 表示源码已合入，运行等价仍待内网；`inactive_non_target` 表示按 P2 计划停用范围外模型补丁，后续 P4 再删源文件；`retired_loader` 表示旧入口已退出。

| 原相对路径 | 处置 | 原生位置 | 说明 |
| --- | --- | --- | --- |
| `__init__.py` | retired_loader | `vllm/platforms/npu.py` | 原插件/patch 聚合加载器退出活动路径；不提供旧 namespace 壳。 |
| `patch/__init__.py` | retired_loader | `vllm/platforms/npu.py` | 原插件/patch 聚合加载器退出活动路径；不提供旧 namespace 壳。 |
| `patch/platform/__init__.py` | retired_loader | `vllm/platforms/npu.py` | 原插件/patch 聚合加载器退出活动路径；不提供旧 namespace 壳。 |
| `patch/platform/patch_balance_schedule.py` | integrated | `vllm/v1/core/sched/balance_scheduler.py`<br>`vllm/v1/engine/balance_core.py`<br>`vllm/config/scheduler.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/platform/patch_distributed.py` | integrated | `vllm/distributed/ascend/collectives.py` | 310P 特化通过显式调用，不改 torch.distributed；910B3 直接调用 torch。 |
| `patch/platform/patch_fusion_matcher_compat_ops.py` | integrated | `vllm/compilation/passes/fusion/matcher_utils.py` | NPU 不构造 GPU matcher，移除伪造的 torch.ops 属性；真实算子由原生模块注册。 |
| `patch/platform/patch_glm_tool_call_parser.py` | integrated | `vllm/tool_parsers/glm4_moe_tool_parser.py`<br>`vllm/entrypoints/openai/chat_completion/serving.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/platform/patch_kv_cache_interface.py` | integrated | `vllm/v1/kv_cache_interface.py`<br>`vllm/v1/attention/backends/mla/indexer.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/platform/patch_kv_connector_worker_metadata.py` | integrated | `vllm/v1/core/sched/scheduler.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/platform/patch_mamba_config.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/platform/patch_mamba_config_310.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/platform/patch_minimax_m2_config.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/platform/patch_minimax_usage_accounting.py` | integrated | `vllm/entrypoints/openai/chat_completion/serving.py`<br>`vllm/entrypoints/openai/engine/protocol.py` | 公共 usage/full/stream 行为合入服务层；非目标 MiniMax reasoning parser 不激活，归 P4。 |
| `patch/platform/patch_multiproc_executor.py` | integrated | `vllm/v1/executor/multiproc_executor.py`<br>`vllm/v1/executor/ascend_multiproc_executor.py`<br>`vllm/v1/executor/abstract.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/platform/patch_sched_yield.py` | integrated | `vllm/distributed/utils.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/platform/patch_torch_accelerator.py` | integrated | `vllm/utils/mem_utils.py`<br>`vllm/v1/worker/npu_worker.py`<br>`vllm/v1/worker/npu_runner_state.py` | 各使用方显式选 NPU memory API，不改 torch.accelerator。 |
| `patch/worker/__init__.py` | retired_loader | `vllm/platforms/npu.py` | 原插件/patch 聚合加载器退出活动路径；不提供旧 namespace 壳。 |
| `patch/worker/patch_bert.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_cudagraph.py` | integrated | `vllm/v1/cudagraph_dispatcher.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_deepencoder2.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_deepseek_mtp.py` | integrated | `vllm/model_executor/models/deepseek_mtp.py`<br>`vllm/model_executor/models/deepseek_v2.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_distributed.py` | integrated | `vllm/distributed/parallel_state.py`<br>`vllm/distributed/device_communicators/npu_communicator.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_draft_quarot.py` | integrated | `vllm/model_executor/model_loader/ascend/quarot.py`<br>`vllm/model_executor/models/llama_eagle3.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_gdn_attn.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_huanyuan_vl.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_kimi_k25.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_logprobs.py` | integrated | `vllm/v1/sample/ops/logprobs.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_mamba_utils.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_minimax_m2.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_minimax_m2_linear_attn.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_module.py` | integrated | `vllm/v1/attention/backends/gdn_attn.py` | GDN 布尔排序直接转换 dtype，删除 torch wrapper。GDN 模型入口不增加到认证范围。 |
| `patch/worker/patch_multimodal_merge.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_npugraph_ex_triton.py` | integrated | `vllm/compilation/ascend/torchair_backend.py`<br>`vllm/compilation/ascend/compiler_interface.py` | 单个 compiler 的私有函数依赖绑定，不改已安装 TorchAir 模块/类；内网核验私有 API、字典元数据及图执行。 |
| `patch/worker/patch_qwen3_5.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_qwen3_next.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_qwen3_next_mtp.py` | integrated | `vllm/v1/worker/utils.py` | 2026-09-30 更正：GLM DSA 精确 latent/indexer 兄弟对及 MTP 的 NPU KV 绑定逻辑合入原生函数，保留对象引用及原单缓存/其他重复层名语义；不恢复 Qwen 模型支持或全局补丁。 |
| `patch/worker/patch_rejection_sampler.py` | integrated | `vllm/v1/sample/rejection_sampler.py`<br>`vllm/v1/sample/ascend/rejection_sampler.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_routed_experts_capturer.py` | integrated | `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_triton.py` | inactive_non_target | — | 非 GLM-5.2 文本/MTP 依赖，移出加载清单，源码保留待 P4 裁剪；不声称该模型仍获认证。 |
| `patch/worker/patch_unquantized_gemm.py` | integrated | `vllm/model_executor/layers/utils.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_v2/__init__.py` | retired_loader | `vllm/platforms/npu.py` | 原插件/patch 聚合加载器退出活动路径；不提供旧 namespace 壳。 |
| `patch/worker/patch_v2/patch_block_table.py` | integrated | `vllm/v1/worker/npu/v2/block_table.py`<br>`vllm/v1/worker/npu/v2/runner_state.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_v2/patch_eagle.py` | integrated | `vllm/v1/worker/npu/v2/common/spec_decode/eagle/speculator.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_v2/patch_input_batch.py` | integrated | `vllm/v1/worker/npu/v2/input_batch.py`<br>`vllm/v1/worker/npu/v2/runner_state.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_v2/patch_model_state.py` | integrated | `vllm/v1/worker/npu/v2/model_states/__init__.py`<br>`vllm/v1/worker/npu/v2/runner_state.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_v2/patch_triton.py` | integrated | `vllm/v1/worker/npu/v2/common/sample/gumbel.py`<br>`vllm/v1/worker/npu/v2/common/sample/logprob.py`<br>`vllm/v1/worker/npu/v2/common/sample/penalties.py`<br>`vllm/v1/worker/npu/v2/common/input_batch.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_v2/patch_uva.py` | integrated | `vllm/v1/worker/npu/v2/common/buffer_utils.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `patch/worker/patch_weight_utils.py` | integrated | `vllm/model_executor/model_loader/weight_utils.py` | 行为合入正式定义或显式 NPU 分发，不再执行导入时替换。 |
| `platform.py` | retired_loader | `vllm/platforms/npu.py` | 原插件/patch 聚合加载器退出活动路径；不提供旧 namespace 壳。 |
