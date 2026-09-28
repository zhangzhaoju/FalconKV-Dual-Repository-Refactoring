# P2：vLLM 原生 Ascend 源码交付

2026-09-27：已按用户要求完成 P2 阶段源码改造、可在本机执行的检查及内网验证交接。两仓工作分支均为 `p2`，原 `p1` 和 `main` 保留不动。**源码工作完成；构建、NPU/ABI、模型正确性和性能验收待用户在内网统一执行。** P1/基线复测及已知问题修复继续后置，P1 出口没有改为通过。

优先阅读 [内网执行指南](intranet-validation.md)。配对提交见 [交付清单](baseline/p2-native-integration-20260927.json)，本机结果见 [验证记录](results/native-integration-20260927/README.md)，可转入内网的增量 Git 包、源码快照与校验值见 [源码交付包](deliveries/native-integration-20260927/README.md)。

## 配对版本与保留基线

| 仓库 | P2 包版本 | 保留的 `p1` / `main` 提交 |
| --- | --- | --- |
| `p1-repos/vllm` | `0.18.0+ascend.p2` | `230fbc0218e656cb90f8a8c2261150345a1d8ee5` |
| `p1-repos/LMCache` | `0.4.3+ascend.p1` | `547ae7c10b0e510b864c8d0f5233d6f54f279359` |

LMCache 原生化属于 P3；本次仅同步安装配对约束，以及诊断桥、live-source event handoff、共享 Mooncake transport 三处对 vLLM 的引用。LMCache 包版本沿用 P1，但必须使用交付清单中的新提交，不能拿旧同版本 wheel 混装。目录名 `p1-repos` 沿用历史，不代表当前检出的是 P1。

后续 P1 复测从保留分支建立独立 worktree；修复另建分支，并记录向 P2 移植的提交。原四仓、基线日志和原始报告未修改，本次没有推送远端、替换服务或执行安装。

## P2 完成范围

| 工作 | 最终实现及行为 |
| --- | --- |
| 平台、配置、量化 | 内置 `NPUPlatform` / `PlatformEnum.NPU`，`vllm.config.ascend`，声明式量化选择；不依赖 Ascend entry point，关闭可选插件仍保留 NPU 平台 |
| 平台有效补丁 | canonical MLA spec、双组布局与合并校验、worker metadata 先交付后消费输出、HCCL coordinator、GLM tool parser/usage、恢复预算及调度配置均进入源码定义 |
| Runner 公共契约 | 公共 input batch、异步输出及图类型；v1/v2 原生 NPU state 和 Runner，取消 GPU Runner 继承，使用显式 NPU stream/event/memory API |
| 原生执行部件 | Attention/MLA/SFA、算子、采样、量化、ACL 图、HCCL/并行、LoRA 相关依赖迁入 `vllm`；不可变 NPU layer 表替代 OOT 注册，使用 `forward_npu` |
| MTP 与权重 | DeepSeek MTP/rot.weight、加载器、Quarot 实例内加载、rejection/logprob 行为合入；DSA latent/indexer、CPU KV offload、checkpoint、RemoteFill callback 与恢复路径保留 |
| 全局补丁移除 | 无活动旧 patch 导入、GPU 模块函数替换或 CUDA wrapper；TorchAir 字典支持改为单个 compiler 的私有依赖绑定，不修改已安装依赖 |
| 命名空间与制品 | 活动 Python 代码统一为 `vllm.*`；扩展为 `vllm._ascend_C`，CANN/图/配置资源以 `vllm` 为安装锚点；wheel 拒绝旧 namespace 和补丁归档 |
| 测试与交接 | 迁移相关 UT/e2e 的入口；新增原生契约、编译器隔离、旧 wheel 内容拒绝测试；提供静态门禁、内网真实导入/子进程/设备探测工具 |

本次迁移盘点覆盖原命名空间的 355 个文件：305 个位于原生归属，49 个插件/补丁文件保留为非运行归档，1 个只含 CUDA wrapper 的文件删除；另外记录了 42 项 Runner/组件提取来源。完整文件、哈希、前缀映射和继承方法清单分别见 [最终迁移清单](baseline/native-integration-migration.json)、[原始搬迁记录](baseline/namespace-migration.json)、[Runner 提取记录](baseline/runner-extraction.json)。

[补丁处置表](patch-migration.md) 逐个覆盖全部 49 个归档文件：27 项有效行为合入，6 个聚合/插件入口退出，16 项非目标模型补丁停用并待 P4 裁剪。参考归档不进入 wheel/sdist；不存在空的 `vllm_ascend` 兼容壳。迁移脚本是一次性变更过程记录，**不要对最终工作树重新执行**；`record_migration.py` 和结果目录的 `verify.py` 可重复运行。

## 保留的边界

- 认证目标仍是 GLM-5.2 文本、910B3、DSA 双组/MTP 开启、C8 关闭，2P2D 的 TP8/DP2 与 TP4/DP4。GLM-5.3、其他模型和其他芯片未新增认证。
- GPU/CPU/其他模型原文件、310P 和共享多模态类型的整体裁剪属于 P4；本次解除原生执行对 GPU Runner 和设备 wrapper 的依赖，不借源码搬迁删除未验收的依赖。
- `torch.ops._C_ascend`、`libvllm_ascend_kernels.so`、CANN vendor 名、已有诊断输出目录和环境变量继续沿用 ABI/运维协议名，不是旧 Python 插件依赖。
- 层对象自己的权重后处理、分片 forward、回调等实例行为保留；真实 torch 算子注册也保留。这些与修改其他模块或已安装第三方类的兼容性补丁不同。
- TorchAir 的私有 compiler/converter API 有显式契约检查。本机完成隔离及字典传递测试，内网仍须核对实际 TorchAir 版本并跑 npugraph_ex；不以接口 stub 证明真实图执行。

## 验证状态

本机 vLLM 主机测试 **167 项通过**，LMCache 配对安装测试 **22 项通过**，合计 **189 项**；vLLM 另有 43 个 subtest 通过。一个依赖 torch 的 case 和两个依赖 torch 的测试文件未在本机执行，内网命令明确包含这些项目。静态门禁覆盖 356 个原生 Python 文件；生产/工具变更执行选定 Ruff 规则及语法检查。完整范围、日志和限制以 [本批验证记录](results/native-integration-20260927/README.md) 为准。

本机没有安装 torch/CANN、构建 wheel/sdist、编译扩展或连接 NPU。上述结果不代表运行等价、性能达标或 P2 出口通过；用户统一验证完成后再归档结论。历史 [首批平台结果](results/native-platform-20260927/README.md) 保留当时语义，不用它替代本批全量源码检查。

## 后置的 P1/基线问题

原始证据见 [日志评估](../p1/results/log-review-20260927/README.md)。用户已调整为先完成 P2，原报告的故障事实仍有效：

| 待办 | 后续证据 |
| --- | --- |
| Mooncake `-704` / RF-D-004 后 TP rank 恢复分歧及两 D 引擎 RPC timeout | 各 rank、latent/indexer 两组一致重算或失败/成对重启 |
| 零 token SSE 错误计成功，duration 汇总口径错误 | 修复统计口径后重算成功率与墙钟吞吐；请求/取消可对账 |
| TPOT p95/p99 回退信号 | 固定请求、输出、采样、缓存温度和环境重复对照，排除截尾样本影响 |
| P1 正式验收不完整 | 制品/配置身份、ABI、数值、两种 TP/DP 布局、长稳证据补齐 |
