# P4 源码整改交付报告（2026-10-03）

本轮已完成本地 P4 分支、源码裁剪、主机回归和交付材料；状态是**待内网验收的源码候选**。
未执行 CANN 编译、NPU/模型推理或远端推送，未停止或覆盖已有服务。

本报告下述提交、裁剪统计和测试数保留首次交付快照。
2026-10-03 后续已修复 LMCache 的 SoC 大小写门槛，当前 LMCache 提交为
`1fc7439b96d091378b527819dc92902115eee58d`，vLLM 保持 `0879e419001f424700569936962631ca0360f5af`。
当前精确配对以 [p4-pair.json](baseline/p4-pair.json) 为准；
修复后的 [主机报告](results/lmcache-soc-case-20261003/verification.json) 8/8 项通过，
LMCache 增至 145 passed / 38 subtests，vLLM 测试数不变。
新增 15 项真实 CMake script-mode 回归；CANN 编译、ABI 和功能验收仍待内网完成。
原因与执行步骤见 [内网指南 §3.1](intranet-validation.md#lmcache-soc-retry)。本次修复未推送远端。

## 1. 冻结与配对版本

两仓均以 `p3-frozen-20261003` 冻结 P3，然后创建 `p4`；P1/P2/P3/main 分支未前移。
原四仓、模型权重和历史日志未删除。

| 仓库 | 冻结 P3 | 首次 P4 交付 | 包版本 |
| --- | --- | --- | --- |
| vllm-dual | `8767fe1121a085e8cefed03eeaa1444b55227521` | `0879e419001f424700569936962631ca0360f5af` | `0.18.0+ascend.p4` |
| LMCache-dual | `a4e2131e890727edadb6bb61cde369900f72c3fd` | `1bedaf89564b3d4b2f0de67187111b03a037a716` | `0.4.3+ascend.p4` |

机器可读的提交、tree、版本与范围见 [p4-pair.json](baseline/p4-pair.json)。
内网按精确 SHA 检出并逐容器核对，不能只看包名中的 `.p4`。

## 2. 完成的整改

- 设备入口和 native 构建仅接受 910B3；删除其他厂商平台、算子/通信实现及 310P/A3/A5 专用路径。
- 模型注册仅开放 GLM-5.2 原生文本及其内部同 checkpoint MTP；删除其他模型、蒸馏、多模态、pooling、LoRA、训练入口与非 MTP 推测实现。
- 移除其他模型配置改写、Mamba 状态执行、causal-conv1d 和 LoRA/Gemma 专用算子绑定；保留 GLM 所用 DeepSeek/Eagle 共享层。
- LMCache 删除 SGLang/MindSpore、CacheBlend、旧 v0、独立 GPU multiprocess 服务及 GPU 专用存储/传输/序列化实现。
- 保留 DSA 双组、MTP、CPU KV 卸载/共享、跨实例缓存、P/D、RemoteFill、checkpoint/恢复；保留原生连接器的完成通知、冷加载准入和 staged-SFA event handoff 回归。
- 删除对应非目标文档、示例、测试、依赖和旧 CI；两仓仍保留启用的 GitHub Actions，但 P4 workflow 仅运行源码/主机检查，不在 GitHub 构建 CANN 或连接内网模型。
- `p1_dev.py` 文件名不变，已升级为 P4 配对构建和 strict editable 安装入口；每次 native 构建使用新 staging 目录。

保留少量请求 DTO、配置字段、layout 枚举和历史命名用于序列化/类型契约或明确拒绝旧配置，
不提供被删能力的执行实现。CPU 主机计算及 KV 存储不是 CPU 推理后端。
CATLASS/kvcache-ops 等共用材料保持已审核的固定版本，其通用定义不扩大硬件支持范围。
GLM-5.3 尚未纳入支持；架构白名单不能代替内网 checkpoint 身份核验。

## 3. 删除范围与可恢复性

以冻结 P3→本次 P4 的 `git diff --no-renames` 统计，包含源码、测试、文档和 CI：

| 仓库 | 删除路径 | 新增路径 | 修改路径 | 跟踪路径：裁剪前→后 |
| --- | ---: | ---: | ---: | ---: |
| vllm-dual | 4382 | 9 | 232 | 5907→1534 |
| LMCache-dual | 730 | 8 | 38 | 1548→826 |

不是模型文件或运行缓存的删除量。上述删除均可从冻结标签恢复。
[最终统计](baseline/final-pruning-summary.json) 包含分目录统计、删除路径集合哈希及复现命令。
分批 `*-prune-batch*.json` 是执行审计记录；早期候选中恢复的原生 MultiConnector、
IdentityReasoningParser 以最终提交为准。未删除用户权重、基线日志或正在运行的资源。

## 4. 已执行验证

首次交付入口为 [handoff-20261003/verification.json](results/handoff-20261003/verification.json)，8/8 项通过。
其他 `results/host-*` 目录是整改中间记录（包含曾发现并修复问题的失败报告），不能替代该交付报告。

| 检查 | 本机结果与边界 |
| --- | --- |
| vLLM 主机测试 | 264 passed、300 subtests passed，1 deselected；不含下面列出的 torch 测试文件 |
| LMCache 主机测试 | 130 passed、38 subtests passed；不含下面列出的 torch 测试文件 |
| 源码导入/导出与 profile 检查 | vLLM 774 个、LMCache 311 个包内 Python 文件；动态导出/native 模块单列为未验证 |
| native 归属检查 | 两仓通过；LMCache 保留 6 类、663 项方法契约，2 项 310P 方法显式退休 |
| vLLM 原生绑定静态一致性 | 46 个 schema、46 个实现、34 个 Meta；Meta 均有对应 schema，保留 DSA/MTP 必需算子 |
| 未定义名称 | 两仓 `ruff check --select F821` 通过；不是完整类型或全部 lint 检查 |
| shell / diff | 两个 CANN 构建脚本 `bash -n`、两仓 `git diff --check` 通过 |
| LMCache 文档 | Sphinx 7.2.6 `-E -W --keep-going` 构建通过，核对 HTML 目录、安装命令和链接 |

本机测试解释器为 Python 3.12.3；产品构建目标仍是内网 Python 3.11/aarch64。
本机未安装 torch/CANN，以下留给内网 `verify.py --with-installed-torch` 补跑：

- vLLM `test_cold_resume_native_metadata.py`、`test_glm52_topk_ownership.py`。
- vLLM 用例 `production_draft_expansion_can_exceed_target_capacity`。
- LMCache `test_glm52_metadata.py`。

主机测试证明的是选定 Python/契约路径，不证明 NPU 数值结果、ABI、性能或 2P2D 长稳。

## 5. 下一步

按 [P4 内网执行指南](intranet-validation.md) 依次进行：发布源码及冻结标签、精确配对检出、
四容器重新编译安装、源码/制品身份核对、冷导入与全部选定 CPU 回归，然后模型/缓存回归。
需重新编译两仓，不能复用旧 `.so` 或仅切换 editable 指向的分支。

必选验收仍包括 wheel/sdist 重建、离线生成、无缓存在线、2P2D TP8/DP2 与 TP4/DP4、
DSA/MTP、CPU 共享缓存/跨实例 RemoteFill、抢占与冷恢复、故障与长稳、负向边界及性能对照。
现有 P3 日志不能补齐全部矩阵，也不能用延迟求和的 duration 计算并发吞吐。
完整结果审核通过后才能宣告 P4 阶段验收完成。
