# P4 RoPE 启动失败分析与修复（2026-10-04）

本报告保留 RoPE 修复这一轮的证据与配对。随后权重加载后的 `aclnnAddRmsNormBias` 失败
已在 [算子注册报告](custom-opp-startup-fix-20261004.md) 单独分析；最新配对和执行步骤以
[配对清单](baseline/p4-pair.json) 及 [冻结 P4 复测指南](intranet-validation-frozen-p4.md#custom-opp-retry) 为准。

## 结论

本次直接阻断点是 P4 裁剪破坏了 `get_rope()` 的互斥分支，导致合法的
`rope_type="default"` 创建对象后仍无条件执行异常。不是 checkpoint 缺失 RoPE 配置，
也不是 MTP 参数、LMCache 安装或通信超时问题。
已修复 RoPE 工厂，并扩大同类控制流审计，修复另外四处可由源码和主机测试复现的问题。
不修改模型参数、不关闭 RoPE/MTP/DSA、不恢复已退休能力。

两仓 `p4` 已本地提交，尚未推送；主机检查通过，NPU/2P2D 验收仍待内网完成。

## 日志证据

日志目录：`p4_logs/logs_all_no_rope`。四个节点首个同类异常如下，时间取日志原文。

| 角色 | 日志文件 | 首个异常行 / 时间 |
| --- | --- | --- |
| P DP1 | `7.150.5.207_P_dp1_port7910.log` | 262 / 10-04 09:09:38 |
| P DP0 | `7.150.7.133_P_dp0_port7910.log` | 253 / 10-04 09:09:58 |
| D DP0 | `7.150.1.218_D_dp0_port7920.log` | 238 / 10-04 09:10:01 |
| D DP1 | `7.150.3.14_D_dp1_port7920.log` | 237 / 10-04 09:10:17 |

调用链为 `NPUWorker.load_model` → `NPUModelRunner.load_model` → `get_model` →
`DeepseekV2DecoderLayer` / attention 构造 → `get_rope` →
`ValueError: Unknown RoPE scaling type default`。
旧源码位于 `vllm/model_executor/layers/rotary_embedding/__init__.py:140`。

日志显示已越过上一轮 MTP Runner 状态初始化，进入模型构造；这不代表 MTP 推理已验收。
配置仍是 `.p4`、GLM-5.2、TP8/DP2、`load_format=auto`、最大长度 140000、
MTP 1 token 和原生 `LMCacheConnectorV1`。Prefill eager / Decode PIECEWISE 均遇到相同异常。
日志未提供本轮运行的精确 Git SHA，复测必须用配对及安装源码哈希检查补齐身份记录。

## 根因与扩大检查

首轮 P4 自动裁剪把 `elif` / `else` 拆成独立语句。上一轮已修复裁剪工具，
但工具修复不会自动修复此前生成的其他运行文件。本次 RoPE 工厂中 default、linear、yarn
分支不再互斥，末尾的异常无条件执行，后续缓存写入与返回不可达。
修复恢复 `if/elif/else`，保留既有 default、linear、yarn、deepseek_yarn、
deepseek_llama_scaling 构造参数和缓存语义；多模态及不支持类型仍明确拒绝。

为检查同源问题，在内存中分别用旧、新裁剪工具转换冻结 P3 源码，并比较 AST：
旧工具取设计仓 `e7388f9`，已修复工具取 `df6bcbf`；输入取两仓 `p3-frozen-20261003`。
范围为 P3→P4 仍存在且修改过的包内运行 Python 文件：vLLM 195 个、LMCache 17 个，
共 212 个，未跳过文件、无转换异常。没有重新运行全仓写入式裁剪。

共发现 13 处方法级差异，处置如下。另将函数体替换为空语句比较模块/类结构，未发现额外差异。

| 处置 | 文件/方法与理由 |
| --- | --- |
| 本轮修复：RoPE | `get_rope` 恢复互斥和返回；对应本次四节点首个致命错误 |
| 本轮修复：权重准备 | `DefaultModelLoader._prepare_weights` 的 `pt` 恢复为 `elif`，避免 auto→hf / safetensors 误入未知格式异常 |
| 本轮修复：权重迭代 | `safetensors_weights_iterator` 恢复 eager 与 safe_open 的互斥，防止同一张量产出两次 |
| 本轮修复：Ray | `initialize_ray_cluster` 恢复初始化保护；已初始化时不再次调用 `ray.init` |
| 本轮修复：缓存尾块 | `LMCacheConnectorV1Impl._wait_for_save_impl` 仅对非最后一次 prefill 向下对齐 chunk；最后一次保留 token/mask/slot_mapping 尾部及完成标记 |
| 上轮已修复，保持 | `NPUModelRunnerState.__init__`、`NPUModelRunner.propose_draft_token_ids` |
| 当前代码已手工改写，保持 | `ModelConfig.attn_type`、`SpeculativeConfig.__post_init__`；当前保留正确分支或 MTP 专用逻辑 |
| 无语义差异，保持 | `VllmConfig.__post_init__`、`EngineArgs._set_default_chunked_prefill_and_prefix_caching_args`：仅空语句差异 |
| 旧方法已移除，保持 | `UnquantizedFusedMoEMethod.process_weights_after_loading` 不再是当前原生实现入口 |
| 最终状态相同，保持 | `LMCacheEngine._common_init` 的中间 fmt 赋值后仍被相同的 MLA 条件覆盖，中间没有调用 |

权重加载、Ray 和尾块保存问题不是这份日志已触发的错误；它们是扩大审计发现的同源隐患。
其中 Ray 非当前 multiprocess 启动路径，缓存修复不恢复 CacheBlend。
此差分审计限定于上述文件和裁剪器差异，不是对所有裁剪缺陷、动态导入或硬件行为的完整证明。

## 该轮配对与验证

| 仓库 | 分支 | 精确提交 | 包版本 |
| --- | --- | --- | --- |
| vLLM | `p4` | `8a5c71069ab4d80c64c60e9375af04f8a7b4da9d` | `0.18.0+ascend.p4` |
| LMCache | `p4` | `241166dfde710e9a2cbe24ba0bd24b094b4bbe41` | `0.4.3+ascend.p4` |

该轮配对包含此前 MTP 与 SoC 修复；[配对清单](baseline/p4-pair.json) 随后已更新为最新修复。
P3 冻结标签及 P1/P2/P3/main 均不变；未修改原始四仓、日志、模型或内网服务。
本次未改变运行模块集合、native 源码或 ABI。启动身份检查增加四个 vLLM 修复文件；
跨仓配对检查另覆盖全部已跟踪包内 Python 文件，包括 LMCache adapter。

[主机报告](results/rope-startup-20261004/verification.json) 9/9 项通过：

- vLLM：334 passed、300 subtests passed、1 deselected。
- LMCache：164 passed、38 subtests passed。
- 裁剪工具：25 passed。

本轮新增 vLLM 48 项回归，覆盖 RoPE 选择/参数/缓存/拒绝边界、权重格式与迭代、Ray 初始化保护和身份检查；
LMCache 19 项回归从真实公共 `wait_for_save` 入口覆盖完整/不足 chunk、缓存前缀及 final/nonfinal 保存；
裁剪工具增加 11 项分支语义用例，并对生成器消费后的输出与副作用进行比较。
针对运行文件，修复前新增用例分别有 28 项和 5 项失败，修复后全部通过。
新增测试通过 lint，修改的运行文件通过未定义名称检查，diff 检查通过。

测试执行提取的生产函数/方法，设备、模型、Ray 和缓存后端使用替身；
不验证 RoPE 数值 kernel、实际权重加载、真实尾块传输或 NPU 性能。
本机 Python 3.12，未安装 torch/CANN；内网 Python 3.11 的全部选定 CPU tensor 回归需用
`verify.py --with-installed-torch` 补跑。不能把本次主机结果等同于 CANN/ABI/NPU 或 P4 验收通过。

## 内网下一步

按 [冻结 P4 指南 §3.2](intranet-validation-frozen-p4.md#mtp-startup-retry) 发布两仓 `p4`、上传新 `design/p4`，
并在四个测试容器逐一切换至最新清单配对，不再按本报告旧 vLLM SHA 运行。
LMCache 至少需要本报告的 `241166df` 缓存修复。
已正确安装 P4 strict editable 且源码身份检查通过时，无需重编译原生扩展或重装；
先停止本轮自己的测试进程，再更新源码并启动新进程，不能在运行服务下热切换。
普通 wheel 必须重新打包安装两仓；首次从 P3 升级 P4 仍需完整原生重建。

保留原 checkpoint/启动配置，检查四实例完成权重加载、KV 初始化/预热和请求输出，
然后核对 DSA 双组、MTP 接收计数、decode 并发、缓存尾块/RemoteFill 与抢占恢复。
归档精确 SHA、安装路径/哈希、启动参数、四节点日志和同负载性能，继续完整 P4 验收矩阵。
