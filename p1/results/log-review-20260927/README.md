**2026-09-26 补充基线/P1 日志：P2 启动评估**

分析日期：2026-09-27。结论：**暂不建议正式启动会改变运行行为的 P2 原生化改造，P1 出口仍未通过。** 可以并行做 P2 的依赖盘点、符号映射和迁移设计；先修复共同的缓存异常恢复问题与压测统计问题，再建立可用的运行对照。此次没有修改生产源码、部署或运行 NPU 测试。

本次完整扫描 `p1_logs/基线_2P2D_20K_140K_测试` 与 `p1_logs/P1_2P2D_20K_140K_测试` 的 13 个文件：两组各四个 P/D 服务日志、一个客户端日志、一个 benchmark JSON，以及 P1 的 proxy 日志。基线未提供对应 proxy 日志。未将 09-25 的旧 P1 日志混入本次统计。

逐文件 SHA-256、统计结果、故障位置和关键摘录见 [comparison.json](comparison.json)。在工作区根目录运行以下命令可重建统计；只使用 Python 标准库并写入本目录：

```bash
python3 design/p1/results/log-review-20260927/analyze.py
```

**1. 此次已经补充了哪些有效证据**

四个角色的 `non-default args` 在仅归一化 profiler 输出目录后，基线与 P1 完全一致。两组客户端加载的 60 个输入文件及顺序一致，均为 60 个线程、计划 3000 个请求。这比仅有单侧成功日志更有对照价值。

| 项目 | 日志证据与判断 |
| --- | --- |
| 模型及拓扑 | 相同 GLM-5.2-w4a8c8-0723 路径，2P2D、TP8/DP2、EP 开启；两 P、两 D 均有运行日志 |
| 调度配置 | `max_model_len=140000`、`max_num_batched_tokens=4096`、`max_num_seqs=16`、服务 seed=1024 |
| DSA/C8 | runtime KV 分组 `(79, 22)`，KV dtype 为 bfloat16；DSA index payload 也记录为 bfloat16，符合本轮 C8-off 方向 |
| MTP/图 | 单 speculative token；存在 SpecDecoding acceptance 指标和 PIECEWISE/staged 执行路线记录，覆盖了实际执行，不能仅称为启动 smoke |
| 缓存 | RemoteFill commit、实际搬运和外部缓存命中记录存在；但异常恢复不通过 |
| P1 运行来源 | 日志显示 vLLM `0.18.0+ascend.p1`、LMCache `0.4.3+ascend.p1`，堆栈进入两个 P1 仓的 strict editable 路径 |

这些证据支持“P1 合仓后的主要服务链路能够运行”。它们还不能证明权重/config/tokenizer 内容 hash、全部环境和缓存温度一致；两侧 LMCache YAML 路径不同，原始配置文件未随本批日志提供。服务 seed 相同也不代表请求是确定性采样：故障 dump 中可见 `temperature=1.0, top_p=0.95, seed=None, max_tokens=10000`。

配置与元数据位置：两组 D0 日志第 13、58、4531、4759 行；两组 D1 日志第 4133、4361 行。客户端文件/线程数见两组 ansible 日志第 62–68 行。

**2. 首要阻塞：两轮都出现缓存读取失败、TP 异常分歧与 Decode 引擎退出**

| 事件 | 基线 | P1 |
| --- | --- | --- |
| D1 Mooncake page get 失败 | 17:05:58，`kv_group=1`，22 层 index 组 | 16:19:36，`kv_group=0`，79 层 latent 组 |
| rank0 处理 | `RF-D-004`，`action=RECOMPUTE` | `RF-D-004`，`action=RECOMPUTE` |
| passive ranks | TP1–TP7 均报 `Shared CPU cache rank0 error envelope` 的 `ValueError` | TP1–TP7 同样报错 |
| D0 最终异常 | 17:10:57，`sample_tokens` RPC timeout | 16:24:34，`sample_tokens` RPC timeout |
| D1 最终异常 | 17:10:57，`sample_tokens` RPC timeout | 16:24:35，`sample_tokens` RPC timeout |
| 失败时调度快照 | D0：9 running / 20 waiting；D1：7 / 22 | D0：1 running / 25 waiting；D1：2 / 14 |

两侧底层批量读取结果中都有 `-704`，返回值未满足期望字节数。这里不在缺少本次 Mooncake SDK/服务端日志时直接解释该错误码的底层原因。`completed_pages=0` 是失败批次的日志字段，也不能据此断言所有页面都没有写入过；返回数组中还包含正常的字节数。

关键证据：

- [基线 D1](../../../../p1_logs/基线_2P2D_20K_140K_测试/7.150.1.46_D_dp1_port7920.log)：第 32901 行 page get `status=error`；32917 行 rank0 请求重算；32968 行 passive rank 异常；34424 行 RPC timeout。
- [P1 D1](../../../../p1_logs/P1_2P2D_20K_140K_测试/7.150.1.46_D_dp1_port7920.log)：第 36767 行 page get `status=error`；36779 行 rank0 请求重算；36846 行 passive rank 异常；38133 行 RPC timeout。
- [基线 D0](../../../../p1_logs/基线_2P2D_20K_140K_测试/7.150.5.81_D_dp0_port7920.log)：第 44040 行 RPC timeout。
- [P1 D0](../../../../p1_logs/P1_2P2D_20K_140K_测试/7.150.5.81_D_dp0_port7920.log)：第 53319 行 RPC timeout。

重复打印的 traceback 按同一故障去重理解，不能把 14 行 `ValueError` 当作 14 个独立请求失败。两轮的 D1 都涉及 7 个 passive TP rank；D0 没有同样的本地 page-get 错误证据，但最终也超时。

客户端虽然记录了 Ctrl+C，服务端还存在长时间无更新和明确的引擎 fatal，不能将全部异常解释成正常停测。P1 两 D 最后一次 `Activity: generated` 均在 16:18:38，早于 16:19:36 的错误日志；基线最后一次在 17:05:58。因此，日志确认了共同的失败形态，但**尚不能把完整卡顿起点、另一 D 实例的超时都严格归因于同一个已定位的函数**。

优先排查方向是错误处理在所有 rank 上是否一致，而不只是增加 RPC 超时。本机当前 [cache_engine.py](../../../../p1-repos/LMCache/lmcache/v1/cache_engine.py) 中：第 1459 行起将 error envelope 抛为 `ValueError`；第 5314 行起计算 passive 的 `remote_fill_load`；第 5355 行起只有符合该条件才包装为可恢复的 `_RemoteFillMaterializationError`；第 6555、6866 行起分别负责 passive/rank0 重算处理。日志中 rank0 的 RECOMPUTE 与 passive 的直接 ValueError 相符，应核验各 rank 的配置、local-full hint、请求状态与异常传播。

这是一条有源码支持的排查线索，尚不是经 NPU 复现证明的完整根因。源码核对时，本机原仓与 P1 的 `cache_engine.py`、`mooncakestore_connector.py`、`patch_multiproc_executor.py` 内容相同，进一步支持“共同路径缺陷”的判断。核对的本机 HEAD 为 LMCache `547ae7c10b0e510b864c8d0f5233d6f54f279359`、vLLM `230fbc0218e656cb90f8a8c2261150345a1d8ee5`；日志没有提供足够材料证明内网运行的完整提交/制品身份与这两个 HEAD 完全对应。

**3. 压测结果存在统计漏洞，不能按 0 failure 或吞吐汇总放行**

| 项目 | 基线 | P1 |
| --- | ---: | ---: |
| 计划请求数 | 3000 | 3000 |
| 已记录 `success=true` | 125 | 132 |
| 占计划请求比例 | 4.17% | 4.40% |
| `failed` 字段 | 0 | 0 |
| 输入/输出同时为 0、仍记成功 | 3 | 3 |
| 有正输入和正输出的记录 | 122 | 129 |
| 有效记录的输入 token 范围 | 20,214–129,265 | 20,214–129,265 |
| 报告 `duration`，秒 | 28,981.06 | 27,096.06 |
| 逐请求 `latency_seconds` 求和，秒 | 28,981.059 | 27,096.055 |

3000 是计划量，大部分尚未执行或未形成结果；不能把未记录请求全部算为失败，也不能据此宣布 3000 请求成功。两轮客户端均明确打印 `Received Ctrl+C` 和 `Interrupted!`。

**duration 与并发请求耗时之和吻合，并非墙钟耗时。** 报告中 2.41 / 2.19 tok/s 和 223.57 / 240.11 total tok/s 使用了这个分母，不能用于集群吞吐验收；也不能把 28,981 秒当成已完成八小时长稳。客户端文件名到结果时间戳仅约 19 分 18 秒 / 23 分钟，且这些文件名本身也不能替代精确的计时埋点。现有材料不足以可靠恢复整轮吞吐。

两轮 D 日志各有三次 HTTP 400，原因是输入至少 130001 tokens，加 `max_tokens=10000`，超过 `max_model_len=140000`。140K 是输入加输出的上下文上限。有效压力请求需要控制这个总预算；超限输入应单列为负向用例。

两轮 JSON 中的三个零 token “成功”具有相同请求 ID。P1 的 [proxy 日志](../../../../p1_logs/P1_2P2D_20K_140K_测试/p174_proxy_tp8_dp2.log) 第 26–28、214–216、484–486 行显示，客户端侧 HTTP 200 与下游 HTTP 400 同时存在。当前 [proxy 源码](../../../../p1-repos/vllm/ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_enhanced.py) 第 2840 行起会把异常转成 SSE error 后发送 `[DONE]`。这支持“客户端将错误流当作正常结束”的排查方向；本批没有 benchmark 客户端源码及逐请求 SSE 原文，不能完成每条错误流与 JSON 记录的一一核验。无论如何，`failed=0` 不足以证明端到端成功。

另有 P1 proxy 第 814–815 行的两条流中断错误。必须在复测时同时记录 HTTP、SSE error、完成原因、超时/取消、在途状态及 token usage，不能只凭 HTTP 200 或 `[DONE]` 记成功。

**4. 性能有改善信号，也有尾延迟回退信号，尚不构成等价证明**

全体已记录结果的平均 TTFT 为 181.33 → 158.70 秒（-12.48%），平均 E2E 为 231.85 → 205.27 秒（-11.46%）。但两轮完成请求集不同：共同 ID 为 112 个，基线独有 13 个、P1 独有 20 个。全体样本的 TPOT p95 为 108.44 → 203.84 ms（+87.97%），平均值掩盖了尾部变化。

进一步按共同请求 ID 配对，排除两侧相同的三个零 token 记录，得到 109 对。其输入 token 数、context 字符数、消息数等记录字段相同；99 对输出长度不同，且未提供输出 token 序列，不能判定数值等价，也不能把输出长度差直接当作质量错误。

| 共同有效请求的指标 | 基线 | P1 | P1 相对变化 |
| --- | ---: | ---: | ---: |
| TTFT mean | 184.44 s | 168.73 s | -8.52% |
| TTFT p50 | 182.27 s | 161.86 s | -11.20% |
| TTFT p95 | 330.48 s | 295.57 s | -10.56% |
| TPOT mean | 85.88 ms | 92.61 ms | +7.83% |
| TPOT p50 | 90.05 ms | 87.27 ms | -3.09% |
| TPOT p95 | 106.28 ms | 114.36 ms | +7.60% |
| TPOT p99 | 116.03 ms | 206.09 ms | +77.62% |
| E2E mean | 231.35 s | 208.03 s | -10.08% |

这些是单轮、提前终止、仅已完成请求的观察值，存在截尾偏差；配对请求也没有消除输出长度、路由、缓存命中和批次调度差异。因此既不能认定 P1 已满足性能门槛，也不能据此确认 P1 原生代码引入了上述幅度的独有回归。

百分位按 `(n-1)*p` 线性插值重算。原报告的 ITL 是每请求 `itl_ms` 的分布，不等同于全体 token 间隔分布：基线/P1 仅 75/78 条记录有该字段；共同非空样本为 62 对。没有核验客户端 token/chunk 计时实现前，不将其包装成完整 token 级 ITL 验收。

[现有性能标准](../../../03-baseline-and-validation.md) 第 184–200 行要求固定条件、预热、至少三轮重复，并提出 5% 回退建议值。文件明确该建议值需先随基线冻结，本批没有冻结证明；本报告未把它擅自升级成已批准硬阈值。+7.60% 的配对 TPOT p95 是需要复测解释的信号。

**5. 启动 P2 前的处理顺序**

| 优先级 | 工作 | 应取得的结果 |
| --- | --- | --- |
| 1 | 修复/定位 Mooncake 读取失败后的跨 rank 恢复；同时核查本次 `-704` 来源 | 分别覆盖 group0/group1，非输出 TP rank 失败，以及两 D/EP 联动；按协议一致重算、失败或成对重启，不能一部分继续执行、一部分抛异常后拖到超时 |
| 2 | 修正 benchmark 的成功判定和墙钟计时；明确输入+输出预算 | SSE error、HTTP 400、流中断与取消能正确统计；成功/失败/取消/在途可对账；保存精确起止时间、完整参数和输出 |
| 3 | 在固定配对版本上重跑同条件对照 | 有效正向负载自然完成；冷/热缓存分开、固定采样/输入/输出上限，至少三轮，解释 TPOT 尾延迟和资源变化；另补确定性输出对照 |
| 4 | 补齐 P1 既定出口证据 | 两包 wheel 与 sdist 重建、干净安装及 ABI/加载；TP4/DP4 对照；无缓存离线/在线、故障恢复、资源与长稳结果，并归档运行 SHA/制品与模型/config hash |

第 1 项是本批直接观测到的运行故障，第 2 项是本批结果无法可靠验收的原因；第 4 项属于仍缺失的既定证据，不是断言这些能力必然失败。无需为了复测把 3000 条中的非法超长请求包装成正常成功；可先冻结一组规模可控的有效用例自然跑完，再扩大压力与长稳负载。

依据 [P1 出口要求](../../README.md) 第 102–109 行，以及 [阶段流程](../../../02-refactoring-workflow.md) 的 P1/P2 定义，当前适合继续关闭 P1 问题并准备 P2 迁移材料。直接改变 Runner、平台补丁、图或通信路径，会把已有的跨 rank 故障与结构迁移混在一起，失去可靠对照。

**本机验证范围：** 已复算两个 JSON 的请求/输入/输出总数、均值与百分位，核对 112 个共同请求的字段、四角色启动参数和 13 个输入文件 hash；未用本机统计代替内网功能或性能复测。
