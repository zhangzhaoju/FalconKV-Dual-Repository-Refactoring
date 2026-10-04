# P4 Prefill / Decode Waiting 对比复核（2026-10-04）

## 结论与处理

本次用户观察依据是“以前 Decode 有 Waiting Req，现在没有”。归档日志中 **P4 两个
Decode 节点仍有 Waiting，Running 也并非被限制为 1**。按相同早期窗口比较，未发现
能够支持“Prefill 明显变慢”的证据；目前不能据此认定存在需要修改推理代码的性能回归。

主要比较偏差是：P4 Decode 指标只覆盖约 4 分钟，而 P3 覆盖约 16 分钟；P3 后段缓存
命中显著增加，同时出现大量 Decode 排队。这些阶段不应直接比较。P4 文件截断并不证明
测试只运行了 4 分钟，也不能据此推断它后续一定达到 P3 性能。

本轮修正采集/比较流程，新增只读分析工具和回归测试；**不修改两仓运行代码，不调整
TP/DP、MTP、DSA、并发预算或缓存策略，不人为制造 Waiting 作为“修好”的判据**。
本机候选配对仍为 vLLM `137fd3bc144875075ae73be2782ec8b5abf5a002` /
LMCache `241166dfde710e9a2cbe24ba0bd24b094b4bbe41`。
仅此次工具/文档更新不要求重新编译或 editable 重装；此前安装修复仍按原指南执行。
内网各容器实际 SHA/安装路径还需随复测归档，不能由 `.p4` 包版本推断。

## 1. 输入与证据有效性

- 基线：工作区 `p3_logs/logs_all_success`。
- 候选：工作区 `p4_logs/logs_all_prefill_lower`。
- 完整计算结果、输入 SHA256、指标对应的物理行号见
  [analysis.json](results/prefill-review-20261004/analysis.json)。

候选目录中的以下文件与 P3 **逐字节相同**，不能作为本轮 P4 压测结果：

| 文件 | SHA256 |
| --- | --- |
| `7.150.3.14_benchmark_20261003_151741.json` | `752d1d2b34312989f189f0777782dcda968dd468685282254fd7fbecc990fb64` |
| `7.150.3.14_ansible_test_tp8_dp2_20261003_150005.log` | `46f1cef538a1294c463505c02736dd0d918a4a993c5cbf8c381178571ee32d39` |

本轮新客户端日志是 `7.150.3.14_ansible_test_tp8_dp2_20261004_152813.log`：
声明 60 文件 / 3000 请求，但当前截取只有 31 条完成记录，其中 30 条正输出、1 条
零输入/零输出；没有对应的新完整 benchmark JSON。因此尚不能做整轮吞吐/TTFT 分位数验收。
基线也只有部分请求结果，不能把计划的 3000 请求当作已完成数量。

## 2. P4 的 Waiting 没有消失

可直接定位候选日志：

- `7.150.1.218_D_dp0_port7920.log:5738`：10-04 15:28:50，Running=1，Waiting=1。
- `7.150.3.14_D_dp1_port7920.log:5294`：10-04 15:29:16，Running=0，Waiting=1。

下表分别以每个 D 节点第一条吞吐日志为相对零点，使用 `[0, T)` 窗口。D0 比较前
250 秒；D1 候选覆盖不足 250 秒，双方统一截取前 232 秒，不补零、不延长候选数据。

| 节点 / 窗口 | P3 最大 Running | P4 最大 Running | P3 最大 Waiting | P4 最大 Waiting | Waiting>0 的采样条数 P3 / P4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D0 / 250 秒 | 7 | 7 | 1 | 1 | 16 / 17 |
| D1 / 232 秒 | 9 | 8 | 2 | 1 | 22 / 13 |

采样条数不是等待秒数，日志可能存在采样缺口。全日志比较则会得到 P3 最大 Waiting
16/26 与 P4 1/1，但对应的观察时长分别是 999/955 秒与 251/232 秒，不能直接归因到 Prefill。

当前 `vllm/v1/core/sched/scheduler.py:2296` 中 Waiting 为
`len(self.waiting) + len(self.skipped_waiting)`，这个口径与冻结 P3 相同。
它表示本地调度器的瞬时队列，受请求到达、KV 就绪、Decode 完成速度和输出长度等共同影响，
不是 Prefill 算力或服务健康指标。没有持续排队也可能是供需均衡，不能单独判定性能下降。

## 3. 同阶段 P 侧完成量接近，P3 后段缓存状态明显不同

P 节点分别以自身第一条 `scheduler_lookup` 的单调时钟为零点，取前 250 秒。
只匹配同节点、同 clock domain 的 lookup 与完整 prompt 的
`remote_fill_producer_terminal`，按请求/transfer 去重：

| 节点 | P3 terminal 数 | P4 terminal 数 | P3 完成请求的未缓存 token | P4 完成请求的未缓存 token |
| --- | ---: | ---: | ---: | ---: |
| P0 | 22 | 19 | 1,178,028 | 1,180,470 |
| P1 | 20 | 23 | 1,179,666 | 1,160,388 |
| 合计 | 42 | 42 | 2,357,694 | 2,340,858 |

完成请求的未缓存 token 合计相差约 **-0.71%**，暂不支持“明显变慢”的判断。
但 terminal 是 P 侧持久化完成边界，不是纯模型/NPU kernel 计时；边界上的在途工作未计入，
两个 P 窗口也不是共同墙钟窗口。此估算不能替代系统吞吐，不能据此宣告性能等价或验收通过。

按 lookup 事件的 prompt token 加权，P0/P1 的 LMCache 命中比例为：

| 阶段 | P3 P0 / P1 | P4 P0 / P1 |
| --- | --- | --- |
| 前 250 秒 | 10.40% / 11.76% | 10.43% / 11.51% |
| 250 秒之后的剩余采集 | 90.66% / 82.67% | 12.72% / 0%，仅 2 / 1 条 lookup |

这是 lookup 事件加权比例，可能包含重复查询，不能冒充去重后的请求命中率。P3 后段缓存
命中更高，可以减少重复 Prefill 计算并加快向 D 侧提供请求；该阶段也确有较大的 D 队列。
这支持“冷热阶段/窗口不同”的解释，但 P4 后段证据不足，还不能确认其热缓存稳态性能。

已核对两轮 P 节点的非默认 CLI 参数一致：TP8/DP2、MTP=1、4096 批 token、
`max_num_seqs=16`、eager、模型与量化配置相同；LMCache 配置除实例随机标识外一致。
两轮均打印 DSA 双组开启，D 侧均有 staged graph 路由证据。没有证据支持用关闭 MTP/DSA、
调整并发或重新改动上一轮 OPP 注册修复来处理本次现象。

另需区分：`Avg prompt throughput` 使用 computed prompt token，并在收到相应请求的
prefill 输出时累计；长 prompt 分块期间可能表现为连续零值后出现峰值。
该语义见当前 `vllm/v1/metrics/loggers.py:144` 与 `vllm/v1/metrics/stats.py:344`，
并非 P4 新引入的计数变化。客户端 TTFT 还包含排队、P/D 交接等时间，不能直接当纯 Prefill 耗时。

## 4. 本轮交付与复测

新增 [analyze_prefill.py](tools/analyze_prefill.py)：仅依赖标准库，不导入 torch、不访问服务。
它检查四节点输入、clock domain、重复 terminal、完整保存边界，缩短双方共同窗口，
标记重复 benchmark/客户端制品，并拒绝覆盖已有输出。解析失败不会被默认为“零请求”。
报告不自动给出性能 PASS/FAIL，也不证明所有运行时错误都不存在。

```bash
cd /workspace/zzj || exit 1
P4_COMPARE=$(mktemp -d /tmp/p4-prefill-review.XXXXXXXX)
/usr/local/python3.11.14/bin/python -B design/p4/tools/analyze_prefill.py \
  --baseline p3_logs/logs_all_success \
  --candidate p4_logs/logs_all_prefill_lower \
  --window-seconds 250 \
  --output "$P4_COMPARE/analysis.json"
```

主机验证：新增 17 项回归通过，连同原有裁剪工具测试共 **42 passed**，ruff 检查通过；
见 [测试记录](results/prefill-review-20261004/design-tests.xml)。未执行 NPU 或实际压测，
未修改冻结版本、旧日志、原始四仓及正在运行的服务，未推送远端。

下一轮按 [内网指南 §5.1](intranet-validation.md#prefill-comparison) 配对复测：固定有效负载、
顺序、输入/输出长度、采样参数、到达并发和缓存初始状态；分别采集冷段、热段和完整请求结果。
先补齐与 P3 后段可比的 P4 数据，不以是否出现 Waiting 作为继续运行/结束测试的条件。
如果相同缓存状态、相同负载的长窗口仍有 TTFT/吞吐差异，再依据请求关联记录定位
P 排队、模型执行、保存/传输和 D 调度，必要时在内网单独采集短段 NPU profile。
