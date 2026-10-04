# P4 MTP 启动失败分析与修复（2026-10-04）

本文件保留首轮 MTP 修复的历史配对与验证结果。后续 RoPE/缓存控制流修复已更新两仓，
当前版本与结论见 [RoPE 报告](rope-startup-fix-20261004.md) 及
[执行指南 §3.2](intranet-validation.md#mtp-startup-retry)，不要再按下述历史 SHA 复测。

## 结论

本次失败是 P4 裁剪引入的 Runner 控制流回归，不是用户的 MTP 启动参数错误。
合法 `mtp` 在创建草稿 proposer 后，仍无条件执行 `raise ValueError`，四个 P/D 节点因此均无法初始化。
已在本地 vLLM `p4` 修复；没有关闭 MTP/DSA、恢复其他推测方法或调整并发参数。
主机检查通过，真实 NPU/2P2D 仍待内网复测，不能据此关闭 P4 验收。

## 日志证据

日志目录：`p4_logs/logs_all_mtp_failure`。下表为各节点首个该异常的位置，时间取日志原文。

| 角色 | 日志文件 | 首个异常行 / 时间 |
| --- | --- | --- |
| P DP1 | `7.150.5.207_P_dp1_port7910.log` | 206 / 10-03 21:55:06 |
| D DP0 | `7.150.1.218_D_dp0_port7920.log` | 191 / 10-03 21:55:20 |
| P DP0 | `7.150.7.133_P_dp0_port7910.log` | 206 / 10-03 21:55:30 |
| D DP1 | `7.150.3.14_D_dp1_port7920.log` | 190 / 10-03 21:55:41 |

四个文件第 7 行均传入 `method=deepseek_mtp, num_speculative_tokens=1`；
Engine 配置分别在第 38/40/39/39 行显示 `SpeculativeConfig(method='mtp', ..., num_spec_tokens=1)`。
这证明别名归一化已经完成，不能把 `mtp` 报错解释为方法名拼错。
配置同时显示 TP8/DP2、每实例 `max_num_seqs=16`、原生 `LMCacheConnectorV1`。
Prefill 为 eager、Decode 为 PIECEWISE 图模式，但二者在共同的 Runner 状态构造处失败。

调用链是 `NPUWorker.init_device` → `NPUModelRunner.__init__` →
`NPUModelRunnerState.__init__` → `ValueError: Unknown speculative decoding method: mtp`。
旧文件 `vllm/v1/worker/npu_runner_state.py:273` 的异常早于 Runner 的后续初始化与模型加载完成。
这是当前直接阻断点；日志不证明修复后模型加载、算子、缓存或性能全部正常。

## 根因与修改

P3 的多方法选择由 `if/elif/.../else` 组成。P4 移除退休方法时，
`design/p4/tools/specialize_source.py` 将 AST 中位于 `orelse` 的 `elif` 当成独立 `if` 处理，
使原本只针对不支持方法的 `else: raise` 被提到外层，成为有效 MTP 的必经路径。

修改范围如下：

- `vllm/v1/worker/npu_runner_state.py`：只对非规范化 `mtp` 拒绝；有效 MTP 能创建 proposer、rejection sampler 并完成剩余状态初始化，再进入 NPU 专用草稿 setup。
- `vllm/v1/worker/npu_model_runner.py`：恢复 `if not drafter / elif MTP` 的互斥关系。此处存在同源裁剪问题，会在无 drafter 时仍访问推测配置；它不是本轮日志的首个异常，但已补回归并修复。
- `tools/check_npu_bootstrap.py`：增加 MTP 配置、Runner 状态/执行器和 Ascend 草稿选择器的已安装源码身份核对，不再只依赖相同 `.p4` 版本号。
- `design/p4/tools/specialize_source.py`：保留 `elif` 与前置条件的关系，选中 fallback 时生成 `else`，删除无 fallback 的退休尾分支时不产生独立语句。
- `design/p4/tools/verify.py`：把裁剪工具的语义回归加入常规主机检查，合计由 8 项变为 9 项。

只检查并修复了本次 MTP 相关控制流，没有重新运行全仓批量裁剪。对 Runner 状态、Runner 执行器、
共享 Eagle、Ascend Eagle 与 speculative 配置五个文件做了裁剪前后方法级差异核对；
当前 P4 speculative 配置已单独重写，不含本次错误形态。这不是对所有运行路径的硬件验证。
独立 EAGLE/ngram/suffix/medusa/draft-model 等仍不支持，共享 Eagle 层只供 GLM 同 checkpoint MTP 使用。

## 交付配对与验证

| 仓库 | 分支 | 该轮精确提交 |
| --- | --- | --- |
| vLLM | `p4` | `275122e99093dafe9a7e7c759d6d30a7a5824fc4` |
| LMCache（本次未改动） | `p4` | `1fc7439b96d091378b527819dc92902115eee58d` |

包版本不变：`0.18.0+ascend.p4` / `0.4.3+ascend.p4`。
该轮配对曾写入 [配对清单](baseline/p4-pair.json)，清单现已随后续修复更新；
P3 冻结标签及 P1/P2/P3/main 均不变，原始日志未改动。
本次修复仅本地提交，未推送远端。

[主机报告](results/mtp-startup-20261004/verification.json) 的 9/9 项通过：

- vLLM：286 passed、300 subtests passed、1 deselected。
- LMCache：145 passed、38 subtests passed。
- 裁剪工具：14 passed。

新增 vLLM 22 项用例执行生产代码的完整状态构造函数与草稿 setup/proposal 方法，
覆盖 MTP 1/2 token、PP 末端/非末端、同步/异步、非 MTP 普通生成、拒绝退休方法、
prefill/decode × padded/unpadded proposal，以及 staged-SFA graph key 传递和源码身份检查。
张量分配、通信和模型对象使用主机替身；不验证设备 kernel、数值、显存或吞吐。
裁剪工具比较转换前后的返回值、异常与副作用，并检查重复转换保持一致。

新增测试文件通过 lint；修改的运行文件通过未定义名称检查及 `git diff --check`。
测试环境是本机 Python 3.12，未安装 torch/CANN，排除的 CPU tensor 用例仍须内网用
`verify.py --with-installed-torch` 补跑。此报告不表示 CANN/ABI/NPU 或完整功能验收通过。

## 内网下一步

按 [执行指南 §3.2](intranet-validation.md#mtp-startup-retry) 发布/同步精确 SHA，逐一核对四个容器。
这次只修改已有 Python 执行文件；已正确安装 P4 strict editable 且源码身份匹配时，
无需重编译 CANN 或重装 LMCache，重启自己这轮测试的新进程即可。
普通 wheel 需要重新构建/安装 vLLM wheel；尚未完成 P3→P4 升级时仍需两仓完整重建。

保持原 MTP/DSA 双组、TP8/DP2、并发预算和 C8 关闭设置，重新验证四实例初始化、请求完成、
MTP 草稿/接受计数、decode 并发及缓存恢复。归档精确配对、启动配置、四节点日志和同负载的性能结果。
