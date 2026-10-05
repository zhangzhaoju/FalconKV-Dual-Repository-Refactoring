# P5 联合资格验证准备

2026-10-05 按用户要求冻结 native-layout，并将 P5 与 P6 的内网验证合并。
P5 两仓分支为 `p5`；最终使用包含全部 P5 的 `p6` 配对执行一次完整验收。
本阶段交付发布准备工具与操作流程，不将其记作资格验证已完成。

## 冻结与候选

| 内容 | vLLM | LMCache |
| --- | --- | --- |
| 冻结标签 | `native-layout-frozen-20261005` | 同左 |
| 冻结 SHA | `18fde02dc54a432f81ce5350c3358429a40f4bd3` | `e8236d8e1c41cd10752204d8cb9ef3b6074f999e` |
| P5 SHA | `a09d7ecca2f90ee6df14534a30747e3bb9069992` | `62740030fbf03d2b280333e71ecb26f071ab6966` |
| P5 P6 候选包 | `0.18.0+ascend.p5p6rc1` | `0.4.3+ascend.p5p6rc1` |

[冻结记录](baseline/native-layout-frozen-20261005.json)包含 tag 对象、tree、保留 refs 和
native-layout 日志文件哈希。[P5 配对](baseline/pair.json)仅供追溯，安装使用
[最终 P6 配对](../p6/baseline/pair.json)。不移动 `refactor/native-layout`、P4 或更早分支。

`p4_logs/logs_all_native_layout_suc` 支持 TP8/DP2 主链正常运行，但没有完整逐容器 SHA
和安装证据。旧压测中断、零输出误计成功、duration 为请求耗时求和，不能用于完整性能
签核；此前比较仍有 P99 TTFT 与 D1 尾段等待待复测。冻结是源码冻结，不是消除这些待办。

## 已交付内容

- 两仓候选版本、开发态安装、运行检查和 CI 配对更新；保留全部模型/缓存/原生实现指纹。
- [兼容清单](compatibility.json)：固定环境、GLM-5.2、两种 TP/DP、DSA/MTP/C8-off、ABI
  及 cache namespace 策略。镜像 digest、驱动与模型身份待内网补证，不伪造来源。
- `tools/qualify.py`：源码配对核对、两仓 wheel/sdist 及解包重建编排、45 项待验收清单、
  证据覆盖/哈希/人工签核检查。不会启动服务、安装框架或自动把用例改为通过。
- `tools/compare_perf.py`：至少三轮完整负载、真实墙钟、冷/热分开、输入身份一致、
  median/p95/p99 和逐轮数据；拒绝 HTTP 错误、零输出、中断、NaN 和缺失 token 间隔。
- `tools/prepare_image.py` 与 `image/`：生成固定 base digest 和精确双 wheel 的镜像上下文，
  不自动拉取/构建/推送镜像；镜像安装不替换基础依赖，仅保留三条既有 pip 豁免。
- [P6 备份和切换准备](../p6/README.md)，以及[统一内网执行指导](intranet-validation.md)。

本轮不更改推理算法、编译优化、量化、并行/调度参数或缓存协议，也不扩大硬件/模型范围。
编译、安装、NPU、功能、性能、长稳和实际切换仍在内网完成。完整证据内网留存，经人工
审核脱敏后交接，不接入外部大模型或自动上传诊断数据。

## 验收口径

本机最终检查：vLLM 372 passed、1 deselected、300 subtests；LMCache 204 passed、38 subtests；
新交付工具 26 passed。Sphinx 无警告，原生/运行时指纹与冻结输入一致。
详情见[主机结果](results/host-20261005/summary.json)，不替代内网构建与模型结果。
上述完整主机报告绑定文档更新前的 P6 配对；后续仅更新两仓说明文档，
当前精确提交及补充检查见 [P6 文档更新记录](../p6/README.md)。不重写此前报告的提交身份。

保留 [原验收矩阵](../03-baseline-and-validation.md) 的 A01 至 A15、B01 至 B11、C01 至 C13。
另以 PERF-TP8、PERF-TP4、SOAK-TP8、SOAK-TP4、IMAGE、RESTORE 记录性能、长稳、镜像和
恢复出口，共 45 项。B/C 每项按原矩阵覆盖必保组合，不能用一个 TP8 smoke 替代。

`qualify.py review` 只核对完整性、文件哈希和负责人声明，不理解任意日志内容，也不产生
生产切换授权。性能报告还应包含内存、MTP、图 replay、缓存搬运与输出正确性等原矩阵证据。
通过建议门槛不等于正式门槛已确认，阈值必须在测试前固定，不能看结果后调宽。

源码/主机结果与内网状态分列记录；未测保持 pending。无需在 P5 与 P6 重复同一套完整
验收，但不同拓扑、冷/热场景及至少三轮重复不因合并阶段而省略。
