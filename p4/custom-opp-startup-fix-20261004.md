# P4 AddRmsNormBias 启动失败分析与修复（2026-10-04）

## 结论

四节点已越过上一轮 RoPE/模型构造错误，加载权重后在预热阶段失败。
直接错误是 CANN 无法找到 `AddRmsNormBias` 的算子注册信息，
不是 RoPE/MTP 配置错误，也不能仅因 Decode 出现 `BackendCompilerFailed` 就归因于图编译器。

源码确认的回归是：P4 删除 GPU 导向的 `vllm._custom_ops` 时，丢失了其中
`current_platform.import_kernels()` 的启动副作用，正式入口没有注册本包的
`ASCEND_CUSTOM_OPP_PATH`。自定义算子和 910B kernel 构建项仍然保留。
已在 vLLM `p4` 提交 `137fd3bc144875075ae73be2782ec8b5abf5a002` 修复；LMCache 不变。
尚未推送，NPU 算子执行和 2P2D 回归待内网验证。

## 日志与源码证据

目录：`p4_logs/logs_all_aclnnAddRmsNormBias`。下表为各节点首个直接算子异常行，非后续汇总错误。

| 角色 | 日志文件 | 行 / 日志时间 |
| --- | --- | --- |
| P DP1 | `7.150.5.207_P_dp1_port7910.log` | 407 / 10-04 14:31:52 |
| P DP0 | `7.150.7.133_P_dp0_port7910.log` | 600 / 10-04 14:32:16 |
| D DP0 | `7.150.1.218_D_dp0_port7920.log` | 661 / 10-04 14:32:21 |
| D DP1 | `7.150.3.14_D_dp1_port7920.log` | 468 / 10-04 14:32:44 |

共同错误为：

```text
call aclnnAddRmsNormBias failed
AclNN_Parameter_Error(EZ1001): Get regInfo failed,
The binary_info_config.json of socVersion [ascend910b]
does not support opType [AddRmsNormBias].
```

Prefill 在 `determine_available_memory → profile_run → _dummy_run → model forward →
post_attention_layernorm → AscendRMSNorm.forward_npu` 中实际调用该算子失败。
Decode 在预热的编译流程中，经 `GraphFusionPassManager.configure →
AddRMSNormQuantFusionPass → AddRMSNormQuantPattern.register → make_fx` 进行真实张量 tracing，
同样触发该算子，然后由编译后端异常包装。Prefill eager 也失败，因此关闭 Decode 编译不能解决共同根因。
四文件在异常之前都有 `Loading model weights took`，但这不代表预热、MTP 或请求推理已通过。

冻结 P3 的 `vllm/_custom_ops.py:19` 调用了 `current_platform.import_kernels()`；
修复前 P4 的包内运行源码只保留了方法定义，没有正常启动调用。
唯一显式调用位于诊断工具 `tools/validate_npu_native.py`，所以该工具可能手动补齐路径后通过，
却不能证明服务 Worker 的正式入口正确。这是此前检查覆盖的缺口。

以下实现并未在 P4 中删除：

- `ascend/csrc/build_aclnn.sh` 的 `add_rms_norm_bias` 构建项和 `-c ascend910b`。
- `ascend/csrc/add_rms_norm_bias/op_host/add_rms_norm_bias_def.cpp` 的 `ascend910b` 注册。
- `ascend/csrc/torch_binding.cpp` 的 schema/实现及 `AscendRMSNorm.forward_npu` 调用。

本仓 910B3 构建使用上述 `ascend910b` kernel family；不应为消除错误把该名称改成 `ascend910b3`。
`_ascend_C` 和 ACLNN wrapper 可被加载，不等于 CANN 已能发现本包 vendor 的 kernel 元数据。
公开的 [Ascend op-plugin 路径解析实现](https://github.com/Ascend/op-plugin/blob/master/op_plugin/utils/op_api_common.cpp)
也读取 `ASCEND_CUSTOM_OPP_PATH`，将各 vendor 目录用于查找自定义 op_api；这仅佐证路径机制，
不替代本次内网 CANN 8.5.1 的运行证据。

现有日志没有运行时路径 dump 和精确 Git SHA，不能排除内网另有制品不完整或环境覆盖问题。
本次补齐路径、资源文件及真实算子三层检查，复测时分别确认，不把主机测试当作硬件修复已验收。

## 实施修改

1. `NPUPlatform.register_builtin_components()` 在其他原生组件之前注册本包 OPP；
   `initialize_native_ops()` 也先注册，覆盖直接 Worker 初始化路径。
2. 注册不导入 `_ascend_C`、不初始化设备；保持设备可见性设置和原来的惰性加载。
   本包路径置首、去重且可重复调用；保留其他 vendor 路径，不使用可能过期的全局布尔锁。
3. strict editable 以安装链接树定位资源，不把 `vllm.__file__` resolve 到源码目录；
   缺少本包 vendor 目录即明确失败，不静默借用其他安装。
4. `check_npu_bootstrap.py` 增加初始化源码身份核对，并经正式入口验证环境首项、
   `op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json` 的存在/JSON 可读性/哈希。
   文件路径来自本仓 `ascend/csrc/cmake/func.cmake` 安装规则，不证明全部算子条目或二进制完整。
5. `validate_npu_native.py` 删除手动补调，观察正式入口是否正确注册；显式 `--device-smoke`
   时新增四个真实 `AddRmsNormBias` 用例（float32/bfloat16、有/无 bias），与内置算子参考结果对照。

本轮仅修改 Python 代码、检查和测试，不改 native 源码/ABI，不恢复其他厂商、模型或已删除功能。
不关闭 MTP、DSA 双组、量化融合或 custom op，不要求升级/降级 CANN、torch。

## 配对与主机验证

| 仓库 | 分支 | 精确提交 |
| --- | --- | --- |
| vLLM | `p4` | `137fd3bc144875075ae73be2782ec8b5abf5a002` |
| LMCache | `p4` | `241166dfde710e9a2cbe24ba0bd24b094b4bbe41`（本轮未改） |

包版本仍为 `0.18.0+ascend.p4` / `0.4.3+ascend.p4`，按 [配对清单](baseline/p4-pair.json)
和安装源码身份核对，不能只按版本号判定。冻结 P3、P1/P2/P3/main、原四仓及用户日志未改变。

[主机验证报告](results/custom-opp-20261004/verification.json) 9/9 项通过：

- vLLM：354 passed、300 subtests passed、1 deselected。
- LMCache：164 passed、38 subtests passed。
- 裁剪工具：25 passed。

新增 20 项回归覆盖正式组件入口、直接 Worker 初始化、空/已有/变化的环境路径、
幂等性、缺失资源、editable 链接树、冷子进程和探针不得手工掩盖注册缺失。
最初 14 项注册回归在修复前全部失败；修复后及后增的 6 项检查全部通过。
新增测试执行完整生产平台/初始化模块，torch/设备/算子依赖使用替身，不执行 NPU kernel。
新增测试与两工具通过 lint，修改运行文件通过未定义名称检查，diff 检查通过。

本机 Python 3.12.3，无 torch/CANN 构建或模型运行；真实设备探针本机未执行。
内网仍需补跑 Python 3.11 `--with-installed-torch`、真实算子、ABI、模型及性能回归。

## 内网复测

按 [执行指南 §3.2](intranet-validation.md#custom-opp-retry) 发布 vLLM 修复、上传新 `design/p4`，
四个容器逐一更新、核对，并在空闲测试卡执行新探针，然后用原配置重新启动 2P2D。
不能只更新一个容器，不能在运行中的 editable 服务下切源码；必须启动新进程。

已完成 P4 strict editable 且源码身份、OPP 资源检查通过时，无需重新编译或重装；
若资源缺失或链接树陈旧，则按指南重建重装受影响仓库，不手工补路径借用旧制品。
普通 wheel 必须更新 vLLM；LMCache 已对应 `241166df` 时不用重复构建。
首次 P3→P4 升级仍需两仓完整原生重建。

保留 GLM-5.2、TP8/DP2、DSA 双组/MTP、C8 关闭及原并发预算，
核对四实例预热/KV 初始化、非空请求输出、decode 并发、MTP 计数和缓存恢复。
归档新 SHA/安装路径/元数据哈希、探针报告和四节点日志；P4 阶段验收尚未完成。
