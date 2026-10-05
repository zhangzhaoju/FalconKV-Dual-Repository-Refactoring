# P4：冻结 P3 后的最小推理框架整改

本阶段按用户确认，仅保留 **Ascend910B3 + GLM-5.2 原生文本生成**。
P4 是待内网验收的源码候选，不把主机检查结果等同于编译或模型运行通过。
本机不安装 torch/CANN、不构建 native 制品、不操作内网服务，也不直接连接内网大模型。

当前安装入口：[P4 editable 安装与内网验证指南](intranet-validation.md)。
本次验证对象已是冻结 P4 派生的 `refactor/native-layout` / `+ascend.layout1`，
两仓均需重新编译安装；原 `+ascend.p4` 指令归档到 [冻结 P4 历史指南](intranet-validation-frozen-p4.md)。

## 冻结与分支

| 仓库 | 冻结 P3 SHA | P4 分支 / 包版本 |
| --- | --- | --- |
| vllm-dual | `8767fe1121a085e8cefed03eeaa1444b55227521` | `p4` / `0.18.0+ascend.p4` |
| LMCache-dual | `a4e2131e890727edadb6bb61cde369900f72c3fd` | `p4` / `0.4.3+ascend.p4` |

两仓冻结标签均为 `p3-frozen-20261003`；P4 从这一配对版本创建。
`p1-repos` 只是现有目录名，未另建 worktree。原始四仓及 P0～P3 分支/日志不改动。
冻结记录及证据文件哈希见 [P3 输入](baseline/p3-before-p4-20261003.json)。
冻结前 P4 精确配对见 [p4-pair.json](baseline/p4-pair.json)，
首次交付的提交、删除统计及测试结果见 [交付报告](handoff-report.md)。
2026-10-03 追加 LMCache SoC 大小写修复 `1fc7439b`；
原因与历史重试命令见 [冻结 P4 指南 §3.1](intranet-validation-frozen-p4.md#lmcache-soc-retry)。
2026-10-04 首轮 MTP 修复为 vLLM `275122e99` / LMCache `1fc7439b`，
[MTP 报告](mtp-startup-fix-20261004.md) 保留为历史记录。
随后根据 RoPE 启动失败补齐同源控制流修复，当时为 vLLM `8a5c71069` / LMCache `241166df`；
见 [RoPE 历史报告](rope-startup-fix-20261004.md)。
随后针对 `aclnnAddRmsNormBias` 失败恢复正式启动入口的 CANN 自定义算子路径注册，
该次配对为 vLLM `137fd3bc1` / LMCache `241166df`；LMCache 当轮未修改。
见 [算子注册分析与修复报告](custom-opp-startup-fix-20261004.md) 和
[历史复测指南 §3.2](intranet-validation-frozen-p4.md#custom-opp-retry)。该次修复未自动推送远端。

2026-10-04 追加 [Prefill / Decode Waiting 复核](prefill-review-20261004.md)：
P4 两个 D 节点仍有 Waiting；同早期窗口的 P 侧完成量接近，现有日志不足以确认明显性能回归。
新增只读比较工具，排除混入的旧 benchmark 文件；本次未修改运行代码或推进配对版本。
按 [性能采集要求](intranet-validation.md#prefill-comparison) 补齐同冷热阶段的完整结果。

两仓源码已按用户要求以配对附注标签 `p4-frozen-20261004` 本地冻结；冻结操作未自动推送。
见 [冻结记录](baseline/p4-frozen-20261004.json) 与
[根 ascend 目录移除分析](ascend-layout-analysis-20261004.md)。随后已按审核完成根 `ascend/`
目录整改，独立分支为 `refactor/native-layout`，未推进 `p4` 或移动冻结标签。
新配对为 vLLM `18fde02dc54a432f81ce5350c3358429a40f4bd3`、
LMCache `e8236d8e1c41cd10752204d8cb9ef3b6074f999e`，包版本分别为
`0.18.0+ascend.layout1`、`0.4.3+ascend.layout1`。
见 [目录整改交付](../native-layout/README.md) 和 [新布局配对清单](../native-layout/baseline/pair.json)。
冻结不代表完整验收通过；新增运行证据及剩余压测统计问题记录在冻结清单中。

## 整改内容

- 删除其他厂商后端、310P/A3/A5 专用实现、其他模型/蒸馏模型、多模态、训练、LoRA、pooling。
- 仅注册 GLM 原生架构及内部同 checkpoint MTP；禁止回退到 Transformers/custom 模型。
- 删除 LMCache SGLang/MindSpore/CacheBlend、旧 v0、独立 GPU multiprocess 服务、GPU 专用存储/传输与量化序列化实现。
- 保留 DSA 双组、MTP、CPU KV 卸载/共享、跨实例缓存、原生 MultiConnector/Mooncake、2P2D、RemoteFill、checkpoint 与恢复。
- 原生 CANN 构建入口限定 `ascend910b3`，删除 LoRA 内核/绑定；去除共享路径的 CUDA 执行调用。
- 删除旧设备/模型的文档、示例、CI 和直接依赖；GitHub Actions 仍开启，但改为无需 torch/CANN 的源码与主机检查。
- `p1_dev.py` 文件名不变，随当前分支校验版本、目录和材料；P3→P4、P4→新布局均需重建并安装两仓。历史纯 Python 修复的免重装条件不适用于本次目录整改。

保留的 DeepSeek/Eagle 层是 GLM 共享实现，不代表支持其独立 checkpoint。
历史 DTO、字段、layout enum 中的 GPU/LoRA/MM 命名有少量兼容槽位，未恢复执行实现；启用被删除功能应明确报错。
CPU 主机算子和 KV 存储不在“删除其他设备推理”范围内。
共用 CANN/CATLASS/kvcache-ops 材料维持审核过的固定版本；通用定义、历史类型槽位不代表支持其他设备或模型。
LMCache 六个合入类的 663 项方法契约保留；另有两项 310P 专用方法按明确退休清单删除。

架构白名单不能鉴别共享同一架构的未来模型版本。GLM-5.3 尚未验收；仍需由内网审核
GLM-5.2 checkpoint 标识、配置、量化描述和权重索引哈希。关闭 C8 不等于修改 W4A8 权重。

## 验证与交接

使用 [内网执行指南](intranet-validation.md)。P4 不宣告阶段验收完成，尚需内网构建、
普通 wheel/sdist 重建、四节点配对安装及离线/在线/2P2D/缓存恢复回归。

当前新布局的源码/主机子集入口（两仓保持同级目录）：

```bash
python3 -B p1-repos/vllm/tools/run_layout_host_checks.py
python3 -B p1-repos/LMCache/tools/run_layout_host_checks.py
```

Python 环境需已有 pytest、msgspec、pyzmq、pydantic、numpy、setuptools；
`cmake` 需在 PATH 中，检查不调用 CANN 编译器，也不会自动安装依赖。
上述入口明确列出延后的 tensor 用例，不因环境有 torch 而自动补跑。
新布局主机结果见 [最终报告](../native-layout/results/final-20261004/summary.json)；
内网仍需完成真实材料、构建、安装路径、ABI/NPU 与模型验证。
旧 `design/p4/tools/verify.py`（含 `--with-installed-torch`）仅适用于冻结 P4 布局，
不能在 `refactor/native-layout` 分支直接运行。

以下是冻结 P4 的历史验证记录，不是本次文档更新重新运行的结果：
算子注册修复后的 [主机报告](results/custom-opp-20261004/verification.json) 9/9 项通过：
vLLM 354 passed / 300 subtests（1 deselected），LMCache 164 passed / 38 subtests，
裁剪工具 25 passed。当轮增加 20 项注册/资源路径/冷进程/探针回归；
执行真实平台和初始化模块，设备依赖使用主机替身，仍不代表内网编译和运行验收。
探针新增的真实 `AddRmsNormBias` 检查仅在显式 `--device-smoke` 时执行，本机未执行。
此前 [RoPE 报告](results/rope-startup-20261004/verification.json)、
[MTP 报告](results/mtp-startup-20261004/verification.json) 和
[SoC 报告](results/lmcache-soc-case-20261003/verification.json) 保留为历史记录。

删除清单位于 `baseline/*-prune-batch*.json`，最终实际删除以 P3→P4 Git diff 为准：
原生 MultiConnector 与 IdentityReasoningParser 在审计后恢复，不应按早期删除候选清单判断其缺失。
所有删除均可从冻结标签恢复；未删除用户模型权重、原始日志或现有服务。

## P3 证据边界

现有成功日志支持 TP8/DP2 2P2D、DSA staged 路径与 MTP，decode 最大 Running 分别为 12/14，
汇总 MTP 接收率约 86.29%。这不是全部矩阵完成：TP4/DP4、离线/no-LMCache、制品重建、
故障与长稳及运行时精确 SHA 仍需补齐。benchmark 179 行中有 3 条超长度 HTTP400/零输出；
duration 是延迟求和，不能据此推算并发系统吞吐，更不能当作 3000 请求完整压测。
P4 必须使用相同有效负载与实际墙钟时间进行配对比较。
