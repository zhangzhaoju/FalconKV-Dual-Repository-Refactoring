# staged SFA 修复合入记录（2026-09-23）

状态：四个原始基线仓已合并修复分支，P1 两仓已完成对应移植并提交。**仅完成本地源码合入；未推送、未构建/安装、未替换服务，不代表 P1 运行验收通过。**

## 1. 固定来源与本地提交

本批拉取并固定四仓的 `origin/fix/staged-sfa-event-handoff`，共 7 个新增提交。原仓使用正常 merge，保留当前基线分支和 P0 依赖适配；P1 以本身当前 HEAD 为基础按目录映射移植，不合并回旧构建入口。

| 原仓 | 当前分支 | 修复分支 tip | 合入后本地 HEAD |
| --- | --- | --- | --- |
| vllm | dsa-two-groups | `b9b18239ebea064e61e2622c78da1c43746ea63e` | `09e8409be1274fa3b0a5e1d4a2d016a8f66115de` |
| vllm-ascend | sparse | `6fc30da651cd018f843753254114cc2321f47e89` | `2e48d7a61d9fab3d40babbacaecf95a76cfb0e06` |
| LMCache | dev-qzy | `22747f0256f32c4b377ea7d970d505681c0bccde` | `d24029eddf358bb139b66023ab21364ef4d438b4` |
| LMCache-Ascend | dev-qzy | `e18b939e8f8c2d20ba0a9ecd1692a6f3e5bb1dab` | `3452df44e66ce8a217c4badfe32d73783e66bd2d` |

vllm-ascend 的 merge 提交为 `c458c17ecfe2c618be7616b01fab5ab7a81180b2`，其后仅增加一个测试路径适配提交：把修复分支测试中的兄弟目录 `LMCache-NPU` 更正为本地实际的 `LMCache`，不改断言或生产逻辑。

| P1 仓 | 分支 | 本地新提交 | 合入前 HEAD |
| --- | --- | --- | --- |
| p1-repos/vllm | main | `3827b16205a3b748629e38ad739dfc55c6b04d49` | `89a40e2a3e53465907763cfd8fa692723b45b2b0` |
| p1-repos/LMCache | main | `b5a9db3577d28097f9051ea7b9f6ef68fb11b8eb` | `5b09009c5264cb61d204b7660ae400b4392db46a` |

六仓均创建了本地备份分支 `backup/pre-staged-sfa-event-handoff-20260923`，指向各自合入前 HEAD。没有修改远端配置、删除原仓或强制覆盖历史。四个修复分支 tip 均已验证是对应基线 HEAD 的祖先；完成时六仓工作区干净。

**这两个 P1 新提交尚未推送 GitHub，直接在内网执行 git pull 目前拿不到本批修复。** 需要另行授权发布，或通过批准的源码/提交传输渠道交接；不可把仍指向旧版本的远端 main 当作已修复。

## 2. 合入内容与保留项

本批涉及 25 个文件：10 个生产代码文件、15 个回归测试文件。范围不只是一处事件修改：

| 来源 | 主要修复 |
| --- | --- |
| vllm | 在全部调度模式初始化 bootstrap readiness，避免未初始化状态 |
| vllm-ascend | staged SFA 使用普通 NPU Event；生产者事件在图外、每次交接时 record，允许多个消费者等待，不在 capture 内 reset/record |
| vllm-ascend | GLM MoE router 的权重、计算与 logits 保持 FP32；forward 中不记录日志 |
| vllm-ascend | 保留冷恢复的 native metadata/proof，约束实际恢复 frontier、compact query 预算、dummy 长度与 KV 容量；抢占后清理旧 compact 状态 |
| LMCache | shared layer-page handle 使用单层偏移/尺寸；仅发布 frontier 一致的 prepared sparse source |
| LMCache-Ascend | 同步模式安全关闭；保留 flat store 有效 token 元数据；避免重复加入已复用的 chunk plan |

映射保持为 `vllm-ascend → p1-repos/vllm/ascend`、`LMCache-Ascend → p1-repos/LMCache/ascend`；另外两仓映射到对应 P1 根目录。

10 个生产文件已逐文件核对 SHA-256，P1 与合并后的原仓完全一致。P1 的测试仅适配跨仓路径；已有断言以及修复分支新增断言均保留。四处与原仓不同的测试路径及哈希见[本批修复清单](baseline/staged-sfa-event-handoff-20260923.json)。

保留既有 torch 2.9 候选依赖、两个 distribution 的根构建入口、ACLNN 安装目录修复、CMake 嵌套修复及固定子模块版本。未实施新的模型/设备裁剪，未改权重、量化元数据或服务配置。DSA 双组、MTP、C8-off，以及 TP8/DP2、TP4/DP4 的既定验收范围不变。

## 3. 来源审计

历史 `baseline/source-manifest.json`、bundle 和 source-01 报告保持不变。本批使用追加清单记录前后提交、文件映射和精确哈希，并绑定历史清单的 SHA-256。

`audit_sources.py` 新增可选 `--updates` 参数。指定本批清单时逐项验证全部 25 个文件（包括新增文件和测试）；未登记的 donor 生产改动、登记文件哈希不符、缺文件、错误历史清单、重复/越界路径仍报错。并非按整个生产目录放行。省略该参数仍沿用历史快照审计，不能用于确认本批修复。

本机复核示例（输出文件必须是新的）：

```bash
python3 -B design/p1/tools/audit_sources.py \
  --workspace p1-repos \
  --manifest design/p1/baseline/source-manifest.json \
  --updates design/p1/baseline/staged-sfa-event-handoff-20260923.json \
  --output /tmp/staged-sfa-source-audit.json
```

内网应一并同步本批 `design/p1/tools`、`tests` 和两个来源清单，按[更新后的内网流程](intranet-next-steps.md)使用固定新提交、导出新的构建副本，不覆盖旧 build 或报告。

## 4. 本机验证结果

本机为 Python 3.12，有 pytest/numpy，但无 torch/torch_npu；目标 Python 3.11、CANN/NPU 环境未在本机验证。所有下列测试均未执行原生构建或安装。

| 检查 | 结果与限制 |
| --- | --- |
| 四仓修复分支 ancestry / 六仓备份与 Git 状态 | 通过 |
| P1 25 项修复来源哈希 | 通过；10 个生产文件与原仓完全一致 |
| P1 生产 Python AST | 2240 个文件解析通过；不等于 import、ABI 或功能验证 |
| P1 构建/交付与来源审计约束 | 24/24 通过 |
| 已有 P1 CMake 入口解析测试 | 5/5 通过；没有配置、编译或设备探测 |
| 新增无 torch 轻量回归 | 原仓 55/55、P1 55/55 通过 |
| MC2 recovery 回归 | 原仓和 P1 均 14/15 通过；剩余 1 项因缺 torch 失败 |
| 既有 host 子集 | 原仓和 P1 均 107/112 通过；5 项均因缺 torch 失败，0 skip |
| 其他 torch/框架 UT、NPU 图回放、构建/ABI/在线离线验收 | 本机未执行，待内网 |

55 项轻量回归覆盖 scheduler bootstrap、sparse recovery budget、staged dummy capacity、同步 cache engine 关闭；**不包括 NPU Event 的真实流排序验证**。没有使用假 torch、删断言或跳过失败来宣称全部通过。

证据保存在 [本批结果目录](results/staged-sfa-20260923/)：

- [提交后来源审计](results/staged-sfa-20260923/source-audit-final.json)
- [P1 轻量测试 XML](results/staged-sfa-20260923/p1-handoff-light.xml)、[原仓对照 XML](results/staged-sfa-20260923/baseline-handoff-light.xml)
- [P1 host 汇总](results/staged-sfa-20260923/p1-host/summary.json)、[原仓 host 汇总](results/staged-sfa-20260923/baseline-host/summary.json)
- [约束测试日志](results/staged-sfa-20260923/contracts.log)、[CMake 测试日志](results/staged-sfa-20260923/cmake-entry.log)
- [验证汇总](results/staged-sfa-20260923/validation-summary.json)

## 5. 对新基线日志的确认

用户报告修复已验证，归档位置为 [base/base_success/基线正常日志](../../base/base_success/基线正常日志/)。本次核对未再检出此前 D 端 ExternalEvent 多 wait、producer event 等待失败及 WorkerProc 致命异常，D 端仍启用 staged SFA 图；这与本批图外交接修复的目标一致。

但同目录 [压测汇总](../../base/base_success/基线正常日志/ansible_test_tp8_dp2_20260923_085203.log) 第 615～616 行记录 `448 successful / 73 failed`，前部存在 ConnectionError。因此分别记录“用户已确认事件问题修复”和“该归档不是零失败的完整性能/功能验收”。本轮不据此否定合入，不擅自改拓扑或服务配置；失败请求的归属及完整基线数据由负责人继续归档。

启动版本字符串也不能单独证明运行目录的精确 Git HEAD。后续归档应附四仓/双仓 SHA、工作区状态、实际导入路径和配置哈希。

## 6. 内网剩余验证

先交接/发布本批固定提交，再按内网流程重跑来源审计、24 项约束、55 项轻量回归、112 项 host、依赖预检及新的 wheel/sdist 构建。原四仓和 P1 分开运行、分开归档，不混用安装环境。

在依赖齐备的验证容器内，补跑本批移植的全部 15 个测试文件，重点包括：

- `ascend/tests/ut/compilation/test_sfa_event_handoff.py`：严格的当前代事件 record/wait 模拟。
- `ascend/tests/ut/ops/test_gate_linear.py`、`ascend/tests/ut/attention/test_sfa_v1.py`：router FP32 和 attention metadata；使用 Ascend 测试子树入口，避免与根 vllm 的 `tests` 包同名冲突。
- cold-resume native metadata、GLM top-k ownership、完整 MC2 recovery。
- LMCache shared-layer handles / sparse source frontier，以及 Ascend flat-store tokens / sparse chunk plan / 同步关闭。

NPU 最小事件回归仅在预约、隔离的验证卡上执行；测试会进行 32 次图回放和两个消费者等待。以下命令接在内网流程已建立的 `P1_BUILD_WORKSPACE`、`P1_RUN` 环境中，不在现有服务容器执行：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q --tb=short --noconftest --import-mode=importlib \
  -c /dev/null -p no:cacheprovider \
  --junitxml="$P1_RUN/sfa-event-npu.xml" \
  "$P1_BUILD_WORKSPACE/vllm/ascend/tests/e2e/singlecard/test_sfa_event_handoff.py"
```

该用例含 NPU 不可用时的 skip 条件；必须看到实际 `1 passed, 0 skipped` 才算此项通过，退出码 0 但 skipped 不算。它只验证底层事件交接契约，不能替代 GLM 模型回归。

随后分别完成 TP8/DP2、TP4/DP4 的 GLM-5.2 在线/离线、DSA×MTP、长上下文、多消费者、CPU KV、RemoteFill/checkpoint、抢占/冷恢复及性能对照；有效配置继续关闭 C8。完成前不宣告 P1 出口通过，不替换现有服务。

## 7. 回溯与交接

所有本地合并/移植提交均附来源说明和 Signed-off-by。源码变更已提交；工作区外层不是 Git 仓，因此本说明、清单和检查工具需随 `design/p1` 单独交接，没有混入公开代码仓。

需要回溯时先查看各仓 `backup/pre-staged-sfa-event-handoff-20260923`。后续产生新改动后由负责人决定从备份创建新分支或以正常 revert 回退；本轮不执行 reset、clean 或覆盖源码。
