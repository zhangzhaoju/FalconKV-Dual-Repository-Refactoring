# P0 执行记录：GLM-5.2 / Ascend 双仓重构

最新阶段说明：用户已另行授权 P1，DSA 双组/MTP 开启、C8 关闭，覆盖 TP8/DP2 与 TP4/DP4；基线由其他开发人员验证后归档。最新原四仓内网 host 已为 112/112 通过，不替代 NPU 基线。下文保留 P0 首批时点记录，其中“不进入 P1”、原始 HEAD、未初始化子模块等不是最新状态；当前以 [P1](../p1/README.md) 为准。P0 原始清单和采集结果未被重写。

启动日期：2026-09-20。设计审核基线：`design@fc20041197f664ddf586258c8227b6bd5c2e330c`。用户授权启动 P0，随后明确当前只需支持 `GLM-5.2-w4a8c8`，GLM-5.3 后续扩展。本机工作已启动并形成第一批交付；**P0 尚未通过出口，不进入 P1**。

本阶段没有合仓、删除模型/设备实现、改写推理算法、安装依赖或执行构建；没有访问内网、调用外部大模型或运行 NPU。原始四仓的分支和 HEAD 保持不变，必要修复留在工作区，未创建提交。

## 1. 当前范围和部署

唯一当前模型为 GLM-5.2-w4a8c8，保留其现有 MTP、DSA、CPU KV 卸载、共享缓存、跨实例缓存、P/D 分离、checkpoint、RemoteFill 及恢复能力。其他完整模型对外入口退出当前范围；GLM-5.3 只记录为后续扩展。

源码预期主架构为 `GlmMoeDsaForCausalLM`、`model_type=glm_moe_dsa`，实际 checkpoint 配置仍待核对。该模型复用 `deepseek_v2.py`；现有 speculative 配置将 `glm_moe_dsa` 映射到内部 `DeepSeekMTPModel`。这些名字不代表继续支持独立 DeepSeek 服务，也不能按文件名直接删掉共享实现。

| 角色 | 节点 | 每节点卡数 | TP / DP 的当前解释 |
| --- | --- | ---: | --- |
| Prefill | P0、P1 | 8 × 910B3 | 每副本 TP8，两副本组成 P 侧 DP2 |
| Decode | D0、D1 | 8 × 910B3 | 每副本 TP8，两副本组成 D 侧 DP2 |

总计 4 节点、32 卡，2P2D。实际 DP 进程组和路由启动方式由内网配置核实，不额外乘一次 DP2。EP、PP、CP、通信库/通道、权重 revision/hash、量化提供方及 C8/MTP 层配置尚未冻结。

软件目标固定为 CANN 8.5.1、torch 2.9.0、torch_npu 2.9.0。内网 torch 2.9.0 已安装，直接复用；本机不构建。公网材料可通过 proxy 获取；验证环境不接入外部大模型/Agent，诊断材料不自动上传。

## 2. 已完成的基线保护

| 仓库 | 分支 | 原始 HEAD | 备份/恢复 |
| --- | --- | --- | --- |
| vllm | dsa-two-groups | `764dea93053f7bfd24fdb78b6ca6996350cb9636` | 当前分支及可达 tags 的 bundle，临时 bare 恢复、HEAD/tree 比对、fsck 通过 |
| vllm-ascend | sparse | `326dd77b004a7fc1f393e80e39780adadc4b14cd` | 同上 |
| LMCache | dev-qzy | `4353ac9bf0b47f48868e1099103c6417ae9a6b84` | 同上 |
| LMCache-Ascend | dev-qzy | `07ebf5c4f7bc622bfce9f3a06fb01c20682ca3aa` | 同上 |

采集时四仓均无跟踪文件改动、未跟踪文件或被忽略文件。没有切换其他功能分支。四个 bundle 约 226 MiB，保存于 `private/backups/`，已被本目录 `.gitignore` 排除；文件校验值、tree、可达 tags 和恢复记录见 [source-manifest.json](baseline/source-manifest.json)。备份不等同于构建产物，不包含未下载的子模块载荷。

两个子模块尚未初始化，必须按固定 commit 补齐，不能直接追随分支最新版本：

| 主仓/子模块 | 固定 commit | 当前状态 |
| --- | --- | --- |
| vllm-ascend / csrc/third_party/catlass | `716fd7baa7fb7f6cac0488bb628fd1dd0e875641` | 只有 gitlink；实际源码未备份 |
| LMCache-Ascend / third_party/kvcache-ops | `9f18d2339bc58a43429f7d5bdaef1628c820eff5` | 只有 gitlink；实际源码未备份 |

来源和交付要求见 [子模块材料清单](review/submodule-materials.json)。子模块未齐、工具链未核对前，不宣称构建链完整。

## 3. 静态清单与边界

本轮索引 7,296 个 Git 跟踪文件（另外 2 个 gitlink 单独记录）、解析 4,203 个 Python 文件，解析失败 0。得到 97 个 patch 文件/调用点文件、784 个修改候选、327 条 registry 字典记录和 2,762 条依赖/下载/API 检查线索。计数不等于已确认补丁数、独立模型数或已解决缺陷数。

| 产物 | 内容与使用限制 |
| --- | --- |
| [源码快照](baseline/source-manifest.json) | 每文件 Git 对象、SHA-256、mode、大小；对应修复前的原始基线 |
| [Python 依赖索引](baseline/python-dependencies.json) | imports、词法条件、类/直接方法、super 调用、动态 import；不导入框架 |
| [补丁候选清单](baseline/patch-inventory.json) | 源文件、符号/语句、行号、词法条件和调用点；目标归属、实际新行为、动态顺序仍须逐项复核 |
| [依赖与下载线索](baseline/dependency-findings.json) | packaging、CMake、Git、镜像及 torch 私有 API/版本判断线索 |
| [文件迁移初表](review/migration-map.json) | 全文件源路径、拟定目标职责、处理原因、验收编号；全部禁止直接批量执行 |
| [继承契约差集](review/class-contracts.json) | 两代 Runner、CacheEngine、adapter 的直接方法覆盖与 super 使用；不假装已经解析全部 mixin/MRO |
| [当前模型范围投影](review/model-scope-projection.json) | 按最新 GLM-5.2-only 需求划分主入口、内部 MTP 和范围外注册 |
| [当前支持矩阵](support-matrix.json) | 唯一当前权重、32 卡拓扑、11 类必保场景及逐项待验状态 |
| [基线问题与风险](findings.md) | 已修复、待内网确认、待后续语义核对分别列出 |

`baseline/model-registry.json` 保留采集开始时、用户进一步收窄范围之前的原始分类；其 `disposition` 不再作为当前白名单。当前范围以 `support-matrix.json` 和 `review/model-scope-projection.json` 为准。不要修改原始备份来抹去范围或修复演进。

主 NPU Runner 有 74 个父类直接方法未在子类同名覆盖，v2 Runner 为 28 个；Ascend CacheEngine 为 112 个，adapter 为 200 个。此为静态差集，可能另受 mixin 影响，不能解释为全部运行路径已确认。它说明 P1/P2/P3 不能用简单文件覆盖替代继承契约迁移。

## 4. 独立修复与本机检查

本轮仅改动 14 个框架文件：

- 4 个构建声明文件：vLLM 的 `pyproject.toml` / `requirements/build.txt`、LMCache 的 `pyproject.toml` 对齐 torch 2.9.0；LMCache-Ascend 的 `pyproject.toml` 固定 torch/torch-npu 2.9.0。vllm-ascend 已经匹配，无须重复修改。
- 10 个 standalone 测试文件：将不存在的 `LMCache-NPU` 同级目录引用改为当前 `LMCache`。不改变被测逻辑、断言或预期结果。

可交接差异、原始 commit、修复后文件校验值和本机测试摘要见 [change-manifest.json](changes/change-manifest.json)；四个对应 `.patch` 与它同目录。差异已通过 `git diff --check` 与反向应用检查。没有自动 commit/push，也没有用新结果覆盖原始源码基线。

| 本机 host 子集 | 修复前 | 修复后 |
| --- | --- | --- |
| vllm | 16 通过 | 16 通过 |
| vllm-ascend | 30 通过、1 个旧目录错误 | 31 通过 |
| LMCache | 39 通过 | 39 通过 |
| LMCache-Ascend | 26 个旧目录错误 | 21 通过、5 个间接依赖 torch 的检查未通过 |
| 合计 | 85 通过、27 失败 | 107 通过、5 失败 |

剩余 5 项均在 `test_checkpoint_initialization.py` 经 `runpy` 加载 `kv_layer_groups.py` 时因本机没有 torch 报 `ModuleNotFoundError`。未安装本机 torch、未放宽断言或将它们记为通过；须在已有 torch 2.9.0 的内网环境重跑。本机 host 子集使用 `--noconftest` 和禁用插件自动加载，核对的是 AST/mock 状态逻辑，不是完整框架导入或 NPU 验收。

P0 工具另有 17 个标准库单元测试全部通过，覆盖：快照不覆盖、词法条件、候选补丁标识、凭据不采集、版本判断、模型元数据哈希、GLM-5.2 范围、32 卡拓扑、构建声明一致性、继承父类定位和备份恢复记录。

## 5. 内网接力与 P0 出口

下一步请内网人员按 [内网执行手册](intranet-runbook.md) 采集实际环境和模型元数据，先补齐子模块和依赖冲突处理，再建立 GLM-5.2-w4a8c8 的运行/性能对照。原始日志保留内网，只交接允许外发、已审核脱敏的摘要；不要求也不建立 AI 到内网的直连。

- [x] 原始四仓 HEAD/tree、工作区状态、文件哈希和可达历史备份已记录，恢复校验通过。
- [x] 当前模型范围和 2P2D/TP8/DP2 拓扑已记录；实际权重配置未冒充已验证。
- [x] 全源码静态索引、文件迁移初表、继承差集和支持矩阵已生成。
- [x] 独立修复已记录；本机可运行检查与失败原因有证据。
- [ ] 子模块完整源码/工具链材料已获得并校验。
- [ ] 补丁有效行为、条件和顺序、模型动态依赖闭包、目标归属逐项核对完毕。
- [ ] 内网实际 Python/OS/驱动/固件、SDK、torch ABI、量化层和 C8 布局已冻结。
- [ ] 原始 requirements 的冲突已按实际目标环境处理，形成可用目标依赖清单。
- [ ] 剩余 host 检查、NPU 算子/导入/构建及 GLM-5.2 必保场景在内网通过。
- [ ] 2P2D/TP8/DP2 的输出、图 replay、MTP/DSA、KV/RemoteFill、故障和性能基线已获得。

以上未完成项闭环前，P0 保持“进行中”，不将其标记完成或直接启动 P1。
