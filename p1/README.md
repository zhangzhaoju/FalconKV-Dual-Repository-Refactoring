# P1 执行记录：合仓与统一构建

状态：已按用户最新授权启动，首批源码改造及本机检查已完成；**内网构建、安装与运行等价验收未完成，P1 尚未通过出口**。

本轮允许在其他开发人员验证基线、后续归档数据期间推进 P1 源码工作。这是明确的阶段启动授权，不代表 P0/NPU 验收已经通过，也不授权替换现有服务。

## 1. 当前生效范围

以 [profile.json](profile.json) 为本轮机器可读配置：

- 仅 GLM-5.2 原生文本生成；GLM-5.3 后续扩展。
- DSA 双组开启（UNBUNDLE 与 TWO_GROUPS 均开启），MTP 开启；现有启动片段为单 token MTP，交接时记录最终有效值。
- C8 关闭。权重目录仍叫 GLM-5.2-w4a8c8，不修改权重或量化元数据；目录标签、W4A8 权重量化、运行时 KV/index cache dtype 分别核实。
- 同时验收 TP8/DP2 和 TP4/DP4，均为 4 节点 × 8 张 910B3、物理 2P2D。前者每节点一个 TP8 实例，P/D 各 DP2；后者每节点两个 TP4 实例，P/D 各 DP4，不重复计算卡数。
- 保留离线、在线、CPU KV 卸载、跨实例缓存、P/D、checkpoint、RemoteFill 及恢复能力。
- 复用 CANN 8.5.1、torch 2.9.0+cpu、torch_npu 2.9.0.post1+gitee7ba04、triton-ascend 3.2.0.dev20260322，候选版本不等于 ABI/功能认证。

## 2. 已实施的源码变更

当前 P1 活动源码位于工作区根目录的 `p1-repos/`：两个仓各有独立 `.git`，使用 `main` 分支，`origin` 分别指向新的 `vllm-dual` / `LMCache-dual` 仓库。初始导入使用的 `p1-worktrees/` 已在核对提交、文件树和额外文件后移除，不再作为开发入口。原始四个 checkout、原重构分支和备份继续保留；没有切换原仓分支、修改其源码或远端、停止服务或执行安装。详见 [工作区迁移记录](workspace-migration.md)。

| P1 仓库 | 唯一 distribution | 暂时保留的 Python namespace |
| --- | --- | --- |
| [vllm](../../p1-repos/vllm/) | vllm 0.18.0+ascend.p1 | vllm、vllm_ascend |
| [LMCache](../../p1-repos/LMCache/) | lmcache 0.4.3+ascend.p1 | lmcache、lmcache_ascend |

版本号是 P1 内部交付标识，不表示升级到了另一个上游分支；源码仍来自下表固定提交。Ascend namespace 源码位于各仓 `ascend/` 下，通过唯一根构建入口映射到 wheel 顶层。不需要再安装两个 Ascend distribution。

1. 导入当前 Ascend 快照、测试、原生源码及许可证；逐文件路径映射见 [source-manifest.json](baseline/source-manifest.json)。插件旧 setup/pyproject 改名归档到 `ascend/upstream-build/`，不再是活动入口。
2. 每仓只有一个根 setup/pyproject/CMake 入口；保留 Ascend 算子、包资源、版本信息和 vLLM 的 Ascend entry points。P1 仍有临时插件发现和运行时 patch，原生化在 P2/P3 完成。
3. 构建依赖固定到候选 torch/NPU 版本；运行依赖统一为 `requirements/ascend.txt`，不再选择 CPU/CUDA/HIP 构建或下载上游预编译 wheel。旧 `requirements/common.txt` 等保留作后续裁剪依据，不能继续作为本轮安装入口。
4. LMCache 同时构建 Ascend c_ops/cache kernels、HIXL/hcomm 及 host native_storage_ops/fs/redis。可选 native Mooncake 保留显式构建开关；它与 RemoteFill 使用的 CANN Python Mooncake 不是一回事。
5. 原生构建仅接受明确的 910B3、Python 3.11/aarch64 和 CANN 8.5.1。自定义算子在新建的私有源码副本中生成，缺子模块直接失败，不修改 Git 全局配置、不在构建阶段自动下载。
6. 校验并复用内网已初始化的固定子模块，生成逐文件材料哈希；从普通源码包或 sdist 重建不依赖 .git。
7. 迁移 host 测试相对路径，保留原断言；建立新构建/交付约束测试及 wheel 内容检查。

尚未执行：生产 Python 算法重写、删除其他模型/设备源码、去除临时 namespace、安装后替换现有服务。多模态及其他模型的传递依赖暂时保留，须在 P4 拆解导入闭包后裁剪；当前不能宣称已是最终极简框架。

## 3. 输入与备份

| 原仓 | P1 输入 HEAD |
| --- | --- |
| vllm | `ded5ce2a5388e1abb799d9b099849f3bd4a72360` |
| vllm-ascend | `d22f0b7cffde1b6ddb87cb44368e46193e811cc9` |
| LMCache | `802e4167afa75c1601b9f9d8619672a686b416bd` |
| LMCache-Ascend | `d959c7a9640681e414dc9dbca644627af67a51e8` |

上述提交包含 P0 独立修复，与最初设计调研 HEAD 不同。四个 bundle 保存在 `baseline/*.bundle`（Git 忽略的大文件），已执行 bundle verify；其 commit/tree/SHA-256 及来源映射记录在清单中。`source-01` 归档制作时尚未创建 P1 提交；其后 P1 改动已提交，并于 2026-09-21 推送到下面两个新仓。不能将原四仓输入 HEAD 当作改造后的交付版本。

| 新仓（`main`） | 本批已发布的 P1 提交 |
| --- | --- |
| [vllm-dual](https://github.com/zhangzhaoju/vllm-dual) | `b2025e53890eb9b65db3cfacba8e0237bea9654d` |
| [LMCache-dual](https://github.com/zhangzhaoju/LMCache-dual) | `5b09009c5264cb61d204b7660ae400b4392db46a` |

两个新仓当前为公开仓，GitHub Actions 按用户选择保持开启；GitHub 工作流结果不替代内网验收。`design` 文档及内网报告没有随两个代码仓推送。

Git 快照未包含被 `.claude/` 忽略规则排除的 `vllm/ascend/.claude/README.md`（非运行时说明）。删除旧 worktree 前已将其逐字保留到 `p1-repos`，与原 donor 文件哈希一致；它仍未进入 Git。内网首次 Git clone 后须按 [内网步骤](intranet-next-steps.md) 从固定 donor 提交恢复，再做严格源码审计；不得忽略缺失项或修改历史来源清单来绕过检查。`source-01` 普通归档已包含该文件。

本机缺 CATLASS 和 kvcache-ops 载荷，只有固定 gitlink。内网旧仓已有相应提交，可使用 [materialize_submodules.py](tools/materialize_submodules.py) 复制已核实的源码；不得声称本机普通归档已经包含完整构建材料。

## 4. 检查结果及局限

下表及 `results/validation-summary.json` 记录初始 `source-01` 批次，保留当时的 worktree 路径和“尚未创建提交”状态，不能当作当前工作区定位。目录移除后的复核单列在 [迁移记录](workspace-migration.md)，不改写旧 host/交付报告。

| 检查 | 本轮结果 |
| --- | --- |
| 新构建/交付约束单元测试 | 17/17 通过；只测试元数据、计划与合成 ZIP，不调用 build backend |
| 生产源码 AST | 两仓共 2239 个 Python 文件解析通过 |
| 快照保持检查 | 无生产 Python 改写、无丢失 donor 文件；差异为构建及测试路径适配 |
| 迁移 host 子集 | 107/112 通过，5 个失败均因本机没有 torch；0 skip |
| 原仓对照 | 同一 LMCache-Ascend 子集也为 21 通过/5 个缺 torch 失败 |
| 已归档 P0 内网 host | 112/112 通过；这是原四仓结果，不冒充 P1 通过 |
| sdist/wheel/native/NPU | 本机未执行；全部待内网 |

证据：[源码审计](results/source-audit-01.json)、[约束测试](results/contracts-01.log)、[迁移后 host](results/host-local-01/summary.json)、[原仓对照](results/host-original-control-01/summary.json)。普通交付包解包后也通过 [源码审计](results/export-audit-01.json) 和 [17 项约束测试](results/export-contracts-01.log)；这不是 sdist/wheel 构建测试。

已生成 [source-01 交付目录](deliveries/source-01/)，其中两份普通源码归档约 36 MiB / 4.3 MiB，SHA256SUMS 核对通过；仍需内网补齐固定子模块。汇总见 [检查记录](results/validation-summary.json)。后续批次保留新目录，不覆盖本批或 P0 报告。

当前约束测试默认读取 `p1-repos/`；内网或临时解包目录使用 `P1_SOURCE_WORKSPACE` 覆盖。审计、普通源码导出和 host 检查均显式传入 `--workspace p1-repos`。`tools/prepare_worktrees.py` 仅保留作初始导入与来源清单的历史实现，不是日常恢复/准备命令。

## 5. 下一步与出口

执行顺序及可复制命令见 [内网接力说明](intranet-next-steps.md)。首轮可先做源码审计、材料校验、依赖预检和 112 项 host 复测；不需要停止现有 GLM 服务。实际编译、安装及 NPU 测试使用隔离环境和预约资源。

P1 出口仍须逐项取得：

- 两仓 wheel 和 sdist 重新展开后的构建、内容检查、扩展加载及 ABI 结果。
- 干净验证容器仅安装两个 distribution；无旧 editable、PYTHONPATH 或原始四仓导入污染。
- 其他开发人员归档两种并行布局的有效 DSA 双组/MTP/C8-off 配置与基线数据。
- 相同输入、权重、采样和拓扑下，P1 离线/在线/缓存/恢复及性能与对应基线对照通过。

未完成这些项目时，保留原四仓与服务，不推进依赖运行等价证明的大规模裁剪，不宣告 P1 完成。
