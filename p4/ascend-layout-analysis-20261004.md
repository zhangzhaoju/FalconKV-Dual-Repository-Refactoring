# P4 冻结与 Ascend 根目录移除分析

本文件供审核后续目录整理方案。本次已冻结两仓当前 P4 源码，并完成只读结构分析；
**尚未执行目录搬迁或删除，也未新建实施分支、编译、安装或推送远端**。

结论是可以取消两仓根目录的 `ascend/`，但应先把仍需使用的内容归入主仓，而不是直接删除。
P2/P3 已完成主要 Python 实现的原生整合；这次要处理的是遗留的构建、原生源码、测试、
部署工具和工程配置的第二套目录。

## 冻结记录

范围仅限 `p1-repos/vllm` 与 `p1-repos/LMCache`，不修改工作区原始四仓。
两个仓库均已创建本地附注标签 `p4-frozen-20261004`：

| 仓库 | 冻结分支 | 冻结提交 | 包版本 |
| --- | --- | --- | --- |
| vllm-dual | `p4` | `137fd3bc144875075ae73be2782ec8b5abf5a002` | `0.18.0+ascend.p4` |
| LMCache-dual | `p4` | `241166dfde710e9a2cbe24ba0bd24b094b4bbe41` | `0.4.3+ascend.p4` |

提交、tree、标签对象、固定子模块、已有分支/标签和本轮日志哈希见
[冻结清单](baseline/p4-frozen-20261004.json)。冻结前两仓工作区均干净。
`design` 中此前尚未提交的工具/文档保持原状，不属于这两个运行仓标签的内容。

此冻结是后续整理的可回退源码基点，不是“P4 全部验收通过”标签：当前日志支持
TP8/DP2 2P2D、DSA staged 和 MTP；仍缺逐容器精确 SHA 关联和其他验收矩阵。
压测被中断，194 条记录中 192 条正输出，两次超长 HTTP400 被误计成功，duration 为延迟求和。
后续不得把这些缺口用目录迁移测试替代。

标签为本地记录，未配置 GitHub 分支保护、未推送远端。后续实施建议从两仓此标签创建
同名新分支 `refactor/native-layout`，不再推进或重写 `p4`、冻结标签及更早阶段引用。
若需要发布冻结快照，可单独推送两仓标签；以下命令本轮未执行：

```bash
git -C p1-repos/vllm push origin refs/tags/p4-frozen-20261004
git -C p1-repos/LMCache push origin refs/tags/p4-frozen-20261004
```

## 当前结构的真实关系

当前打包的 Python 主包只有 `vllm` 和 `lmcache`。两仓 `p1_build.py` 的
`resource_namespace(primary)` 返回主包本身，`setup.py` 从仓库根目录调用统一构建入口。
已经不再把 `ascend/vllm_ascend` 或 `ascend/lmcache_ascend` 作为运行实现发布。

本机 `p1-repos/vllm/ascend/vllm_ascend` 只剩 60 个后代空目录，文件和符号链接均为 0；
LMCache 的对应旧包目录已不存在。这些空目录不是 Git 跟踪的第二套 Python 代码。
其他开发容器可能还存在旧制品或日志，不能依据本机情况直接清空其目录。

按冻结提交统计，`ascend/` 中仍有以下 Git 条目。子模块 gitlink 计为 1 项，不展开其文件：

| 内容 | vLLM | LMCache |
| --- | ---: | ---: |
| 全部跟踪条目 | 648 | 149 |
| `csrc/` | 424，含 CATLASS gitlink | 35 |
| `tests/` | 153 | 93 |
| `examples/` | 8 | 0 |
| `benchmarks/` | 4 | 0 |
| `tools/` | 21 | 1 |
| `docs/` | 3 | 10 |
| 单独的 `third_party/` | 0 | 1，kvcache-ops gitlink |
| 其他配置、CMake、说明与许可等 | 35 | 9 |

完整清单和直接上移时的冲突见
[结构清单](baseline/ascend-layout-inventory-20261004.json)。直接去掉每个路径的 `ascend/`
前缀会产生 vLLM **17 个同名冲突**（14 个内容不同）、LMCache **43 个同名冲突**
（39 个内容不同）。因此不能执行无差别覆盖式移动。

当前根 `CMakeLists.txt` 都执行 `add_subdirectory(ascend)`；`p1_build.py` 和
`p1_dev.py` 仍从 `ascend/` 读取原生源码和材料。这正是空的旧 Python 包目录可以消失，
但整个 `ascend/` 还不能直接删除的原因。

## 目录一致性的边界

建议将“与原始主仓一致”明确为：维持主仓原有的顶层职责，只有一套 Python 主包、
构建入口、测试、示例、文档和工程配置。不恢复已经裁剪的设备、模型、训练或多模态实现，
也不要求目录树与未经裁剪的上游逐文件相同。

本次仅取消**仓库根部的 `ascend/` 容器目录**。主包中按功能组织的目录，例如
`vllm/v1/attention/backends/ascend`、`vllm/model_executor/layers/quantization/ascend`、
`vllm/distributed/kv_transfer/ascend`，是正式原生模块，不是嵌套仓，建议本轮保留。
如还希望重命名这些内部模块，应另列一次运行接口迁移，不与根目录整理混做。

建议目标布局如下，仅展示关键路径，未列出的现有目录保持：

```text
p1-repos/vllm/
  vllm/                         正式 Python 主包，现有模块路径不变
  csrc/                         现 ascend/csrc，保留算子相对布局
    third_party/catlass/         固定材料版本不变
  cmake/                        主构建所需的 NPU 构建模块
  tests/                        合并 standalone、ut、e2e 测试
  examples/disaggregated_prefill_v1/
  benchmarks/ops/
  tools/
  docs/
  CMakeLists.txt                 唯一顶层构建入口
  setup.py / pyproject.toml / p1_build.py / p1_dev.py

p1-repos/LMCache/
  lmcache/                      正式 Python 主包，现有模块路径不变
  csrc/                         NPU 源码与现有主机存储源码按子组件合并
  third_party/kvcache-ops/       固定材料版本不变
  cmake/
  tests/                        统一 fixtures，消除重复包装测试
  tools/
  docs/source/                  正式文档归入现有 Sphinx 结构
  CMakeLists.txt                 保留主机存储和 NPU target
  setup.py / pyproject.toml / p1_build.py / p1_dev.py
```

`p1_dev.py` 名称本轮建议不改，继续提供 `materials / doctor / build / editable / install / verify`。
新布局应使用独立版本标识和配对清单，避免与冻结 P4 的同版本旧制品混淆；无需恢复插件包。

## 内容迁移与冲突处理

| 原位置 | 建议归属 | 处理原则 |
| --- | --- | --- |
| 两仓 `ascend/csrc/` | 根 `csrc/` | 当前按相对文件路径没有与根 `csrc/` 的同名冲突；保留源码内容、文件模式和相对 include 布局，更新构建入口 |
| vLLM `ascend/cmake/utils.cmake` | `cmake/utils.cmake` | 校正相对目录，不能误删 CANN 自带构建引用 |
| LMCache `ascend/cmake/ascend_soc.cmake` | `cmake/ascend_soc.cmake` | 保留 910B3 限定和大小写规范化修复 |
| 两仓 `ascend/CMakeLists.txt` | 根 CMake 与 `cmake/npu_extensions.cmake` 等子模块 | 合并 target 定义与作用域，不覆盖根文件、不再保留第二个顶层 project |
| LMCache `ascend/third_party/kvcache-ops` | `third_party/kvcache-ops` | 修改 `.gitmodules` 的 path，提交 SHA 不变 |
| 两仓 `ascend/submodule-materials.json` | 建议根 `submodule-materials.json` | 这是生成的材料清单，不是 Git 跟踪源码；同步 material/doctor/sdist 校验，保留每个 payload 的哈希 |
| vLLM `ascend/tests/` | 根 `tests/` | 合并包标记，修正依赖目录深度的 `parents[n]` 和源码路径；不改变测试断言 |
| LMCache `ascend/tests/` | 根 `tests/` | 对同名测试逐项合并功能；独立新增用例可使用明确的 NPU 后缀，避免覆盖或递归导入 |
| vLLM `ascend/examples/disaggregated_prefill_v1/` | `examples/disaggregated_prefill_v1/` | 八个文件成组迁移，保留本地模块依赖；同步内网 proxy 和启动脚本路径 |
| vLLM `ascend/benchmarks/ops/` | `benchmarks/ops/` | 保留现有算子比较入口，不在本轮改压测算法 |
| 两仓 `ascend/tools/` | 根 `tools/` | 合并 `check_repo.sh`、`check_spdx_header.py` 等同名工具，核对仍有效的脚本路径 |
| 两仓 `ascend/docs/` 与 donor README | 主仓 `docs/` 和 README | LMCache 正式说明按仓约定整合为 Sphinx 页面；历史审计归入设计记录，撤销第二套文档构建入口 |
| donor 点文件、AGENTS、贡献说明等 | 根对应配置或退役 | 只保留一套生效规则；合并仍适用内容，移除旧插件/旧设备指引，不复制成另一个嵌套仓 |
| `ascend/LICENSE` | 根 LICENSE 与必要的第三方说明 | 两仓该文件分别与自己的根 LICENSE 完全相同；可去重，同时更新打包 license-files，保留来源说明与文件版权头 |

### 构建链必须同时改动

1. `p1_build.py` 中 `native = ROOT / "ascend"`、材料校验路径和通用 addon 残留分支需要同步收敛。
   两仓各自保持可独立构建，不改成依赖外部 design 目录或另一仓的构建脚本。
2. vLLM `csrc/CMakeLists.txt` 是 CANN 自定义算子的独立构建内容，不能简单用它替代原根
   PyTorch 扩展入口。保留 `build_aclnn.sh` 的隔离源拷贝、空 staging、安装目录自动创建和
   发布前资源校验；移动目录时不改变算子清单。
3. LMCache 根 CMake 已构建 `native_storage_ops`、`lmcache_fs`、`lmcache_redis` 及可选
   `lmcache_mooncake`。必须与 `c_ops`、cache kernels、HIXL/HCOMM/HCCL 分支共同保留，
   避免合并后只剩 NPU 扩展或丢失主机存储能力。
4. `.gitmodules` 与材料加载器一起更新。CATLASS 固定
   `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`，kvcache-ops 固定
   `9f18d2339bc58a43429f7d5bdaef1628c820eff5`；本轮不升级材料、不自动联网拉取。
   子模块移动后须验证全新 clone 与已初始化 checkout，两者的 Git 控制路径不能靠手工复制猜测。
5. `MANIFEST.in` 当前 `graft ascend`，`pyproject.toml` 当前引用 `ascend/LICENSE`；都必须改。
   vLLM 新增根 `csrc/`、`cmake/`，LMCache 新增根 `cmake/`、`third_party/` 和材料清单的
   sdist 纳入规则，继续排除 build/output/.git/缓存文件。
6. 将 `p1_dev.py` 的报告输出保护从旧 `ROOT/ascend` 更新到新源码/材料目录，防止把日志写进
   `csrc` 或子模块导致材料污染；同步 `.gitignore` 的生成文件路径与 wheel 内容校验。

### 保持安装与运行契约

目录整理不需要重命名以下产物或运行接口：

- `vllm._ascend_C`、`libvllm_ascend_kernels.so`、`vllm/_cann_ops_custom/vendors/vllm-ascend/`。
- `lmcache.c_ops`、`libcache_kernels.so`、现有传输/主机存储扩展。
- `ASCEND_CUSTOM_OPP_PATH`、已审核的 `VLLM_ASCEND_*` 配置、`ascend` 量化/传输协议名称。
- 正式 `LMCacheConnectorV1`、DSA 双组、MTP、RemoteFill、event handoff 与缓存保存语义。

这些名称属于 ABI、资源定位或已有配置契约，不能通过全局替换 `ascend` 来“清理”。
前几轮修复过的 OPP 注册必须继续从实际安装包/strict editable 链接树寻找资源，不能
`resolve()` 回源码目录后误认为原生库缺失，也不能依赖已删除的旧目录兜底。

### 测试整合不能只移动文件

LMCache 旧测试树的 `bootstrap.py` 把根测试树注册为 `lmcache_tests`，许多 donor 测试再导入
对应根测试；`conftest.py` 还在导入时初始化 NPU 并修改 fixture 工具。
例如两处 `tests/v1/test_cache_engine.py` 不同：donor 文件包装/导入根测试函数。
直接覆盖后可能形成自引用，并丢失原测试；直接将旧 conftest 合并到根目录还可能让
仅主机测试开始依赖 NPU。

建议合并为直接测试原生 `lmcache` 的用例和显式 NPU fixtures，删除仅为两棵测试树服务的
别名/包装层。对现有主机、CPU tensor 与 NPU 测试分别列出迁移前后 nodeid 映射和用例数量，
不能以批量跳过/删除不同版本的测试来消除冲突。

GitHub Actions 的两仓 `p4-host.yml` 与 `design/p4/tools/verify.py` 当前还引用 `ascend/tests`。
新布局实施时需要新布局对应的验证入口和 CI 路径，继续保留设备测试只在内网执行的边界。
P0～P4 的历史清单、报告和一次性裁剪脚本保留为冻结证据，不做全局路径替换；尤其不能
再次运行旧裁剪工具，将重新归位的 `csrc/`、`cmake/` 当作上游遗留目录删除。

## 建议实施顺序

1. **新分支与映射**：审批后从配对标签创建新分支，首先更新分支说明；给 797 个现有条目
   逐项标记目标路径、合并对象或退役理由，包含 gitlink、可执行位和来源记录。未跟踪内容单独盘点。
2. **原生构建归位**：把 csrc/cmake/子模块迁移与 CMake、材料加载器、manifest、许可路径
   作为同一完整批次提交，不留下“文件搬了但构建入口仍指向旧目录”的中间交付。
3. **测试与工具归位**：合并 fixtures 和重名测试，成组迁移部署示例、benchmarks、工具，
   同步实际启动命令、CI、验证工具和文档。普通包与 editable 不依赖仓库当前工作目录。
4. **取消遗留容器**：确认每项已迁移或获批退役后，移除根 `ascend/` 的跟踪内容。
   未跟踪日志、生成材料和旧 build 先登记并保留到明确的备份位置；空目录仅在验证确实为空后清理。
   不用全仓 `git clean`，不删除他人的日志、模型或运行中的服务文件。
5. **验证与配对交付**：主机检查通过后再在内网两仓全新构建、逐容器重新安装及回归。
   即便旧 editable 仍能加载之前的 `.so`，也不算新布局构建通过；此次迁移验收不复用旧链接树证明成功。

## 验收与回退门槛

| 范围 | 必须满足 |
| --- | --- |
| 源码布局 | 新提交不再包含根 `ascend/` 条目；没有回填该目录的软链接、导入 shim 或双源码路径 |
| 引用审计 | 当前构建、安装、CI、测试和正式运行脚本不再依赖根 `ascend/`；内部后端目录、CANN SDK 路径、ABI 名称和历史证据分类保留 |
| 行为边界 | 本轮不改变模型/算子算法、缓存调度、MTP/DSA 参数，不恢复非 910B3/非 GLM-5.2 功能 |
| 主机测试 | 原构建资源、SoC、MTP、RoPE、导出、保存尾块与 OPP 回归保持通过；迁移后的测试与 fixtures 无损，新增布局/材料路径检查 |
| 制品 | 全新 clone 与 sdist 解包均能独立构建；普通 wheel、strict editable 均只提供主包和完整原生资源；新制品不借用旧 checkout/build |
| 安装 | 四个测试容器逐一记录新配对 SHA、安装路径、资源与材料哈希；无旧 `.pth`/PYTHONPATH 掩盖错误 |
| 运行 | 离线、在线、无 LMCache、TP8/DP2 与 TP4/DP4、DSA/MTP、缓存保存/恢复及 P/D 回归；功能与冻结配对比较 |
| 性能 | 固定负载和冷热阶段，以有效请求及共同墙钟统计；纠正 HTTP400/零输出成功误计，不使用请求耗时求和计算系统吞吐 |

本机仅承担只读审计、静态/主机测试，不安装 torch/CANN、不编译 native、不直连内网模型。
原生构建、ABI 和运行验收仍在既定 910B3 / CANN 8.5.1 / torch 2.9.0 /
torch_npu 2.9.0.post2 环境中执行。

出现回归时，在独立测试环境用两仓 `p4-frozen-20261004` 配对源码、原材料和配置重建/重装；
不在运行中的 editable 服务下直接切换，不移动标签，不用旧 P3 或单仓混装代替配对回退。
源码标签不包含内网环境、模型、生成材料清单和已有 wheel，相关制品仍需另外归档。

## 本次交付状态

已创建并校验配对附注标签，记录原有引用、源码树、子模块和日志哈希；已完成全部跟踪条目
及同名冲突盘点，明确构建、测试、部署和资源契约的迁移范围。
本轮没有搬迁或删除任何源码、日志、材料或空目录，现有运行仓仍保持干净。
方案按现有设计文档格式整理，等待审核后进入新分支实施。
