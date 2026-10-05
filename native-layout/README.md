# 原生目录整改交付

2026-10-05 更新：本分支已以 `native-layout-frozen-20261005` 冻结；两仓分别保持
`18fde02dc54a432f81ce5350c3358429a40f4bd3` 与 `e8236d8e1c41cd10752204d8cb9ef3b6074f999e`。
内网日志显示 TP8/DP2 主链可运行，但完整矩阵、精确安装来源及性能尾延迟仍待资格验证。
后续在包含完整 P5 的 P6 候选上统一验证，使用[P5 P6 指导](../p5/intranet-validation.md)，
不要将下文 layout1 历史安装命令误用到 P6。

本次落实已审核的根 `ascend/` 移除方案。两仓从 `p4-frozen-20261004` 派生到
`refactor/native-layout`，不推进 P4 分支或修改任何已有冻结标签。
本阶段是目录与构建整合，不是新一轮模型、算子或性能改动。

配对提交见 [pair.json](baseline/pair.json)，最终检查见
[主机验收记录](results/final-20261004/summary.json)。内网执行请使用
[P4 editable 逐容器安装指导](../p4/intranet-validation.md)；
新目录及 wheel/sdist 验证见 [安装与回归指令](intranet-validation.md)。
两仓源码已提交到本地新分支；目录整改交付时未自动推送 GitHub。

## 已实施内容

| 内容 | vLLM | LMCache |
| --- | --- | --- |
| 唯一产品 Python 包 | `vllm/` | `lmcache/` |
| 原生源码 | 根 `csrc/` | 根 `csrc/`，同时保留原主机存储组件 |
| 主扩展构建 | 根 CMake 引入 `cmake/npu_extensions.cmake` | 同左，保留 HIXL/HCOMM/HCCL 和主机 target |
| 固定子模块 | `csrc/third_party/catlass` | `third_party/kvcache-ops` |
| 工程目录 | 根 `tests/tools/examples/benchmarks/docs` | 根 `tests/tools/docs` |
| 新包版本 | `0.18.0+ascend.layout1` | `0.4.3+ascend.layout1` |

仓库根 `ascend/` 已取消，没有同名软链接兜底。包内按功能组织的 Ascend 后端、量化、
传输目录仍是正式模块；不重命名 ABI、环境变量、算子 vendor 或协议名称。
两仓运行时 **1110 个文件**及原生实现 **479 个文件**与冻结 P4 的内容指纹一致。
构建编排、包版本、安装脚本、测试定位与文档不在“内容不变”的断言范围内。

原 797 个 donor 跟踪条目均有去向，完整清单随各仓提交在
`docs/design/layout-migration.json`，包含原路径/blob、目标、动作、原因与文件模式。
子模块固定提交不变。重复 LICENSE 去重，文件版权头保留；旧 donor 工程配置退役，
由根工程规则接管。不恢复 P4 已裁掉的厂商、模型、多模态、训练或插件实现。

LMCache 同名测试以明确的 `_npu.py` 后缀共存，取消 `lmcache_tests` 别名和测试导入期
monkeypatch。原本已被 P4 排除的两份 multiprocess server 包装测试，以原字节保存在
`docs/design/retired-tests/*.py.txt` 的下级目录，不作为现役测试或新增跳过项。
历史测试中不适用于当前 profile 的其余用例未宣称已全部可运行。

## 验证边界

配对提交上的最终结果：vLLM **371 passed、1 deselected、300 subtests passed**；
LMCache **203 passed、38 subtests passed**。Sphinx 构建无警告，生成 HTML 的目录链接、
页面标题与命令代码块已检查。未执行的张量/NPU 用例不计入通过数。

主机入口为各仓 `python -B tools/run_layout_host_checks.py`；`--list` 显式列出测试范围
及延后到内网的张量用例。主机检查包括源码指纹、包/版本、材料路径、strict editable
资源映射、sdist 文件规则与不调用编译器的 CMake 解析。LMCache Sphinx 使用
`-E -W --keep-going` 构建，正式新指南已链接到目录。

本机没有安装 torch/CANN、没有构建 native/wheel、没有访问内网模型或停止服务。
主机测试使用 x86_64/Python 3.12 的现有工具环境，详见
[环境记录](results/final-20261004/host-environment.json)；它不是内网 Python 3.11/aarch64
及 setuptools 79.0.1 构建环境的替代验证。
源码/主机检查通过不等于 CANN 编译、ABI、NPU 或 2P2D 验收通过。特别是
`check_native_layout.py --source-only` 会允许并报告本机未初始化的固定材料；内网必须
登记真实材料后执行不带此参数的完整门槛。

调试过程的 `results/precommit-*` 是检查历史，不是交付结论：首次运行选错主机 Python，
后续新增的 `.git` 目录测试也曾捕获遗漏。对应问题已修复，以最终提交上的
`results/final-20261004` 为准，不删除失败证据。

## 保留与回退

原始四仓和先前 `design/p4` 未提交工作未被本轮改写。退役的跟踪文件都可从
`p4-frozen-20261004` 恢复。本机未跟踪的旧目录内容采用搬迁备份，未直接删除：

- vLLM：`build/layout-retired-sstv662u/ascend`，包含旧本地说明及缓存文件。
- LMCache：`build/layout-retired-hhpsapqh/ascend`，原目录仅有空子目录。

这些 `build/` 备份不进入 Git、sdist 或 wheel。不要把它们复制回新分支根目录。
回退时使用独立容器的冻结两仓配对版本，重建/安装原制品；不在运行中的 editable
服务下切换分支，不强制移动标签，也不单仓回退。

`tools/migrate_sources.py` 与 `tools/update_paths.py` 是本次一次性迁移的工程记录，
不是安装入口，不要在完成整改后的源码或内网容器再次执行。
