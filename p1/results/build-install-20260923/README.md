# P1 构建与开发安装：本机实施记录

日期：2026-09-23。状态：入口、测试及指导已完成；**尚未进行内网 native 编译、pip 安装、ABI 或 NPU 验证，P1 未通过出口**。本批不修改原四仓、不停止服务、不安装依赖、不推送远端。

## 1. 固定交付版本

| 仓库 | 本地 main 提交 |
| --- | --- |
| vllm | `ba361feb8d13d2377698ec758cdf644fc00063be` |
| LMCache | `11f8ff086e75cf1fd4dcdd3d3807c4bca06d5fc3` |

两仓提交前均创建 `backup/pre-p1-build-install-20260923`，工作区干净。源码更新叠加在此前 staged-SFA 修复上，不替换原四仓基线。累计审计清单为 [p1-build-install-20260923.json](../../baseline/p1-build-install-20260923.json)：保留上一批 25 条记录，增加本批 16 条精确文件哈希；历史来源清单和上一批清单未改写。

## 2. 本批能力

- 两仓各自携带 `p1_dev.py`，提供 `materials`、`doctor`、`build`、`install`、`editable`、`verify`。不依赖外层 design 工具完成日常编译安装；正式验收仍需 design 检查。
- 固定材料从干净的本地子模块导入或注册，不在构建时下载；校验全部载荷哈希。
- wheel 与 editable 首次均完整编译 Ascend native；每次新建 `build/p1-native/run-*`，不复用旧设备对象或 ACLNN 源码副本，不手动删除失败现场。
- strict editable 使用保留的 native 产物和 Python 链接目录，覆盖两个 namespace、host 扩展、CANN 资源和生成的版本信息。仅修改已有 Python 文件时重启调测进程即可；新增文件、改动 native/依赖等需要重装。
- vLLM 唯一新增运行时适配位于 `ascend/vllm_ascend/platform.py`：custom-op 查找保留链接目录，不解析回原始源码。普通 wheel 的资源路径测试保持通过；未改动 DSA/MTP 算法。
- 安装需明确 `--isolated-env`，拒绝旧四包及冲突版本；该参数不自动创建容器，也不能检测所有在用进程。pip 不解析依赖、不访问 index，保留现有 torch 制品。
- 每次命令要求新输出目录，保留命令日志、真实退出码及产物哈希；`verify` 只核对安装位置和文件，不冒充 ELF/ABI/NPU 验证。

## 3. 实际本机检查

| 检查 | 结果及证据 |
| --- | --- |
| vllm 开发安装合约 | [21/21](dev-vllm-both-native-paths.log) |
| LMCache 开发安装合约 | [21/21](dev-lmcache-both-native-paths.log) |
| custom-op 资源路径 | [2/2](dev-resources.log) |
| 既有交付/来源审计合约 | [24/24](contracts.log) |
| CMake 入口语法/ABI 参数 | [5/5](cmake-entry.log)，无实际 native 编译 |
| SFA 轻量回归 | [55/55](sfa-light.log)，[JUnit](sfa-light.xml) |
| 源码审计 | [41 条精确哈希、2240 个运行时 Python AST 通过](source-audit.json) |
| 既有 host 子集 | [107/112；5 fail，0 error/skip](host/summary.json) |
| 代码样式 | [LMCache 配置的 Ruff 检查](lint-final.log) 通过；两仓新增 helper/test 内容一致，选定文件格式检查通过 |
| 命令预览 | [vllm editable](dry-run-vllm.log)、[LMCache build](dry-run-lmcache.log)：executed=false，没有执行 pip 或创建输出目录 |
| 指导脚本语法 | [22 段 Bash](guide-shell-syntax.log) 仅执行 bash -n，通过 |
| 新 LMCache 文档页 | [独立 Sphinx -W 检查通过](sphinx-page-final.log)，已检查生成 HTML 的命令/段落；第一次标题警告已修正，保留旧日志 |
| 完整 LMCache 文档站 | [未通过](sphinx-full.log)：缺 sphinxawesome_theme，没有为此安装依赖或改写原站配置 |

前六行合计 128 项通过，不含 host 子集。host 的 5 个失败全部来自 checkpoint initialization 导入 `torch`，本机没有该依赖；与上一批本机/原仓对照的缺依赖现象一致，但仍保留真实失败，不记为跳过或通过。内网门槛仍为 112/112，必须在既有 torch 2.9.0 环境复测。

新增合约测试模拟 CMake/ACLNN 命令并生成显式标记的假文件，两条 native 分支均检验了连续构建目录隔离。合成 wheel ZIP 测试没有调用真实 wheel backend。不能据此声称 CANN 链接或 PEP 660 完整安装已经通过。

## 4. 未验证项与交接

本机为 Python 3.12.3、setuptools 68.1.2、pytest 9.0.2；目标为 aarch64/Python 3.11、setuptools >=77.0.3,<81、CANN 8.5.1 及既定 torch/NPU 候选。本机仍缺两个固定子模块载荷、torch/torch_npu 和目标 CANN/NPU，未执行真实构建或安装。

旧 LMCache 日志确认设备对象为 ELF EXEC，新的构建目录隔离针对重复链接旧产物；是否解决目标环境全部问题仍需新的干净编译日志。不要用旧成功片段证明本版本 native 构建通过。

先发布或交接上述两个提交，并单独同步 design/p1。随后按[编译安装与调测指南](../../development-build-install.md)准备材料、doctor、安装和路径核验；正式 P1 出口按[内网主流程](../../intranet-next-steps.md)执行 wheel/sdist、加载/ABI、GLM-5.2 TP8/DP2 与 TP4/DP4、DSA 双组/MTP/C8-off、缓存/恢复和基线对照。不得在当前 GLM 服务容器里重装。

本批没有清理原模型/其他设备代码、改变候选依赖版本、关闭 Actions 或宣告运行等价。开发安装只是调测手段，不缩减 P1 验收范围。
