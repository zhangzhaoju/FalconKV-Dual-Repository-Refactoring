# P1 内网候选版本同步记录

日期：2026-09-23。依据用户提供的内网 preflight 元数据，将候选调整为 `torch-npu==2.9.0.post2`、`transformers==5.2.0`。torch 本体仍为 `2.9.0`（允许既有 `2.9.0+cpu`），CANN、triton 及其他依赖不变。本次不安装/升级/降级包、不编译、不推送、不修改原四仓或服务。

## 修改与交付

preflight 的版本要求来自所选源码目录，而非工具内部硬编码。因此两仓的 `requirements/ascend.txt`、`requirements/build.txt`、`pyproject.toml`、`p1_build.py` 及仓内说明同步调整；外层 profile、约束测试及操作指导同步刷新。preflight 报告新增 `workspace`，便于识别误用旧快照。没有增加忽略版本错误或无条件接受所有 5.x 的逻辑。

| 仓库 | 本地提交，尚未推送 |
| --- | --- |
| vllm | `230fbc0218e656cb90f8a8c2261150345a1d8ee5` |
| LMCache | `547ae7c10b0e510b864c8d0f5233d6f54f279359` |

两仓备份分支为 `backup/pre-intranet-candidates-20260923`。本批[累积清单](../../baseline/p1-intranet-candidates-20260923.json)记录 46 个文件的当前精确哈希，引用前一批清单的 SHA-256；历史来源与旧批次清单未改写。

## 本机验证

- [约束/预检 31/31](contracts.log)：新增 7 项回归验证实际候选通过，旧版/未批准新版拒绝，缺包不跳过，旧源码声明不会被工具覆盖，架构/CANN/工具门槛保留。版本信息与 CANN 文件是测试夹具，不是内网实测。
- [vllm 开发安装 21/21](dev-vllm.log)、[LMCache 开发安装 21/21](dev-lmcache.log)、[资源路径 2/2](resources.log)，合计 75 项通过，无 skip。native 命令仍为模拟。
- [源码审计通过](source-audit.json)：46 个文件哈希、2240 个运行时 Python AST；两仓工作区干净、构建 helper 一致。目标子模块载荷仍未在本机补齐。
- Ruff 检查通过；两份外层指南共 19 段 Bash 仅做语法检查，没有执行其中的构建/安装命令。
- [修改页独立 Sphinx 构建通过](sphinx-page.log)，已检查生成 HTML 中的候选版本和说明；[完整文档站未通过](sphinx-full.log)，原因为本机缺少 `sphinxawesome_theme`，没有为此安装依赖或更改全站配置。

未进行真实 native/PEP 660/wheel 安装及 ABI/NPU 验收，不能据此认定 Transformers 5 或 torch_npu post2 已与该分支运行兼容。

## 内网接续

同步两个提交及 `design/p1` 后，按[内网指导第 3.1 节](../../intranet-next-steps.md#31-本次候选版本不匹配的重试方法)重新预检。不能仅替换 preflight.py；也不能继续使用旧 `p1-check/run.*/source` 或带旧依赖元数据的 wheel。正式验收重新导出固定提交的新批次，保留旧报告。

后续仍须完成制品来源/哈希、`pip check`、torch/NPU ABI、tokenizer/config、GLM-5.2 离线/在线与 DSA/MTP 等既定回归；匹配直接依赖元数据不等于 P1 通过出口。
