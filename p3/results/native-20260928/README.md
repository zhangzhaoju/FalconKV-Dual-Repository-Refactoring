# P3 原生源码交付：本机验证结果

日期：2026-09-28。结论：**P3-01～05 源码交付完成，源码及 host 门禁通过；P3 内网构建、ABI、NPU 与业务验收尚未执行。**

最终依据为提交后的 [verification.json](final/verification.json) 和各项原始日志、JUnit XML；`first/`、`precommit/`、`precommit-final/` 仅保留实施过程记录，不作为最终配对身份。

## 配对代码

| 仓库 | p3 提交 | 包版本 |
| --- | --- | --- |
| vLLM | `4e2f7c0047b4bcb14888f1bfabe01d41f70293d4` | `0.18.0+ascend.p3` |
| LMCache | `a4e2131e890727edadb6bb61cde369900f72c3fd` | `0.4.3+ascend.p3` |

两仓工作区干净，均包含完整冻结 P2；P2 的 ref/tree、P1/main ref 与开始 P3 时一致。本次未推送、未修改原四仓或现有服务。精确 tree、分支和检查范围见 [交付清单](../../baseline/p3-native-20260928.json)。

## 实际执行

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| vLLM host 用例 | 176 通过 | [日志](final/vllm-host.log)、[JUnit](final/vllm-host.xml) |
| LMCache host 用例 | 115 通过 | [日志](final/lmcache-host.log)、[JUnit](final/lmcache-host.xml) |
| RemoteFill 协议用例 | 98 通过 | [日志](final/remote-fill-protocol.log)、[JUnit](final/remote-fill-protocol.xml) |
| host 合计 | **389 通过** | 不含任何 NPU 测试 |
| 23 个配置字段的类型/默认值/转换器 | 与 P2 原注入定义一致 | `verification.json` |
| 9 个合并类、694 个方法 | 归一化 AST 对比通过 | [方法比较](final/method-equivalence.log) |
| LMCache 原生源码门禁 | 扫描 422 个 Python 文件，通过 | [源码门禁](final/lmcache-native-source.log) |
| vLLM 原生源码门禁 | 通过 | [源码门禁](final/vllm-native-source.log) |
| 改动文件 Python 3.11 语法 / Git diff 检查 | 通过 | `verification.json` |

方法比较应用明确的 namespace、类名、显式 NPU API 和 `_common_*` 委托迁移规则；`NPUIPCWrapper.to_tensor` 显式指定 `npu:<index>` 单独记录。它证明这些方法未出现规则之外的改写，**不证明运行时、设备流、IPC、ABI 或性能等价**。缓存 key、wire constants、双组布局与原恢复协议未作优化性重写。

本机 Python 3.12.3；通过 AST 的 `feature_version=(3, 11)` 检查目标语法，并非实际在 Python 3.11 执行。使用临时 host venv `/tmp/falconkv-p3-host.yIFSSF`，复用已有 pytest 等工具，另外安装 msgspec 0.21.1、pyzmq 27.2.0 运行协议测试；isort 9.0.1 仅作导入格式化。未安装 torch/CANN 或编译框架。

Host 套件有意跳过需 torch/NPU 的历史用例；准确命令、过滤条件和预期数量均写入 JSON。历史测试 `ascend/tests/v1/test_remote_fill_decoder.py` 在 Python 3.12 语法扫描时有已有的正则 `\+` 转义警告，不是本批语法错误或运行验收结果。

额外检查：LMCache `lmcache/` 下 Ruff F 规则，以及本批新工具、两仓开发工具、正式 Connector 和新增测试的所选 F 规则通过；并未声称整个上游 pre-commit/mypy 通过。Markdown 内 bash 块均仅用 `bash -n` 核对语法，JSON 示例可解析，未执行构建/安装/推送命令。

## 文档验证

新增内容所在的 LMCache P3 RST 页面用 Sphinx 7.2.6、`-C -W -n` 单页构建通过，无警告；[生成 HTML](sphinx-page/index.html) 的标题、命令块和章节链接已检查。

全站构建也已尝试，但在读取既有 `docs/source/conf.py` 时缺少 `sphinxawesome_theme`，返回 2；**不能标为全站通过**。复现命令及结果见 [文档检查记录](docs-checks.md)。新页仍在原 getting_started toctree 中。

## 交接边界

以下全部待内网执行，不包含于上述 389 项：

- 普通 wheel / 独立 sdist 重建 / strict editable 的真实编译、安装与资源验证。
- 新进程导入顺序、spawn、`lmcache.c_ops`/TorchAir ABI、注册 host memory 与 NPU 拷贝。
- 真正跨进程 IPC、通信及内核数值正确性。
- GLM-5.2，DSA 双组/MTP 开启、C8 关闭；4×8 卡 2P2D 的 TP8/DP2 与 TP4/DP4。
- 离线/在线、CPU KV 冷热命中、跨实例/P-D、checkpoint、RemoteFill 取消/超时/失败恢复、成对重启与性能对照。

原 `lmcache_ascend` 和 GPU Connector 源码仅移动到 `ascend/legacy-p3/` 作为参考，未不可恢复地删除；Git 历史完整，归档不进入 wheel/sdist。非目标设备/模型与未启用历史源码的广泛裁剪仍属 P4，尚未启动。

下一步按 [内网操作指导](../../intranet-validation.md) 使用这两个配对提交。必须将旧动态 Connector 的类名/module_path 换成新入口，不能通过创建空旧包或启用全局 `transfer_to_npu` 绕过检查。P2/P3 环境、源码 checkout、缓存及报告分别保留。
