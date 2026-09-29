# P3-01 配置原生化检查（2026-09-28）

本批配对提交：vLLM `744eb71f1ac6b7ab405b91666a99ae5f01fc9b08`；LMCache `470fd4dc6c566bc7b878ba425bc181a155e945ec`。两仓当前为 `p3` 且工作树干净，均直接继承冻结 P2 HEAD；P2 commit/tree 和 P1/main 引用核对未变。未推送远端，未改原四仓或运行服务。

完整机器报告见 [final/verification.json](final/verification.json)，各命令和原始输出、JUnit 见同目录。`precommit/` 是首轮开发中结果，不代替 final 的源码身份。

| 检查 | 结果与边界 |
| --- | --- |
| P2→P3 祖先关系和保留引用 | 两仓通过 |
| vLLM P2 host 回归及 P3 配对检查 | 167 passed |
| LMCache 开发态/构建契约 | 22 passed，native 命令为 mock、产物为合成 fixture |
| 新增配置行为测试 | 16 passed，真实配置模块、不导入 torch/NPU/旧插件 |
| 既有 RemoteFill 配置测试 | 5 passed，已移除插件导入前提 |
| Host 合计 | **210 passed** |
| 原插件 23 字段静态对照 | type/default/env_converter AST 全部一致；8 既有字段 + 15 新合入字段，无重复键 |
| vLLM 原生源码 gate | 通过，未恢复旧插件依赖 |
| Python 3.11 语法 / diff whitespace | 通过；执行解释器为本机 Python 3.12.3，不代表内网 3.11/ABI 验证 |
| Ruff | 修改的 Python 文件 E4/E7/E9/F（忽略既有 E731）通过；新测试/工具默认规则通过，非全仓 lint |
| Sphinx 新页 | 最小项目、`-W` 零 warning 构建通过，并核对 HTML 标题、安装示例和待办说明 |
| Sphinx 整站 | 未通过：既有 `conf.py` 依赖 `sphinxawesome_theme` 在本机缺失；未安装额外依赖，未把单页检查称为整站通过 |

配置覆盖默认值、环境/文件/dict/JSON 转换、早期导入的类引用、环境更新后的校验、RemoteFill sender/receiver 规则、重复校验、输入字典隔离、direct-HBM 禁配、DSA/CPU 约束和冷进程无框架导入。它不等于完整 LMCache 导入顺序或实际设备链路验证；其余运行时补丁尚在。

## 复现

从工作区根目录运行，输出必须指向不存在的新目录：

```bash
python3 -B design/p3/tools/verify.py --output /tmp/falconkv-p3-host-rerun
```

脚本不 checkout、不安装包、不编译、不访问内网；Git 只读检查加源码/host 用例。它允许记录开发工作树状态，但最终 handoff 使用 final 报告中的 clean 提交。固定测试数对应该批；后续新增测试时应独立更新阶段清单和预期。

## 待内网及后续源码完成

- wheel/sdist 独立重建、strict editable 的真实编译/安装和动态库/torch ABI。
- 需要 torch 的 `test_cold_resume_native_metadata.py`、`test_glm52_topk_ownership.py`、`test_mc2_recovery.py::test_production_draft_expansion_can_exceed_target_capacity`；本机沿用 P2 排除项，内网不照抄排除。
- TorchAir、NPU 算子和图、HCCL、GLM-5.2 离线/在线、DSA 双组/MTP、CPU KV、2P2D、checkpoint/RemoteFill/恢复、性能/长稳。
- P3-02～05 的 Engine/Connector/扩展/传输原生化、消除 `lmcache_ascend` 和剩余补丁。

**本报告只通过 P3-01 的 source/host 门槛，不通过 P3 阶段出口。** P2/P3 的统一内网安排、完整剩余批次和当前安装命令见 [P3](../../README.md) 与 [指导](../../intranet-validation.md)。
