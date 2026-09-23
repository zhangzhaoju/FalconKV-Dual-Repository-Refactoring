# P1 pip 标准库误声明豁免验证

2026-09-23，用户明确批准忽略 `op-compile-tool 0.1.0` 对 getopt、inspect、multiprocessing 的三条 missing-distribution 报错。本批仅更新 design/p1；两仓代码、提交、候选版本及原服务不变，没有安装包、修改 SDK 元数据或执行构建。

[check_pip_dependencies.py](../../tools/check_pip_dependencies.py) 运行当前解释器的只读 pip check，保存原始字节输出、SHA-256 和真实退出码。仅完整匹配已批准三条诊断且标准库模块可发现时豁免；脚本退出 0 与 pip 原始退出 1 分别记录为 `passed_with_waivers` 和 `raw_returncode=1`，不混同。其他包/版本、缺包、冲突、未知输出、执行异常、空失败输出均不放行。

检查结果：

- [41 项约束测试全部通过](contracts.log)，其中新增 10 项覆盖精确豁免、子集/顺序、真实依赖不豁免、版本边界、异常退出、标准库缺失、原始日志、禁止覆盖及超时/启动失败。测试中的 pip 子进程均为模拟，没有将结果冒充内网 pip 检查成功。
- [Ruff 和格式检查通过](lint.log)。
- [两份指导共 21 段 Bash 语法检查通过](guide-syntax.log)，没有执行其中的构建或安装命令。

[内网指导第 3.2 节](../../intranet-next-steps.md#32-已批准的-pip-标准库误声明豁免) 提供独立复核命令，正式流程和开发流程均已接入安装前后检查。仅需同步本批 design/p1，不需因本次豁免重建已有候选 wheel。实际内网结果、其他依赖及 ABI/NPU 验收仍须分别归档。
