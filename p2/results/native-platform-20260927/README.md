# P2 原生平台首批检查（2026-09-27）

结论：61/61 主机契约测试通过，P1 保留引用、配对安装版本、修改 Python 文件的语法检查及方法迁移差异检查通过。结果只覆盖本次平台入口批次，不能作为完整框架导入、NPU/ABI、运行等价或 P2 出口证明。

## 可复现命令与结果

在 `falconkv` 根目录运行以下命令，只执行源码检查、Ruff 以及四组 host 测试。测试中的构建/安装操作为 mock 与临时文件 fixture；不会调用真实编译器、pip 安装、torch 或 NPU 运行时。

```bash
python3 -B design/p2/results/native-platform-20260927/verify.py
```

该脚本仅适用于本批迁移，后续批次应另建记录，不能放宽本批的差异断言。再次执行会更新本目录的检查日志及 JSON。

| 测试 | 数量 | 主要验证 |
| --- | ---: | --- |
| [vLLM 原生平台](vllm-test_p2_platform.log) | 15 | NPU 身份、缺 runtime 报错、无 metadata/可选插件过滤时激活、惰性构造、组件注册次序与失败传播、兼容类身份、通信/算子/MoE 分发 |
| [资源路径](vllm-test_p1_resources.log) | 2 | 普通 wheel 资源目录、strict editable symlink 保留以及重复调用 |
| [vLLM 开发安装工具](vllm-test_p1_development.log) | 22 | 版本配对、拒绝旧 P1 混装、已有构建/安装保护和产物映射 |
| [LMCache 开发安装工具](LMCache-test_p1_development.log) | 22 | 同一 P2 配对校验、LMCache 原有辅助工具约束 |

详细结果、命令、解释器版本、修改文件哈希、逐文件 Ruff 诊断与 P1 对照见 [verification.json](verification.json)。其中 `head_at_check` 记录测试时尚未提交的 HEAD，测试的是同时记录 SHA-256 的工作树；最终提交另见 [配对清单](../../baseline/p2-native-platform-20260927.json)。

## 源码与静态核对

- 两仓 `p1` commit/tree 与启动前清单相同，原 `main` 未前移。
- NPUPlatform 原有 28 个方法，27 个 AST 完全一致，`import_kernels` 调整资源锚点；仅新增 `register_builtin_components`，无删除方法。
- 两仓安装工具均接受 vLLM `0.18.0+ascend.p2` 与 LMCache `0.4.3+ascend.p1`，拒绝旧插件和旧 P1 vLLM 混入 P2 环境。
- 新增平台测试与资源测试的完整 Ruff 检查通过；平台/注册/分发迁移相关文件的 `E4,E7,E9,F,I` 检查通过，`git diff --check` 两仓通过。
- 完整 Ruff 扫描并非全部通过：vLLM 涉及文件仍有 151 条风格诊断（E501 136、SIM102 8、SIM117 5、SIM108 1、I001 1）。逐文件、逐规则计数不高于对应 P1 文件；新平台文件对照原 donor 方法，保留配置控制流与字符串以便确认搬迁未改逻辑。LMCache 本批完整 Ruff 通过。未宣称全量 pre-commit、mypy 或上游测试套件通过。

## 后续硬件检查

主机平台测试使用依赖 stub 和生产 AST，开发安装测试使用合成产物，因此不证明 import 闭包、实际进程启动、native 扩展加载、设备通信或模型行为。后续内网应核验：

1. 干净的 P2 配对安装，wheel 与 strict editable 的真实包路径、版本、扩展和 CANN 资源；无旧独立 Ascend distribution 残留。
2. 可选插件关闭/元数据缺失时 NPU 平台和必需组件仍可用；各 worker/engine 子进程注册正确，无 GPU/CPU 模型执行回退。
3. 目标模型无 LMCache 的离线/在线，再覆盖 DSA 双组、MTP、TP8/DP2 与 TP4/DP4 的联合缓存/恢复场景。
4. P1/基线已知故障及性能信号按 [P2 后置清单](../../README.md#5-已后置的-p1基线问题) 处理，不能以本批测试消除验收缺口。
