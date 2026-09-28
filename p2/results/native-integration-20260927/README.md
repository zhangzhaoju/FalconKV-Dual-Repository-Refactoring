# P2 原生集成本机检查

本报告仅证明源码/主机契约检查完成，不证明内网构建、真实框架导入、NPU/ABI、GLM 输出等价或性能验收通过。复现入口：

```bash
python3 design/p2/results/native-integration-20260927/verify.py
```

脚本不安装包、不编译目标扩展；刷新本目录生成的日志、JUnit XML 和 [verification.json](verification.json)。两仓准确提交、变更文件 SHA256、P1/main 保留核对和每条命令都写入 JSON。

| 检查 | 结果与范围 |
| --- | --- |
| vLLM 主机契约 | 167 passed，1 deselected，43 subtests passed；[日志](vllm-host.log) / [JUnit](vllm-host.xml) |
| LMCache 配对安装契约 | 22 passed；[日志](LMCache-host.log) / [JUnit](LMCache-host.xml) |
| Python 语法 | vLLM 2,063 个源码/活动测试文件、LMCache 467 个文件通过，共 2,530 个；同时通过 Python 3.11 grammar 检查；没有导入框架 |
| 原生源码门禁 | 356 个原生文件；检查旧包/patch 导入、GPU Runner/device import、CUDA API、OOT 注册、缺失本地模块；[结果](native-source-gate.log) |
| Python 包完整性 | 原生 `vllm` 包目录的 `__init__.py` 检查通过 |
| Ruff | vLLM 422 个生产/工具变更文件、LMCache 4 个文件，`E4,E7,E9,F,I`（保留既有 E731 忽略）通过；不是全规则/全仓 lint |
| Git 差异 | 两仓 `git diff --check p1` 通过，`p1`/`main` 原提交和 tree 不变 |
| 文档 | 新增 LMCache P2 RST 单页 docutils 解析通过；未宣告全站 Sphinx 构建通过 |

本轮新增/调整的关键契约覆盖：平台发现与 NPU 分发、不可变 layer factory、逐模型 fp32 router、MLA latent/indexer/C8 布局与拒绝不一致合并、metadata 消费顺序、异步通信只等待一次、读取真实当前 NPU stream、TorchAir 私有依赖并发隔离及嵌套字典传递、连接器惰性注册、拒绝混有旧 namespace/补丁档案的 wheel，以及 strict editable/native 资源发布。

本机环境为 Python 3.12，缺少 torch/CANN/NPU 和 pre-commit；没有执行完整 pre-commit、Ascend UT/e2e 或 wheel/sdist/editable 的真实构建安装。host 构建测试使用模拟命令和合成资源，不伪装为实际二进制制品。

本机明确未执行，内网必须补齐：

- `ascend/tests/standalone/test_cold_resume_native_metadata.py`。
- `ascend/tests/standalone/test_glm52_topk_ownership.py`。
- `test_mc2_recovery.py::test_production_draft_expansion_can_exceed_target_capacity`。
- 实际安装导入、spawn、扩展/TorchAir ABI、kernel/HCCL、GLM-5.2、DSA/MTP/P-D/恢复及性能长稳。

完整内网命令见 [统一验证指南](../../intranet-validation.md)。P1 已知共同故障与压测统计问题仍在后置队列，本机测试通过不改变其状态。
