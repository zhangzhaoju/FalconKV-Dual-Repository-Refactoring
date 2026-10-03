# P2 NPU 循环导入修复：源码与主机验证

日期：2026-09-29。结论：已复现并修复 registry 冷启动依赖环，P2 vLLM 主机套件通过；真实安装、GLM 类冷导入、NPU/ABI、2P2D 待内网复测。

输入：vLLM 修复分支 `fix/p2-npu-bootstrap`，提交 `e6f962ea7777bf3c31906050f5610e8cb6f9e417`；LMCache 冻结 P2 `cfe8a1754db743d41c8bb63f8d02ad7c3051948c`。冻结 vLLM `p2`/`p1`/`main` 未前移。本次没有修改 LMCache 代码；最终两仓 checkout 已恢复至 P3，用于联合源码验证。

## 实际结果

| 检查 | 结果与证据 |
| --- | --- |
| 修复前冷导入 | 新测试的四种导入顺序失败；[保留的 registry-parent traceback](before-fix.log) 复现 `npu.py → config → compilation → current_platform` |
| 新增测试 | 8 项通过：4 种新进程导入顺序、常量/重导出契约、子进程失败传播、超时、工具帮助入口 |
| 配对 P2 vLLM host | **175 passed、1 deselected、43 subtests passed**；[日志](p2-paired-host.log)、[JUnit](p2-paired-host.xml) |
| 原生源码门禁 | `tools/check_npu_native.py` 通过，扫描范围 356 个迁移文件；不是所有运行时依赖验证 |
| Python 3.11 语法 | 5 个修复涉及的 Python 文件通过 AST/compile；实际测试解释器为 3.12.3 |
| 选定 lint | 5 个文件的 Ruff `E4,E7,E9,F,I` 通过，Git diff 检查通过；未声称全仓 pre-commit/mypy 通过 |
| 非导入语句变更检查 | 相对冻结 P2，`npu.py` 31 个、`utils/ascend.py` 79 个函数/方法的 AST 去除 import 后相同；不证明设备运行等价 |

初次完整套件使用了 P2 vLLM + 本地 P3 LMCache，出现一项旧 `ascend/lmcache_ascend/v1/cache_engine.py` 路径缺失（174 passed、1 failed，见 [初次 JUnit](p2-host.xml)）。这是跨阶段测试夹具配对错误。随后临时检出 LMCache 的冻结 P2、原样重跑同一套件得到上述 175 项通过，并恢复 LMCache P3；未删除或忽略该失败用例。

测试命令（在独立配对 P2 checkout 的 vLLM 仓执行；输出目录自行选新目录）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest --noconftest -p no:cacheprovider \
  ascend/tests/standalone tests/standalone \
  ascend/tests/ut/core/test_kv_connector_worker_metadata_patch.py \
  --ignore=ascend/tests/standalone/test_cold_resume_native_metadata.py \
  --ignore=ascend/tests/standalone/test_glm52_topk_ownership.py \
  -k 'not production_draft_expansion_can_exceed_target_capacity' -q
```

上述两个文件及一个 case 的排除沿用原 P2 主机范围，因为本机没有 torch；没有扩大豁免。
新增测试执行完整平台选择器和 `npu.py`，但 torch、基础接口及其余依赖链使用明确的 fixture，不冒充实际 GLM 模型导入。

新增内网工具 `tools/check_npu_bootstrap.py --inspect-glm` 会执行真实安装的四种冷导入、registry 子进程协议及未缓存的 GLM 类检查。本机只测其标准库子进程辅助行为和帮助入口；六项真实检查未在本机执行。

同一修复已 cherry-pick 到 P3；[P3 完整结果](../../../p3/results/npu-bootstrap-20260929/README.md) 为 397 项通过。P2/P3 修复六个文件逐字相同。
下一步见 [内网重装与复测](../../npu-bootstrap-fix.md)，新增模块要求重新生成 strict editable 链接树；阶段运行验收仍未通过。
