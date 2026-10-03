# P3 同步 NPU 循环导入修复：主机验证

日期：2026-09-29。P2 修复已通过带来源记录的 cherry-pick 同步到 P3。结果为 **397 项主机测试通过**；本机没有进行真实框架构建、安装、NPU/ABI 或 GLM-5.2 推理。

| 仓库 | 本次配对提交 |
| --- | --- |
| vLLM `p3` | `eb2a33d809b5768d778c4c6d05f6e64b5921fee3` |
| LMCache `p3`（本次不变） | `a4e2131e890727edadb6bb61cde369900f72c3fd` |

vLLM 修复来源为 P2 `fix/p2-npu-bootstrap` 的 `e6f962ea7777bf3c31906050f5610e8cb6f9e417`，六个修复文件在两个分支逐字相同。
两仓工作树干净；冻结 P2 ref/tree、P1/main ref 未变。没有推送远端或修改既有服务。

## 实际执行

依据为提交后生成的 [verification.json](final/verification.json)，含准确命令、源码身份、文件哈希、过滤范围与未执行项。

| 检查 | 结果 |
| --- | --- |
| [vLLM host](final/vllm-host.log) | 184 passed、1 deselected、48 subtests passed；含 8 项新增回归 |
| [LMCache host](final/lmcache-host.log) | 115 passed、38 subtests passed |
| [RemoteFill 协议](final/remote-fill-protocol.log) | 98 passed |
| [vLLM 源码门禁](final/vllm-native-source.log) / [LMCache 源码门禁](final/lmcache-native-source.log) | 通过 |
| [方法比较](final/method-equivalence.log) | 694 项归一化 AST 契约通过，仍非运行等价证明 |
| 23 个配置字段、Python 3.11 语法、Git diff | 通过，见 JSON |

复用先前 P3 host 环境 `/tmp/falconkv-p3-host.yIFSSF`（Python 3.12.3），本次没有安装依赖。命令：

```bash
/tmp/falconkv-p3-host.yIFSSF/bin/python -B design/p3/tools/verify.py \
  --output design/p3/results/npu-bootstrap-20260929/final
```

重跑使用新的输出目录；没有该临时解释器时使用已备齐 pytest、packaging、setuptools、yaml、msgspec、pyzmq 的 host 环境。
验证器的 vLLM 预期用例数由 176 更新为 184，不改变原过滤项。历史 2026-09-28 的 389 项报告不覆盖。
原有 `test_remote_fill_decoder.py` 的 Python 3.12 正则转义 SyntaxWarning 仍存在，不是新增运行验证结果。

完整平台源码的冷导入测试仍使用依赖 fixture；真实安装环境的 registry 子进程、GLM 类导入、构建/ABI、2P2D 及数值/性能全部待内网执行。参见 [P2 修复原因和证据](../../../p2/results/npu-bootstrap-20260929/README.md)、[最新交付清单](../../baseline/npu-bootstrap-fix-20260929.json) 及 [P3 指导](../../intranet-validation.md)。
