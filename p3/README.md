# P3：LMCache 原生 Ascend 源码交付

2026-09-30：已同步 NPU DSA KV 绑定修复，保留普通/MTP 层的 latent → indexer 顺序与对象引用。最新 [配对清单](baseline/kv-cache-binding-fix-20260930.json)、[复测指南](../p2/npu-bootstrap-fix.md) 和 [409 项主机检查报告](results/kv-cache-binding-20260930/final/verification.json) 优先于旧交付；LMCache 提交不变。真实安装、NPU 与 2P2D 复测仍待内网执行。

2026-09-29：已同步 P2 registry 子进程的 NPU 平台循环导入修复。最新 [配对清单](baseline/npu-bootstrap-fix-20260929.json)、[修复说明与重装要求](../p2/npu-bootstrap-fix.md)、[397 项主机测试记录](results/npu-bootstrap-20260929/README.md) 优先于下方历史交付；LMCache 提交不变。真实 GLM 冷导入与 2P2D 复测待内网执行。

P3-01～05 的源码工作已完成，进入内网构建和 P2/P3 联合验证。两个产品仓继续位于 `p1-repos/`，当前均为 `p3`，包含完整 P2 历史；`p2`、`p1`、`main` 与原四仓保持不动。

**源码完成不代表阶段运行验收通过。** 本机没有安装 torch/CANN、构建框架或执行 NPU；真实 wheel/sdist/editable、ABI、GLM-5.2 推理及性能验收待内网执行。P1/基线已知问题仍按用户决定后置，P4 大规模裁剪未启动。

## 交付范围

| 批次 | 实施结果 |
| --- | --- |
| P3-01 | 单一配置类，23 个原插件字段；规范化和校验由 canonical config 负责 |
| P3-02 | `lmcache.c_ops`、通信扩展和内核库归入同一 namespace；host registration 保留；中性 `DeviceConnectorInterface` 与直接实现接口的 NPU Connector |
| P3-03 | CacheEngine、vLLM adapter、动态 wrapper 按方法合并；原先有效的父类委托保留为 `_common_*`；vLLM 正式 Connector 定义 DSA/checkpoint/事件交接/RemoteFill 生命周期 |
| P3-04 | storage factory、P2P/PD、IPC、传输、hash、lookup、NUMA/RPC 进入原生模块；保留禁配规则和传输协议 |
| P3-05 | 取消产品对 `lmcache_ascend`、`sys.modules` 替换及全局 `transfer_to_npu` 的依赖；同步 wheel/sdist/strict editable 布局和配对检查 |

方法清单：`p1-repos/LMCache/docs/p3-native-migration.json`。694 个方法契约逐项进行迁移规则归一化后的 AST 比较；IPC 重建显式指定 `npu:<index>` 为单独审核的设备语义变更。缓存 key、布局标识、wire constants、RemoteFill 协议状态机与已有恢复算法不作优化性重写。

原插件和 GPU Connector 源码移至 `LMCache/ascend/legacy-p3/`，排除于 wheel/sdist，Git 历史仍可恢复。不是保留一个可导入的空兼容壳。历史 `to_gpu/from_gpu`、`GPUKVFormat` 等协议/API 名称仍保留，不能仅凭名称判定为 CUDA 实现。

## 配对与验证

| 项目 | 保留 P2 输入 | P3 包 |
| --- | --- | --- |
| vLLM | `f1be323571e3ca2aab53992234045dd064d1967f` | `0.18.0+ascend.p3`，安装 namespace `vllm` |
| LMCache | `cfe8a1754db743d41c8bb63f8d02ad7c3051948c` | `0.4.3+ascend.p3`，安装 namespace `lmcache` |

- [最新配对交付清单](baseline/kv-cache-binding-fix-20260930.json)：累计含循环导入和 KV 绑定修复的精确提交和保留分支。
- [内网操作指导](intranet-validation.md)：同步、构建、editable、导入/spawn、启动参数迁移和验收。
- [最新源码与 host 结果](results/kv-cache-binding-20260930/final/verification.json)：409 项通过，含准确命令、实际范围及未执行项。
- [9 月 29 日导入修复结果](results/npu-bootstrap-20260929/README.md)：397 项的历史记录，不含本次绑定回归。
- [2026-09-28 原交付清单](baseline/p3-native-20260928.json) 和 [原检查结果](results/native-20260928/README.md)：历史输入，不覆盖为修复后的结果。
- [P3-01 历史清单](baseline/p3-01-config-20260928.json)：保留历史，不覆盖为本批结果。

重跑本机检查：

```bash
python3 -B design/p3/tools/verify.py --output design/p3/results/your-new-run
```

需要 pytest、setuptools、packaging、yaml、msgspec、pyzmq；不需要 torch/CANN。工具返回非零时不能称为通过。配置检查、源码门禁和 host 用例不能替代内网真实导入检查。

## 边界

当前目标仅 GLM-5.2 原生文本、DSA 双组/MTP 开启、C8 关闭；4 节点 × 8 卡、2P2D 的 TP8/DP2 与 TP4/DP4 两套配置。DeepSeek/GLM 共用的 MTP、DSA 组件按目标需要保留，不代表独立 DeepSeek checkpoint 已认证；GLM-5.3 后续扩展。

P4 仍负责清理非目标模型/设备和未启用的历史源码。原来即未实现的 Ascend 独立 multiprocess GPU cache server、非目标 CacheBlend 模型不新增支持，也不作为本批验收入口；这不等于删除目标链路中的 IPC DTO、CPU KV、跨实例缓存或 P/D 能力。P3 的 IPC wrapper 已原生归属，真实跨进程共享仍需内网验证。

不同阶段、不同同阶段提交不能仅按版本后缀混装；使用独立容器、源码 checkout、服务端口和缓存 namespace。不能切换正在被 editable 服务引用的源码分支。两仓本批提交仅在本地，不自动推送或替换任何现有服务。
