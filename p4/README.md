# P4：冻结 P3 后的最小推理框架整改

本阶段按用户确认，仅保留 **Ascend910B3 + GLM-5.2 原生文本生成**。
P4 是待内网验收的源码候选，不把主机检查结果等同于编译或模型运行通过。
本机不安装 torch/CANN、不构建 native 制品、不操作内网服务，也不直接连接内网大模型。

## 冻结与分支

| 仓库 | 冻结 P3 SHA | P4 分支 / 包版本 |
| --- | --- | --- |
| vllm-dual | `8767fe1121a085e8cefed03eeaa1444b55227521` | `p4` / `0.18.0+ascend.p4` |
| LMCache-dual | `a4e2131e890727edadb6bb61cde369900f72c3fd` | `p4` / `0.4.3+ascend.p4` |

两仓冻结标签均为 `p3-frozen-20261003`；P4 从这一配对版本创建。
`p1-repos` 只是现有目录名，未另建 worktree。原始四仓及 P0～P3 分支/日志不改动。
冻结记录及证据文件哈希见 [P3 输入](baseline/p3-before-p4-20261003.json)。
P4 精确配对见 [p4-pair.json](baseline/p4-pair.json)，
本轮提交、删除统计及测试结果见 [交付报告](handoff-report.md)。本次未推送远端。

## 整改内容

- 删除其他厂商后端、310P/A3/A5 专用实现、其他模型/蒸馏模型、多模态、训练、LoRA、pooling。
- 仅注册 GLM 原生架构及内部同 checkpoint MTP；禁止回退到 Transformers/custom 模型。
- 删除 LMCache SGLang/MindSpore/CacheBlend、旧 v0、独立 GPU multiprocess 服务、GPU 专用存储/传输与量化序列化实现。
- 保留 DSA 双组、MTP、CPU KV 卸载/共享、跨实例缓存、原生 MultiConnector/Mooncake、2P2D、RemoteFill、checkpoint 与恢复。
- 原生 CANN 构建入口限定 `ascend910b3`，删除 LoRA 内核/绑定；去除共享路径的 CUDA 执行调用。
- 删除旧设备/模型的文档、示例、CI 和直接依赖；GitHub Actions 仍开启，但改为无需 torch/CANN 的源码与主机检查。
- `p1_dev.py` 文件名不变，版本/验证已更新至 P4；需重新编译并安装两仓，不支持只切分支复用旧扩展。

保留的 DeepSeek/Eagle 层是 GLM 共享实现，不代表支持其独立 checkpoint。
历史 DTO、字段、layout enum 中的 GPU/LoRA/MM 命名有少量兼容槽位，未恢复执行实现；启用被删除功能应明确报错。
CPU 主机算子和 KV 存储不在“删除其他设备推理”范围内。
共用 CANN/CATLASS/kvcache-ops 材料维持审核过的固定版本；通用定义、历史类型槽位不代表支持其他设备或模型。
LMCache 六个合入类的 663 项方法契约保留；另有两项 310P 专用方法按明确退休清单删除。

架构白名单不能鉴别共享同一架构的未来模型版本。GLM-5.3 尚未验收；仍需由内网审核
GLM-5.2 checkpoint 标识、配置、量化描述和权重索引哈希。关闭 C8 不等于修改 W4A8 权重。

## 验证与交接

使用 [内网执行指南](intranet-validation.md)。P4 不宣告阶段验收完成，尚需内网构建、
普通 wheel/sdist 重建、四节点配对安装及离线/在线/2P2D/缓存恢复回归。

本机可复跑：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -B design/p4/tools/verify.py --output /tmp/p4-host-new
```

Python 环境需已有 pytest、msgspec、pyzmq、pydantic、numpy、setuptools；不会自动安装任何依赖。
未安装 torch 时默认排除 CPU tensor 测试，报告会明确记录；内网使用 `--with-installed-torch` 补跑。
动态导出和 native 扩展不能由 AST 审计证明，源码检查报告单列这些未验证项。

删除清单位于 `baseline/*-prune-batch*.json`，最终实际删除以 P3→P4 Git diff 为准：
原生 MultiConnector 与 IdentityReasoningParser 在审计后恢复，不应按早期删除候选清单判断其缺失。
所有删除均可从冻结标签恢复；未删除用户模型权重、原始日志或现有服务。

## P3 证据边界

现有成功日志支持 TP8/DP2 2P2D、DSA staged 路径与 MTP，decode 最大 Running 分别为 12/14，
汇总 MTP 接收率约 86.29%。这不是全部矩阵完成：TP4/DP4、离线/no-LMCache、制品重建、
故障与长稳及运行时精确 SHA 仍需补齐。benchmark 179 行中有 3 条超长度 HTTP400/零输出；
duration 是延迟求和，不能据此推算并发系统吞吐，更不能当作 3000 请求完整压测。
P4 必须使用相同有效负载与实际墙钟时间进行配对比较。
