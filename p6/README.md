# P6 切换与恢复准备

2026-10-05 已从 P5 创建两仓 `p6`，完整继承 P5，最终配对见
[pair.json](baseline/pair.json)。P5/P6 只进行一轮联合内网资格验证，入口为
[统一执行指导](../p5/intranet-validation.md)。本页的“准备完成”不代表生产已经切换。

本轮补齐了两仓内部的基线一致安装指南，P6 当前文档配对为 vLLM
`0a08c6029fecc204fa785c26b2c92a765016b7db`、LMCache
`38f4b1bac48220b452bca9d1b4f66af6d3697dbf`。
包版本仍为 `.p5p6rc1`，安装/构建脚本、依赖和运行实现未改变。
继续使用原 strict editable 和 2P2D 启动命令进行对照；普通 wheel/镜像属于另行记录的发布证据。
内网请同时更新两仓提交与本页的配对清单，避免身份检查使用旧 SHA。
[此前配对](baseline/pair-before-install-docs-20261005.json)保留，旧主机报告仍绑定此前提交，
不覆盖历史证据。冻结标签、P5 和 native-layout 分支保持不变。
本轮补充检查为 78 项主机测试、16 个 subtests、两仓 8 组指令一致性和 Sphinx 无警告构建；
详情见[文档检查记录](results/install-docs-20261005/summary.json)及
[当前源码身份报告](results/install-docs-20261005/source-pair.json)，不是内网验证结果。

## 已实施与未实施

已准备源代码备份/恢复检查工具、候选两仓操作手册和切换/回退清单。
源码主机没有 NPU 构建、安装模型或操作内网服务。原四仓保留原路径和 HEAD，不删除，
不把备份数据打进 wheel/镜像。`refactor/native-layout` 及冻结标签均不前移。

本机四仓 HEAD bundle 放在工作区的 `p6-backups/source-heads-20261005/`，
不是产品仓的一部分。它们只含当前 HEAD 可达历史，不收集其他分支特性。
两个原 Ascend 仓的固定 CATLASS/kvcache-ops 未初始化，因此备份清单明确失败，不宣称
包含完整构建材料。需要在材料齐全的内网重做完整备份，并另存模型、环境、配置和制品。

本机已执行恢复检查，原 vLLM 与 LMCache 的源码 tree 校验通过；两 Ascend 仓因子模块载荷
缺失仍未通过完整检查。证据见[备份清单](results/source-heads-20261005/manifest.json)与
[恢复报告](results/source-heads-20261005/restore-report.json)。隔离恢复副本位于
`p6-backups/restore-heads-20261005/`，不是新的开发入口，不影响原仓。

## 可恢复备份工具

`tools/backup_sources.py` 提供 `create` 与 `verify`，没有删除或切换子命令。
拒绝脏/未跟踪源码、覆盖既有输出、将备份写入源仓、路径越界和 bundle 哈希不符。
递归记录已初始化的固定子模块；缺失或 pin 漂移阻断完整备份。
恢复验证在新目录 clone 本地 bundle、fsck、检出指定 HEAD 并核对 tree，完全不访问网络。

在内网使用实际原四仓绝对路径和一个新的备份目录：

```bash
P6_PY=/usr/local/python3.11.14/bin/python
P6_TOOL=/workspace/zzj/design/p6/tools/backup_sources.py
P6_BACKUP=/workspace/zzj/p6-source-backup-01
"$P6_PY" -B "$P6_TOOL" create \
  --repo vllm=/workspace/zzj/vllm \
  --repo vllm-ascend=/workspace/zzj/vllm-ascend \
  --repo LMCache=/workspace/zzj/LMCache \
  --repo LMCache-Ascend=/workspace/zzj/LMCache-Ascend \
  --output "$P6_BACKUP"
"$P6_PY" -B "$P6_TOOL" verify --archive "$P6_BACKUP" \
  --output /workspace/zzj/p6-restore-check-01
```

路径不存在时先按原始来源恢复，不用新两仓冒充旧四仓备份。此工具不会下载缺失材料。
保留失败输出，修复输入后选择新目录重试。未提交/ignored/LFS payload、模型权重、驱动、
环境、wheel、配置与业务数据不在 Git bundle 内，必须单独清点、校验并恢复验证。
`verify` 即使通过也明确 `production_recovery_verified=false`，不能据此删除原仓。

## 最终切换顺序

1. P5/P6 全矩阵、两种拓扑、性能/长稳及镜像检查通过，并有负责人签核。
2. 在独立容器恢复冻结 native-layout 配对和原基线，确认模型、配置、材料及旧 namespace
   可访问；离线与在线 smoke 成功，恢复动作不能影响当前测试服务。
3. 停止接入并排空/取消在途请求；确认异步 store、checkpoint 与已武装传输完成或按协议失败收敛。
4. 按同一镜像/双 wheel 配对部署，检查四容器身份、HCCL、cache transport 与健康状态。
   需要 paired restart 的错误不能只重启一侧。
5. 默认新旧持久化缓存 namespace 隔离；有确切的模型、量化、分组和 schema 兼容性证据后才复用。
6. 保留旧镜像、两仓制品、配置和 namespace，完成成对回退演练后，才把原四仓移到可恢复归档区。

最终活动源码为 `p1-repos/vllm` 与 `p1-repos/LMCache`。目录名沿用历史约定，无需再迁目录。
源代码备份、设计记录与日志属于归档，不是新增产品仓。活动工作区退出是验收后的独立动作，
本次没有执行 `mv`、删除、生产切换或远端推送。
