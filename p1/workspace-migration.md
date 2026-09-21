# P1 工作区迁移与旧 worktree 清理记录

日期：2026-09-21。结论：`p1-worktrees/` 已无继续保留的必要，完成核对后已移除。后续开发、静态测试和普通源码导出统一使用 `p1-repos/`。本次只清理重复的工作副本，不是 P6 原四仓退役，也不代表 P1 运行验收通过。

## 1. 当前活动仓库

| 本机路径（相对工作区根目录） | 唯一 `origin` | 本批 `main` 提交 |
| --- | --- | --- |
| `p1-repos/vllm` | `git@github.com:zhangzhaoju/vllm-dual.git` | `b2025e53890eb9b65db3cfacba8e0237bea9654d` |
| `p1-repos/LMCache` | `git@github.com:zhangzhaoju/LMCache-dual.git` | `5b09009c5264cb61d204b7660ae400b4392db46a` |

两个新仓均有自己的 `.git` 目录，无 `objects/info/alternates`，不借用原仓或已删除目录的对象库。远端 `main` 已通过只读查询核对为上述提交，本地 `main` 跟踪 `origin/main`。完整性检查发现的旧 `origin/HEAD` 符号引用已在本地校正为 `origin/main`；没有因此修改提交、工作分支或远端设置。

GitHub 仓库仍为公开仓，Actions 按用户选择保持开启。本次清理未创建新提交或执行推送，`design` 的文档/工具更新仍是本地改动。

## 2. 删除前的保护与核对

1. 两个旧 worktree 与对应独立仓的 HEAD、tree 完全相同，且都无未提交的受控文件或普通未跟踪文件。
2. vLLM tree 为 `0bb40602ea957128e1f6117302eddfa0f975ca57`；LMCache tree 为 `9b51f4435a0dc3e690674e5b3e8889bbd3278d70`。
3. 唯一额外内容是旧 vLLM worktree 中被忽略的 `ascend/.claude/README.md`，已逐字保留到 `p1-repos/vllm/ascend/.claude/README.md`。其 SHA-256 为 `91098ed23a7967cf5942b00d92e51b0a6b515f5f7ed416457afd0c7396b126b0`，与原 `vllm-ascend` 文件一致。
4. 保留该文件后，以递归内容比较核对两对源码目录，除 `.git` 元数据外无差异。
5. 两个旧 worktree 的子模块均未初始化，目录为空，没有待保护的 CATLASS/kvcache-ops 载荷；固定 gitlink 继续保留在独立仓的提交中。
6. 两个独立仓均通过 `git fsck --connectivity-only --no-dangling`，确认不是依赖旧 worktree 的残缺副本。

该说明文件仍受原忽略规则影响，不会自动进入 GitHub。内网 Git 拉取后的补齐命令和哈希检查已写入 [内网步骤第 2 节](intranet-next-steps.md#2-复用固定子模块并检查源码)。不要放宽来源审计，也不要用 `git clean -fdx` 清掉已补齐的交付材料。

## 3. 已移除与保留的内容

已通过 Git 的 `worktree remove` 移除以下两个已登记的 worktree，未使用 `--force`，随后用 `rmdir` 删除空父目录：

- `p1-worktrees/vllm`
- `p1-worktrees/LMCache`
- 空的 `p1-worktrees/`

原 `vllm` / `LMCache` 的 worktree 登记已正常清除，不留下失效路径。原四仓的源码、当前分支和远端未改变；两个原仓中的 `p1/ascend-unified` 分支仍指向上表 P1 提交。`baseline/*.bundle`、`deliveries/source-01/`、P0/P1 历史结果均保留。

需要恢复源码时，可从 `p1-repos`、两个新 GitHub 仓库或原仓保留的重构分支取回上述提交；被忽略的说明文件另从已保留副本或原 donor 固定提交恢复。无需为日常工作重建被移除的 worktree，也不要把原始四仓当成当前 P1 开发目录。

## 4. 更新后的入口与复核

从工作区根目录执行本机约束测试，默认读取 `p1-repos`，无需设置环境变量：

```bash
env -u P1_SOURCE_WORKSPACE python3 -B -m unittest discover \
  -s design/p1/tests -p test_p1_contracts.py -v
```

内网仍显式设置 `P1_SOURCE_WORKSPACE` 指向全新验证源码目录；审计、host 检查及普通源码导出继续用 `--workspace` 指定目标，不依赖本机绝对路径。`prepare_worktrees.py` 仅保留作初始导入历史实现，不再是活动入口。完整 Git 拉取、固定提交、材料准备和构建步骤见 [内网接力说明](intranet-next-steps.md)。

删除目录后的复核结果：

| 检查 | 结果 |
| --- | --- |
| 旧目录与登记 | 目录已不存在；两个原仓各只登记自己的原 checkout |
| 独立仓对象完整性 | 两仓 `git fsck --connectivity-only --no-dangling` 通过 |
| 默认路径约束测试 | 不设置 `P1_SOURCE_WORKSPACE`，17/17 通过 |
| 来源保持与生产 Python 语法审计 | 无缺失 donor 文件、无非预期适配，2239 个 Python 文件解析通过；见 [新审计报告](results/source-audit-repos-01.json) |
| 修改的 Python 工具与测试 | Ruff E4/E7/E9/F/I 及格式检查通过 |
| 内网依赖、子模块、构建与 NPU | 本次没有安装、编译或运行验证；固定子模块载荷仍待内网补齐 |

`results/validation-summary.json`、`results/host-local-01/` 等历史报告保留原始 worktree 路径和当时状态；它们是旧批次证据，不是仍在生效的路径配置。本次没有改写这些历史报告、来源清单或已交付归档。
