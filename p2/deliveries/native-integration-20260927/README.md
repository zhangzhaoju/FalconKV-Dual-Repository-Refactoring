# P2 源码交付包

两仓源码已提交，未推送远端，也未构建/安装 NPU 制品。P1/main 分支原样保留。

| 仓库 | 本批 P2 commit | 增量 bundle 所需 P1 commit |
| --- | --- | --- |
| vLLM | `f1be323571e3ca2aab53992234045dd064d1967f` | `230fbc0218e656cb90f8a8c2261150345a1d8ee5` |
| LMCache | `cfe8a1754db743d41c8bb63f8d02ad7c3051948c` | `547ae7c10b0e510b864c8d0f5233d6f54f279359` |

`vllm-p2.bundle` / `LMCache-p2.bundle` 是基于上表 P1 的增量 Git 包，包含 P2 两批提交。`*-p2-source.tar.gz` 是 `git archive` 导出的完整源码快照，便于审阅；它们不是 wheel/sdist，不含 .git、模型、已编译 native 产物或 gitlink 子模块材料。正式构建优先使用上述准确 SHA 的 Git 检出。

`p2-handoff.tar.gz` 包含 design/p2 的配置、映射、检查报告、命令和过程脚本，以及内网流程依赖的 P1 pip 检查工具/说明。不会递归包含本交付目录或 Python 缓存。先校验：

```bash
cd /path/to/native-integration-20260927
sha256sum -c SHA256SUMS
```

在已具备所需 P1 提交的内网仓库中先 `git bundle verify`，再导入到新分支/worktree；路径换成实际路径，新 worktree 不与运行中的 P1 共用。以 vLLM 为例：

```bash
git -C /path/to/existing/vllm bundle verify /path/to/vllm-p2.bundle
git -C /path/to/existing/vllm fetch /path/to/vllm-p2.bundle refs/heads/p2:refs/heads/p2-20260927
git -C /path/to/existing/vllm worktree add /new/p2-repos/vllm p2-20260927
git -C /new/p2-repos/vllm rev-parse HEAD
```

LMCache 对其自身仓库和 bundle 执行同样步骤，核对两仓 HEAD 与上表相符。缺少 P1 前置提交时增量包不能独立 clone，应先取得保留的 P1 仓库历史，不制造替代提交冒充原始身份。

解开交接文档后按 `design/p2/intranet-validation.md` 执行。LMCache 版本仍为 `0.4.3+ascend.p1`，但需要本批新提交；vLLM 为 `0.18.0+ascend.p2`。本机共 189 项主机测试通过；NPU/ABI、模型、缓存/P-D/恢复和性能验收仍待内网。P1/基线复测与已知问题修复按用户要求后置。
