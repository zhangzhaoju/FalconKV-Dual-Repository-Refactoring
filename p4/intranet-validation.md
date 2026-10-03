# P4 内网执行指南

## 1. 执行范围与前置条件

仅在独立测试容器构建/安装；不覆盖正在运行的 P3 基线。
内网可通过 proxy 获取审核过的材料，但框架构建/安装命令使用 `--no-index --no-deps --no-build-isolation`。
不要直接连接大模型服务给外部工具，也不要上传凭据、模型权重或含敏感内容的请求。

保留 Python 3.11.14/aarch64、CANN 8.5.1、torch 2.9.0+cpu、
torch-npu 2.9.0.post2、transformers 5.2.0、triton-ascend 3.2.0.dev20260322。
`torch` 包名中的 `+cpu` 不等于使用 CPU 推理；NPU 能力由 torch_npu 提供。
只豁免此前批准的 op-compile-tool 0.1.0 三条标准库误声明，其余 pip check 错误仍阻断。

上传本次 `design/p4` 及原有 `design/p1/tools/check_pip_dependencies.py`。
两仓仍放在 `/workspace/zzj/p1-repos/{vllm,LMCache}`。必须逐容器执行，不能因同镜像而省略其他节点安装。

若远端尚无 P4，由源码管理人员先发布两仓 `p4` 和 `p3-frozen-20261003` 标签。
本轮源码整改没有自动推送远端。不要再次执行 P1/P2/P3 的初始合仓/物化脚本。

源码管理人员确认 [配对清单](baseline/p4-pair.json) 后，在本机工作区发布（不在内网重造提交）：

```bash
git -C p1-repos/vllm push origin p4 refs/tags/p3-frozen-20261003
git -C p1-repos/LMCache push origin p4 refs/tags/p3-frozen-20261003
```

不使用 force，不覆盖同名但指向不同对象的标签；发生冲突应先核对远端。

## 2. 按配对 SHA 切换源码

以下在每个测试容器执行；不使用旧自动化脚本中的 P1/P2 SHA，也不以包版本单独判断安装正确。
`p4-pair.json` 是交付时生成的配对记录。

```bash
set -euo pipefail
P4_ROOT=/workspace/zzj
P4_PY=/usr/local/python3.11.14/bin/python
cd "$P4_ROOT"
P4_VLLM_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["vllm"]["commit"])')
P4_LMC_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["LMCache"]["commit"])')

git -C p1-repos/vllm diff --quiet
git -C p1-repos/vllm diff --cached --quiet
git -C p1-repos/LMCache diff --quiet
git -C p1-repos/LMCache diff --cached --quiet
git -C p1-repos/vllm fetch origin p4 refs/tags/p3-frozen-20261003:refs/tags/p3-frozen-20261003
git -C p1-repos/LMCache fetch origin p4 refs/tags/p3-frozen-20261003:refs/tags/p3-frozen-20261003
git -C p1-repos/vllm switch --detach "$P4_VLLM_REF"
git -C p1-repos/LMCache switch --detach "$P4_LMC_REF"
git -C p1-repos/vllm rev-parse HEAD
git -C p1-repos/LMCache rev-parse HEAD
```

有 tracked 修改时先交由源码所有者处理，不要 reset/clean；未跟踪的日志不自动删除。
若精确提交尚未发布，停止，不要用 `FETCH_HEAD` 或其他阶段代替。

## 3. 构建与 strict editable 重装

从 P3 升级 P4 时，因删除 native LoRA 绑定并改变了模块集合，**必须重新编译并重装两仓**。
`p1_dev.py` 仍是有效入口，名称未改；它已校验 P4 版本。
已经完成 vLLM P4 安装、仅遇到本次 LMCache SoC 检查失败的容器，可使用 §3.1 的单仓重试步骤。

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export SOC_VERSION=ascend910b3
export VLLM_TARGET_DEVICE=ascend
export USE_MINDSPORE=0
export BUILD_WITH_HIP=0
export VLLM_USE_PRECOMPILED=0
export COMPILE_CUSTOM_KERNELS=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=""
export VLLM_NO_USAGE_STATS=1

mkdir -p "$P4_ROOT/p1-repos/p4-check"
P4_RUN=$(mktemp -d "$P4_ROOT/p1-repos/p4-check/run.XXXXXXXX")
"$P4_PY" -B "$P4_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P4_RUN/pip-before"
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/p1_dev.py" doctor --output "$P4_RUN/vllm-doctor.json"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" doctor --output "$P4_RUN/lmcache-doctor.json"
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/p1_dev.py" editable --isolated-env --output "$P4_RUN/vllm-editable"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" editable --isolated-env --output "$P4_RUN/lmcache-editable"
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/p1_dev.py" verify --mode editable --output "$P4_RUN/vllm-installed.json"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" verify --mode editable --output "$P4_RUN/lmcache-installed.json"
"$P4_PY" -B "$P4_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P4_RUN/pip-after"
```

目录必须是新的；上述 `--isolated-env` 表示操作者确认这是专用测试环境，不是运行中的基线。
保留已有经审核的 CATLASS/kvcache-ops 材料及 `ascend/submodule-materials.json`。
若 doctor 报材料缺失，使用 `p1_dev.py materials --from-submodule <本机已审核子模块路径>`，
固定提交分别为 `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`、`9f18d2339bc58a43429f7d5bdaef1628c820eff5`；不得临时取最新版。
无需手工创建 `_cann_ops_custom`，脚本负责隔离原生构建目录。

<a id="lmcache-soc-retry"></a>

### 3.1 LMCache SoC 大小写错误的修复与重试（2026-10-03）

首次 P4 的 LMCache `1bedaf89` 在 CMake 配置阶段报
`P4 supports only SOC_VERSION=ascend910b3`，但日志中实际传参为 `-DSOC_VERSION=Ascend910B3`。
这是 P4 型号校验的大小写缺陷，不是硬件型号不支持：
`p1_build.py` 使用标准拼写读取 CANN 的 `Ascend910B3.ini`，并把它传入大小写敏感的 CMake 检查。
仅设置 `export SOC_VERSION=ascend910b3` 无法改变该构建参数。

已在 LMCache `p4` 提交 `1fc7439b96d091378b527819dc92902115eee58d` 修复：
校验时忽略大小写，下游统一使用 `Ascend910B3`；空值及 910B2/310P 等非目标型号仍阻断。
vLLM 保持 `0879e419001f424700569936962631ca0360f5af` 不变，包版本仍为 `.p4`，所以必须按 SHA 核对。
普通 wheel 与 editable 共用此构建路径，二者均需该修复；切换为普通安装不能绕过原缺陷。
pip root、setuptools_scm/vcs-versioning、manifest 及本次 kineto 警告均不是这个终止点。
不要通过升级 pip/setuptools/torch 或删除型号门槛处理它。

源码管理人员先将本机 `p1-repos/LMCache` 的新 `p4` 提交发布，并重新上传包含新配对清单的 `design/p4`。
本次没有自动推送；若已发布过 P4，只需追加推送 LMCache：

```bash
git -C p1-repos/LMCache push origin p4
```

以下在**每一个受影响的独立测试容器**执行。不在运行中的 editable 服务下切换代码。
仅适用于 vLLM 已用上述精确 P4 提交编译并安装成功、LMCache 尚未安装成功的情况；
否则按 §2～3 完整安装两仓。保留材料、失败日志和已有构建目录，脚本会自动建立新的 native staging。

```bash
set -euo pipefail
P4_ROOT=/workspace/zzj
P4_PY=/usr/local/python3.11.14/bin/python
cd "$P4_ROOT"
P4_VLLM_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["vllm"]["commit"])')
P4_LMC_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["LMCache"]["commit"])')
test "$P4_LMC_REF" = 1fc7439b96d091378b527819dc92902115eee58d
test "$(git -C p1-repos/vllm rev-parse HEAD)" = "$P4_VLLM_REF"
git -C p1-repos/vllm diff --quiet
git -C p1-repos/vllm diff --cached --quiet
git -C p1-repos/LMCache diff --quiet
git -C p1-repos/LMCache diff --cached --quiet
git -C p1-repos/LMCache fetch origin p4
git -C p1-repos/LMCache switch --detach "$P4_LMC_REF"

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export SOC_VERSION=ascend910b3
export VLLM_TARGET_DEVICE=ascend
export USE_MINDSPORE=0
export BUILD_WITH_HIP=0
export VLLM_USE_PRECOMPILED=0
export COMPILE_CUSTOM_KERNELS=1
export PYTHONDONTWRITEBYTECODE=1
export VLLM_PLUGINS=""
export VLLM_NO_USAGE_STATS=1

mkdir -p "$P4_ROOT/p1-repos/p4-check"
P4_RUN=$(mktemp -d "$P4_ROOT/p1-repos/p4-check/soc-retry.XXXXXXXX")
cd /tmp
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/p1_dev.py" verify --mode editable --output "$P4_RUN/vllm-installed.json"
"$P4_PY" -B "$P4_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P4_RUN/pip-before"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" doctor --output "$P4_RUN/lmcache-doctor.json"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" editable --isolated-env --output "$P4_RUN/lmcache-editable"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" verify --mode editable --output "$P4_RUN/lmcache-installed.json"
"$P4_PY" -B "$P4_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P4_RUN/pip-after"
```

`test`/Git/verify 任何一步失败都应停止核对：常见原因是仍使用旧配对清单、vLLM 未完成 P4 安装、
存在未提交修改或修复提交尚未发布；不要忽略错误继续安装。安装成功后，在同一 shell
保留 `P4_RUN`，从 §4 继续配对核对、冷导入及回归，并将所有容器的新报告归档。

源码主机已用 CMake 3.28.3 的 script mode 复现旧检查错误，修复后新增的 15 项回归全部通过。
两仓 [主机报告](results/lmcache-soc-case-20261003/verification.json) 8/8 项通过，
LMCache 145 passed / 38 subtests；Sphinx `-E -W --keep-going` 构建通过并核对生成的 HTML。
这些检查没有调用 CANN 编译器，不能替代内网 CMake 4.3.1 / Python 3.11 的编译、ABI 与 2P2D 验收。

## 4. 每节点配对与冷导入检查

从源码目录外执行，避免当前目录遮蔽安装结果。脚本不会加载模型或停止服务；
native 库导入可能访问运行时。空闲测试卡上的小量内存操作须显式选择。

```bash
cd /tmp
"$P4_PY" -B "$P4_ROOT/design/p4/tools/check_pair.py" \
  --workspace "$P4_ROOT/p1-repos" \
  --pair "$P4_ROOT/design/p4/baseline/p4-pair.json" \
  --output "$P4_RUN/pair-installed.json"
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --check-lmcache --output "$P4_RUN/bootstrap"
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/tools/validate_npu_native.py" \
  --spawn --output "$P4_RUN/no-lmcache-import.json"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/tools/p4_runtime_smoke.py" \
  --output "$P4_RUN/lmcache-import"
"$P4_PY" -B -m vllm.entrypoints.cli.main --help
"$P4_PY" -B -m vllm.entrypoints.cli.main serve --help
```

可在空闲测试卡补跑 `validate_npu_native.py --device-smoke --spawn --output <新文件>`、
`p4_runtime_smoke.py --npu --output <新目录>`。若设备 API 返回泛称 `Ascend910B` 而非具体 B3，
归档输出与 npu-smi 型号后核对，不放宽检查冒充确认。

内网 torch 已安装，补跑全部选定主机/CPU tensor 回归：

```bash
cd "$P4_ROOT"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$P4_PY" -B design/p4/tools/verify.py \
  --python "$P4_PY" --with-installed-torch --output "$P4_RUN/source-host"
```

## 5. 模型与服务回归

先保留冻结 P3 的实际启动配置和已验证 LMCache YAML，记录 checkpoint/配置/权重索引哈希，
再以 P4 两仓替换代码。仅接受 GLM-5.2 原生文本模型，不因为 architecture 相同而默认接受未来 GLM-5.3。
使用正式 `LMCacheConnectorV1`，删除旧 `lmcache_ascend.*` module path。
DSA 双组/MTP 保持启用，C8 关闭（`enable_sparse_c8=false`）；不改动 TP/DP、并发预算和压测数据掩盖差异。

| 用例 | 必须检查 |
| --- | --- |
| 离线 `LLM.generate` | 文本/预分词输入，输出非空、采样/停止条件；无在线服务依赖 |
| 无 LMCache 在线 | chat/completions、stream、取消、reasoning/tool-call 文本协议；无缓存 import |
| 2P2D TP8/DP2 | 4×8 910B3，P/D 各实例成功、请求完成、Running 并发、DSA 两组缓存绑定 |
| 2P2D TP4/DP4 | 固定拓扑和配置，与对应 P3 配对；不能以 TP8/DP2 结果代替 |
| MTP | 草稿/接受 token 计数及每节点/汇总接受率；不能平均节点百分比 |
| 缓存 | CPU KV 卸载/共享、重复前缀命中、跨实例 RemoteFill，输出/命中一致性 |
| 抢占/恢复 | checkpoint 生成→回收→冷恢复，event handoff，完成通知、超时、重试/清理不挂起 |
| 负向边界 | 非 910B3、其他模型、多模态/embeds、LoRA、pooling、CacheBlend、C8 明确失败，无静默降级 |
| 故障与长稳 | 在专用环境按已有故障矩阵注入；检查无 token 丢失/卡死/泄漏/悬挂任务 |

性能数据用相同有效请求、预热、参数与墙钟区间。剔除并单列 HTTP400/超长/零输出/取消请求。
P3 179 行中只有 176 行正输出，日志 duration 是请求延迟相加，不可拿它计算并发总吞吐。
门槛尚未预设固定退化百分比时，记录逐项差异并审核，不能自行把大幅退化判为通过。

## 6. 普通 wheel 与 sdist 重建门槛

开发态测试成功仍不等于发行制品可用。在构建容器执行：

```bash
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/p1_dev.py" build --output "$P4_RUN/vllm-wheel"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" build --output "$P4_RUN/lmcache-wheel"
"$P4_PY" -B -m build --sdist --no-isolation \
  --outdir "$P4_RUN/sdist-vllm" "$P4_ROOT/p1-repos/vllm"
"$P4_PY" -B -m build --sdist --no-isolation \
  --outdir "$P4_RUN/sdist-lmcache" "$P4_ROOT/p1-repos/LMCache"
mkdir "$P4_RUN/unpack-vllm" "$P4_RUN/unpack-lmcache"
tar -xzf "$P4_RUN/sdist-vllm/vllm-0.18.0+ascend.p4.tar.gz" -C "$P4_RUN/unpack-vllm" --strip-components=1
tar -xzf "$P4_RUN/sdist-lmcache/lmcache-0.4.3+ascend.p4.tar.gz" -C "$P4_RUN/unpack-lmcache" --strip-components=1
"$P4_PY" -B "$P4_RUN/unpack-vllm/p1_dev.py" build --output "$P4_RUN/vllm-sdist-rebuild"
"$P4_PY" -B "$P4_RUN/unpack-lmcache/p1_dev.py" build --output "$P4_RUN/lmcache-sdist-rebuild"
```

从 sdist 解包后禁止借用 checkout 的源文件/PYTHONPATH。在另一个干净测试容器使用
`p1_dev.py install --isolated-env --wheel <明确wheel路径> --output <新目录>` 安装配对 wheel，
随后 `verify --mode wheel`、冷导入和同一功能矩阵复验。核对 wheel/sdist 不含旧插件、其他设备模型/媒体实现，
归档制品 SHA256；不要求非确定性 native 编译得到的两次 wheel 二进制字节完全相同。

## 7. 验收与回滚

汇总四容器 `pair-installed.json`、doctor/pip waiver、构建/安装日志、制品哈希、
source-host/冷导入报告、精确启动配置、请求结果与 benchmark 墙钟统计。
全部必选项经审核后才能关闭 P4 验收；“CLI 可启动”或“主机测试通过”不能代替它们。
若失败，保留现场报告，不要清空 checkpoint 证据或擅自停止其他人的服务。

回滚时在独立环境从两仓 `p3-frozen-20261003` 同时重建并安装，复用冻结 P3 配置。
不要混装 P3/P4，不用 `git reset --hard`，不在运行中的 editable 服务下直接切分支。
