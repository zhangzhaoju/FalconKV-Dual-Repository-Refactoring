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
两仓 P4 已安装成功、遇到 `Unknown speculative decoding method: mtp` 或
`Unknown RoPE scaling type default`，或权重加载后遇到
`aclnnAddRmsNormBias` / `Get regInfo failed` 的容器使用 §3.2。

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
当时 vLLM 为 `0879e419001f424700569936962631ca0360f5af`；后续 Python 控制流修复已更新两仓，见 §3.2。
包版本仍为 `.p4`，所以必须按最新配对清单的 SHA 核对，不能用包版本判断修复是否生效。
普通 wheel 与 editable 共用此构建路径，二者均需该修复；切换为普通安装不能绕过原缺陷。
pip root、setuptools_scm/vcs-versioning、manifest 及本次 kineto 警告均不是这个终止点。
不要通过升级 pip/setuptools/torch 或删除型号门槛处理它。

源码管理人员先将本机 `p1-repos/LMCache` 的新 `p4` 提交发布，并重新上传包含新配对清单的 `design/p4`。
本次没有自动推送；仅补发该 SoC 修复且 vLLM 已符合最新配对时，只需追加推送 LMCache。
后续 RoPE/缓存修复需发布两仓；最新 OPP 注册修复只更新 vLLM，配对见 §3.2。

```bash
git -C p1-repos/LMCache push origin p4
```

以下在**每一个受影响的独立测试容器**执行。不在运行中的 editable 服务下切换代码。
仅适用于 vLLM 已完成 P4 编译安装并与最新配对清单一致、LMCache 尚未安装成功的情况；
否则按 §2～3 完整安装两仓。保留材料、失败日志和已有构建目录，脚本会自动建立新的 native staging。

```bash
set -euo pipefail
P4_ROOT=/workspace/zzj
P4_PY=/usr/local/python3.11.14/bin/python
cd "$P4_ROOT"
P4_VLLM_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["vllm"]["commit"])')
P4_LMC_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["LMCache"]["commit"])')
test "$(git -C p1-repos/vllm rev-parse HEAD)" = "$P4_VLLM_REF"
git -C p1-repos/vllm diff --quiet
git -C p1-repos/vllm diff --cached --quiet
git -C p1-repos/LMCache diff --quiet
git -C p1-repos/LMCache diff --cached --quiet
git -C p1-repos/LMCache fetch origin p4
git -C p1-repos/LMCache merge-base --is-ancestor 1fc7439b96d091378b527819dc92902115eee58d "$P4_LMC_REF"
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

<a id="mtp-startup-retry"></a>
<a id="custom-opp-retry"></a>

### 3.2 P4 Python 启动修复与复测（2026-10-04）

此前已修复 MTP、RoPE、权重加载、Ray 初始化和 LMCache 最后一次 prefill 尾块保存。
最新日志已进入权重加载后的预热，但四节点均报告 `aclnnAddRmsNormBias` 的算子信息缺失。
本轮补回 P4 裁剪时丢失的正式启动入口 `ASCEND_CUSTOM_OPP_PATH` 注册；
不恢复已删除的 GPU 模块，不改变原生算子实现或 ABI。
当前配对为 vLLM `137fd3bc144875075ae73be2782ec8b5abf5a002` 与
LMCache `241166dfde710e9a2cbe24ba0bd24b094b4bbe41`，包含上述全部修复。
保留 checkpoint 原 RoPE 配置、MTP、DSA 双组、TP8/DP2、`max_num_seqs=16` 和 C8 关闭设置；
不通过调整模型配置或关闭功能绕过错误。
证据及验证边界见 [算子注册分析报告](custom-opp-startup-fix-20261004.md)；
[RoPE 报告](rope-startup-fix-20261004.md) 和 [MTP 报告](mtp-startup-fix-20261004.md)
保留为历史记录，不再按其旧配对运行。

本次只修改 vLLM 的 Python 启动/检查代码，LMCache 保持上一轮提交；包版本和运行模块集合不变。
**若两仓已成功安装 P4 strict editable，且下面源码身份与本包算子资源检查均通过，无需重新编译或重装。**
必须先停止自己这轮 P4 测试的进程，再切换源码，并用新进程重启；不能在运行中热切换。
不停止他人的服务，不修改 P3 基线。若尚未完成 P3→P4 升级，仍执行 §2～3 的完整重建。

源码管理人员先发布 vLLM 修复，并重新上传当前 `design/p4`（包含配对清单、工具和测试）。
LMCache `241166df` 若已发布无需再次更新；下面第二条推送只是补齐尚未发布的上一轮修复：

```bash
git -C p1-repos/vllm push origin p4
git -C p1-repos/LMCache push origin p4
```

本次没有自动推送。以下在四个独立测试容器逐一执行；同镜像不能替代逐容器核对。

```bash
set -euo pipefail
P4_ROOT=/workspace/zzj
P4_PY=/usr/local/python3.11.14/bin/python
cd "$P4_ROOT"
P4_VLLM_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["vllm"]["commit"])')
P4_LMC_REF=$($P4_PY -B -c 'import json; print(json.load(open("design/p4/baseline/p4-pair.json"))["repositories"]["LMCache"]["commit"])')
test "$P4_VLLM_REF" = 137fd3bc144875075ae73be2782ec8b5abf5a002
test "$P4_LMC_REF" = 241166dfde710e9a2cbe24ba0bd24b094b4bbe41
for P4_REPO in vllm LMCache; do
  git -C "p1-repos/$P4_REPO" diff --quiet
  git -C "p1-repos/$P4_REPO" diff --cached --quiet
  git -C "p1-repos/$P4_REPO" fetch origin p4
done
git -C p1-repos/vllm switch --detach "$P4_VLLM_REF"
git -C p1-repos/LMCache switch --detach "$P4_LMC_REF"
git -C p1-repos/vllm rev-parse HEAD
git -C p1-repos/LMCache rev-parse HEAD

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export VLLM_TARGET_DEVICE=ascend
export VLLM_PLUGINS=""
export VLLM_NO_USAGE_STATS=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$P4_ROOT/p1-repos/p4-check"
P4_RUN=$(mktemp -d "$P4_ROOT/p1-repos/p4-check/python-retry.XXXXXXXX")
cd /tmp
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/p1_dev.py" verify --mode editable --output "$P4_RUN/vllm-installed.json"
"$P4_PY" -B "$P4_ROOT/p1-repos/LMCache/p1_dev.py" verify --mode editable --output "$P4_RUN/lmcache-installed.json"
"$P4_PY" -B "$P4_ROOT/design/p4/tools/check_pair.py" \
  --workspace "$P4_ROOT/p1-repos" \
  --pair "$P4_ROOT/design/p4/baseline/p4-pair.json" \
  --output "$P4_RUN/prestart-pair-installed.json"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$P4_PY" -B -m pytest -q \
  --noconftest -p no:cacheprovider \
  "$P4_ROOT/p1-repos/vllm/tests/standalone/test_p4_mtp_runner.py" \
  "$P4_ROOT/p1-repos/vllm/tests/standalone/test_p4_model_startup.py" \
  "$P4_ROOT/p1-repos/vllm/tests/standalone/test_p4_custom_opp.py" \
  --junitxml="$P4_RUN/vllm-startup.xml" 2>&1 | tee "$P4_RUN/vllm-startup.log"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$P4_PY" -B -m pytest -q \
  --noconftest -p no:cacheprovider \
  "$P4_ROOT/p1-repos/LMCache/tests/standalone/test_p4_save_boundary.py" \
  --junitxml="$P4_RUN/lmcache-save.xml" 2>&1 | tee "$P4_RUN/lmcache-save.log"
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --check-lmcache --output "$P4_RUN/prestart-bootstrap"
```

90 项 vLLM MTP/模型启动/OPP 注册和 19 项 LMCache 保存边界回归，以及安装/配对/冷导入检查必须通过。
启动源码身份核对包含 MTP、RoPE、权重加载、Ray 和本轮 native-ops 初始化文件。
`check_pair.py` 另核对全部已跟踪包内 Python 文件（包括 LMCache adapter），
防止旧 strict editable 硬链接或其他安装残留。
`check_npu_bootstrap.py` 经正式启动入口检查 `ASCEND_CUSTOM_OPP_PATH` 首项为当前安装包的
`_cann_ops_custom/vendors/vllm-ascend`，并检查/记录该目录下
`op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json` 的存在、JSON 可读性及哈希。
这不是算子执行验证，也不校验该 JSON 的所有条目和其引用的全部二进制文件。
任何一步失败先停止处理；不要删除日志、强制 reset、设置临时 PYTHONPATH 或跳过配对检查。

若源码正确但 editable 链接树仍旧，按 §3 重装受影响的仓库；若本包 vendor 目录或上述元数据缺失，
按 §3 重建并重装 vLLM，保留失败日志与构建材料，安装脚本自动创建新的 native staging。
不要手工复制其他安装的资源、设置其他版本的 OPP 路径、禁用 custom op 或混用 P3 产物。
普通 wheel 需重建安装本轮 vLLM；LMCache wheel 已对应 `241166df` 时无需再重建，
否则也更新到该提交。随后用 `verify --mode wheel` 和 §4 的配对检查；仅 Git 切换不会更新 wheel。
LMCache 必须同步更新至 `241166df`，不能仍停留在 `1fc7439b`；
仅在 P4 strict editable 安装身份及资源检查通过时，才能复用既有安装和 native 制品。

在四个测试容器完成上述检查后，建议先跑本次真实算子探针再启动模型。
**须确认本轮测试服务已停止，且当前可见的逻辑 NPU 0 对应空闲测试卡**；
必要时由操作者在启动这个新进程前按既有规则限定设备可见性，不能抢占他人服务的卡。
本命令不加载模型，但会初始化 NPU 并分配少量张量，在主进程和 spawn 子进程分别执行
float32/bfloat16 × 有/无 bias 共四个 `AddRmsNormBias` 用例，与内置 AddRmsNorm 加 bias 对照：

```bash
cd /tmp
"$P4_PY" -B "$P4_ROOT/p1-repos/vllm/tools/validate_npu_native.py" \
  --device-smoke --spawn --output "$P4_RUN/add-rmsnorm-bias-device.json"
```

要求两进程 `passed=true`、`device_smoke.add_rms_norm_bias` 中四项均通过。
如果仍报 `Get regInfo failed`，归档这个新报告、上述元数据哈希及原生构建日志后继续排查制品，
不要通过升级/降级 CANN、torch 或修改模型配置绕过；本机尚未执行该 NPU 探针。

四容器均通过后，保留 `P4_RUN` 继续 §4 的检查（本节使用 `prestart-` 路径，不覆盖下节报告），
并用原启动参数重新启动 2P2D。验收不仅是异常消失：须四实例完成权重加载、KV 初始化/预热，
确认请求非空完成、DSA 双组与 MTP 草稿/接受计数有效、decode 并发和缓存恢复正常。
新增回归使用主机替身，不验证 NPU 数值或性能；实际结果仍按 §5 与冻结 P3 对照归档。

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

可在空闲测试卡补跑 §3.2 的 `validate_npu_native.py --device-smoke --spawn`（包括四项真实算子用例）
以及 `p4_runtime_smoke.py --npu --output <新目录>`。默认不带设备开关的检查不证明算子可执行。
若设备 API 返回泛称 `Ascend910B` 而非具体 B3，
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
| 缓存 | CPU KV 卸载/共享、重复前缀命中、跨实例 RemoteFill，输出/命中一致性；非逐层保存的最后一次 prefill 保留不足一个 chunk 的尾块 |
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
