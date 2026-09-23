# P1 内网执行步骤（固定 p1-repos 布局）

适用：已授权的 P1 双仓源码及本批修复，不适用于原四仓的旧安装命令。所有编译、sdist/wheel、安装和 NPU 验证都在内网完成；本机仅提供源码和审计工具。当前服务继续保留，基线由其他开发人员验证后归档。

2026-09-23 pip 检查补充：用户已明确批准豁免 `op-compile-tool 0.1.0` 对 getopt、inspect、multiprocessing 的三条缺失分发包误报。新增 `tools/check_pip_dependencies.py`，安装前后都使用同一精确规则；原始输出/退出码继续归档，其他问题不豁免。本次只更新 `design/p1`，两仓源码和下文配对提交不变，详见第 3.2 节。

2026-09-23 最新候选更新：按内网实际安装信息，将两仓及构建校验统一为 `torch-npu==2.9.0.post2`、`transformers==5.2.0`；torch 仍为 2.9.0，不安装、升级或降级任何包。**本次候选调整仅在本地提交，尚未推送 GitHub；内网开始前须先发布或交接，直接 git pull 目前不会取得本次更新。** 固定提交以下文为准，不能继续使用前一批导出的 source 副本。

此前已补齐的 `p1_dev.py` 材料核验、wheel 构建/安装、strict editable 调测与路径验证保留，每次原生编译仍使用新目录。

**正式 P1 验收按本文执行；日常编译、安装与修改 Python 调测按[开发调测指南](development-build-install.md)执行。** editable 不是 P1 出口，首次仍完整编译；不得在现有 GLM 服务容器安装。本批累积清单为 `baseline/p1-intranet-candidates-20260923.json`，保留此前修复与开发安装批次的来源证据。

2026-09-21 更新：用户确认内网与本机布局相同，都是在工作区根目录创建 `p1-repos`，再从 Git 下载两个新仓。下文据此使用已有克隆，不要求移动仓库或反复联网克隆；旧 `p1-worktrees` 不再使用。

## 1. 固定源码位置与本次验证批次

内网工作区根目录按 `/workspace/zzj` 举例；如果实际位置不同，只修改 `P1_ROOT`。目录职责如下：

| 路径 | 用途 |
| --- | --- |
| `/workspace/zzj/p1-repos/vllm` | 长期保留的 vllm-dual Git 克隆 |
| `/workspace/zzj/p1-repos/LMCache` | 长期保留的 LMCache-dual Git 克隆，注意大小写 |
| `/workspace/zzj/design/p1` | 单独同步的检查工具、测试、profile 和来源清单 |
| `/workspace/zzj/p1-repos/p1-check` | 已有报告保留；新运行使用其中新的 `run.XXXXXXXX` 子目录 |
| `p1-check/run.XXXXXXXX/source/{vllm,LMCache}` | 从固定 Git 提交导出的本批构建副本，不是新增 Git 仓库 |
| `p1-check/run.XXXXXXXX/{wheels,sdist,rebuilt-wheels}` | 仅本批产物，不混用旧产物 |

旧版的 `stale ACLNN artifacts` 与本机/内网路径相同无关。新版 backend 每次都在实际源码目录下新建 `build/p1-native/run-*`，不复用旧 ACLNN 源码副本或 LMCache 已链接的设备对象；保留失败现场，不要求手动 mkdir 或删除 build。本正式验收流程仍每批导出固定提交，以明确产物来源；开发路线允许在长期源码目录反复编译。

先单独同步本次更新后的 `design/p1`，至少包含 `tools/`、`tests/`、`profile.json`、本文、开发指南、`baseline/source-manifest.json` 和 `baseline/p1-intranet-candidates-20260923.json`。两仓携带自己的 `p1_dev.py` 与轻量测试，但不包含外层正式验收工具；不重新执行 `prepare_worktrees.py`。原 `source-01` 归档继续保留，不与本次 Git 路线混装。

### 1.1 初始化变量和失败即停的日志入口

以下第 1～4 节在同一个专用 Bash 会话、同一个内网隔离构建容器执行。不要放在承载现有 GLM 服务的容器中安装依赖。新开会话重试时，从本节重新执行，产生新的批次；不要手动将 `P1_RUN` 指回已有批次。

```bash
set -euo pipefail
export P1_ROOT=/workspace/zzj
export P1_SOURCE_WORKSPACE="$P1_ROOT/p1-repos"
export P1_TOOLS="$P1_ROOT/design/p1/tools"
P1_VLLM_COMMIT=230fbc0218e656cb90f8a8c2261150345a1d8ee5
P1_LMCACHE_COMMIT=547ae7c10b0e510b864c8d0f5233d6f54f279359
test -f "$P1_TOOLS/audit_sources.py"
test -f "$P1_TOOLS/materialize_submodules.py"
test -f "$P1_TOOLS/check_pip_dependencies.py"
test -f "$P1_TOOLS/../baseline/source-manifest.json"
test -f "$P1_TOOLS/../baseline/p1-intranet-candidates-20260923.json"
mkdir -p "$P1_SOURCE_WORKSPACE/p1-check"
P1_RUN=$(mktemp -d "$P1_SOURCE_WORKSPACE/p1-check/run.XXXXXXXX")
export P1_RUN
export P1_BUILD_WORKSPACE="$P1_RUN/source"
mkdir "$P1_BUILD_WORKSPACE"
printf '本次报告目录：%s\n' "$P1_RUN"

p1_step() {
  local p1_name=$1
  shift
  local p1_log="$P1_RUN/$p1_name.log"
  local p1_rc
  if [ -e "$p1_log" ] || [ -e "$P1_RUN/$p1_name.exitcode" ]; then
    printf 'STOP: 不覆盖已有步骤，请新建批次：%s\n' "$p1_name" >&2
    exit 1
  fi
  if "$@" > "$p1_log" 2>&1; then
    printf '0\n' > "$P1_RUN/$p1_name.exitcode"
    printf 'PASS: %s\n' "$p1_name"
  else
    p1_rc=$?
    printf '%s\n' "$p1_rc" > "$P1_RUN/$p1_name.exitcode"
    tail -n 60 "$p1_log" >&2
    printf 'STOP: %s；完整日志：%s\n' "$p1_name" "$p1_log" >&2
    exit "$p1_rc"
  fi
}
```

`P1_SOURCE_WORKSPACE` 始终指向已克隆的 `p1-repos`；`P1_BUILD_WORKSPACE` 指向本批副本。本文后面的审计、host、构建均针对后者。不要把二者改成同一路径；长期源码调测另走开发指南，不能把其结果混作本批 wheel 验收。

### 1.2 核对已有 Git 克隆和配对版本

已经克隆的目录不会再次 clone。下面的条件分支只用于尚无代码的全新环境；网络只用于批准的 proxy/Git 来源，不自动配置代理或回退直连。

```bash
if [ ! -e "$P1_SOURCE_WORKSPACE/vllm" ]; then
  p1_step clone-vllm git clone --single-branch --branch main --no-tags \
    --no-recurse-submodules https://github.com/zhangzhaoju/vllm-dual.git \
    "$P1_SOURCE_WORKSPACE/vllm"
fi
if [ ! -e "$P1_SOURCE_WORKSPACE/LMCache" ]; then
  p1_step clone-lmcache git clone --single-branch --branch main --no-tags \
    --no-recurse-submodules https://github.com/zhangzhaoju/LMCache-dual.git \
    "$P1_SOURCE_WORKSPACE/LMCache"
fi
for p1_repo in vllm LMCache; do
  test -d "$P1_SOURCE_WORKSPACE/$p1_repo/.git"
  p1_step "source-status-$p1_repo" git -C "$P1_SOURCE_WORKSPACE/$p1_repo" \
    status --short --branch --untracked-files=normal
  p1_step "source-clean-$p1_repo" git -C "$P1_SOURCE_WORKSPACE/$p1_repo" \
    diff --exit-code --ignore-submodules=all HEAD --
done
test "$(git -C "$P1_SOURCE_WORKSPACE/vllm" rev-parse HEAD)" = "$P1_VLLM_COMMIT" || {
  printf 'STOP: vllm HEAD 与本批配对版本不符，请先核对，不能强制覆盖。\n' >&2
  exit 1
}
test "$(git -C "$P1_SOURCE_WORKSPACE/LMCache" rev-parse HEAD)" = "$P1_LMCACHE_COMMIT" || {
  printf 'STOP: LMCache HEAD 与本批配对版本不符，请先核对，不能强制覆盖。\n' >&2
  exit 1
}
printf 'vllm %s\nLMCache %s\n' "$P1_VLLM_COMMIT" "$P1_LMCACHE_COMMIT" \
  > "$P1_RUN/source-commits.txt"
```

本批固定为 2026-09-23 修复的两个本地新提交，须先完成批准的发布/交接再在内网使用；不要把 SHA 改回旧发布版本以绕过检查，也不能直接追随浮动 `main`。发现未提交源码改动时先由负责人保存并审核，不使用 `reset --hard`、`git clean` 或强制切分支。已有未跟踪材料、构建结果与子模块工作目录不参与下面的 Git 快照导出；它们不会被删除，固定材料在第 2 节重新核验。

### 1.3 从已有 Git 提交导出干净的构建副本

```bash
mkdir "$P1_BUILD_WORKSPACE/vllm" "$P1_BUILD_WORKSPACE/LMCache"
p1_step snapshot-vllm git -C "$P1_SOURCE_WORKSPACE/vllm" archive \
  --format=tar --output="$P1_RUN/vllm-source.tar" "$P1_VLLM_COMMIT"
p1_step snapshot-lmcache git -C "$P1_SOURCE_WORKSPACE/LMCache" archive \
  --format=tar --output="$P1_RUN/LMCache-source.tar" "$P1_LMCACHE_COMMIT"
p1_step unpack-vllm tar -xf "$P1_RUN/vllm-source.tar" -C "$P1_BUILD_WORKSPACE/vllm"
p1_step unpack-lmcache tar -xf "$P1_RUN/LMCache-source.tar" -C "$P1_BUILD_WORKSPACE/LMCache"
sha256sum "$P1_RUN/vllm-source.tar" "$P1_RUN/LMCache-source.tar" \
  > "$P1_RUN/source-tars.sha256"
test ! -e "$P1_BUILD_WORKSPACE/vllm/build"
test ! -e "$P1_BUILD_WORKSPACE/LMCache/build"
```

这是普通 Git 源码快照，不是调用 backend 的 sdist；它只包含已提交内容，不复制长期 Git 克隆中残留的 `build/`、忽略文件、材料清单或子模块载荷。不对整个 `p1-repos` 执行递归拷贝，以免把已有 `p1-check` 一并带进新批次。

## 2. 复用固定子模块并检查源码

Git 交付有一项非运行时说明文件需补齐：原 `ascend/.claude/README.md` 被忽略规则排除，未进入 `vllm-dual` 的提交，但历史来源清单要求保留。仅向本批副本补齐；下面从原 `vllm-ascend` 固定提交提取这一文件并核对 SHA-256，已有文件不覆盖，内容不符立即停止。即使长期 Git 克隆中已补过，该忽略文件也不会自动进入新的 Git 快照。此步骤不启动或调用任何 AI 服务。

```bash
P1_DONOR_NOTE="$P1_BUILD_WORKSPACE/vllm/ascend/.claude/README.md"
if [ ! -e "$P1_DONOR_NOTE" ]; then
  p1_step donor-note-export git -C "$P1_ROOT/vllm-ascend" archive \
    --format=tar --output="$P1_RUN/donor-note.tar" \
    d22f0b7cffde1b6ddb87cb44368e46193e811cc9 .claude/README.md
  p1_step donor-note-unpack tar -xf "$P1_RUN/donor-note.tar" \
    -C "$P1_BUILD_WORKSPACE/vllm/ascend"
fi
printf '91098ed23a7967cf5942b00d92e51b0a6b515f5f7ed416457afd0c7396b126b0  %s\n' \
  "$P1_DONOR_NOTE" | sha256sum --check - || exit 1
```

以下步骤只读取内网保留的原 `vllm-ascend/csrc/third_party/catlass` 和 `LMCache-Ascend/third_party/kvcache-ops` 子模块，向本批副本写入快照及材料清单，不联网。工具要求原子模块 HEAD 精确匹配且工作区干净。若环境只有两个新仓、没有这些原始材料，先由材料负责人补齐，不能把新仓中的空 gitlink 当作完整源码。不要对本批副本执行 `git submodule update`，也不要对已填充目标重复运行材料脚本。

```bash
p1_step materialize python -B "$P1_TOOLS/materialize_submodules.py" \
  --sources "$P1_ROOT" --destination "$P1_BUILD_WORKSPACE"
p1_step source-audit python -B "$P1_TOOLS/audit_sources.py" \
  --workspace "$P1_BUILD_WORKSPACE" \
  --manifest "$P1_TOOLS/../baseline/source-manifest.json" \
  --updates "$P1_TOOLS/../baseline/p1-intranet-candidates-20260923.json" \
  --output "$P1_RUN/source-audit.json"
p1_step contracts env P1_SOURCE_WORKSPACE="$P1_BUILD_WORKSPACE" \
  python -B -m unittest discover -s "$P1_TOOLS/../tests" -v
p1_step cmake-entry python -B \
  "$P1_BUILD_WORKSPACE/vllm/tests/standalone/test_p1_cmake.py" -v
p1_step dev-vllm python -B \
  "$P1_BUILD_WORKSPACE/vllm/tests/standalone/test_p1_development.py" -v
p1_step dev-lmcache python -B \
  "$P1_BUILD_WORKSPACE/LMCache/tests/standalone/test_p1_development.py" -v
p1_step dev-resources python -B \
  "$P1_BUILD_WORKSPACE/vllm/ascend/tests/standalone/test_p1_resources.py" -v
p1_step sfa-light env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q --tb=short --noconftest --import-mode=importlib \
  -c /dev/null -p no:cacheprovider \
  -o 'markers=cpu_test: dependency-light CPU tests' \
  --junitxml="$P1_RUN/sfa-light.xml" \
  "$P1_BUILD_WORKSPACE/vllm/tests/v1/core/test_scheduler_bootstrap_state.py" \
  "$P1_BUILD_WORKSPACE/vllm/ascend/tests/standalone/test_sparse_recovery_budget.py" \
  "$P1_BUILD_WORKSPACE/vllm/ascend/tests/standalone/test_staged_dummy_capacity.py" \
  "$P1_BUILD_WORKSPACE/LMCache/ascend/tests/v1/test_cache_engine_close_cpu.py"
p1_step host env PYTHONPATH="$P1_BUILD_WORKSPACE/LMCache" \
  python -B "$P1_TOOLS/run_host_checks.py" \
  --workspace "$P1_BUILD_WORKSPACE" --output "$P1_RUN/host"
```

固定提交：CATLASS `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`；kvcache-ops `9f18d2339bc58a43429f7d5bdaef1628c820eff5`。构建 helper 重新校验清单内全部文件。任何缺文件、非预期改动或失败退出都必须处理，不能修改清单或跳过失败来继续构建。

本批来源审计同时读取不可改写的历史清单与累积更新清单：既检查此前修复/开发安装，也检查本次候选声明的精确哈希。约束与预检测试 41 项（含 7 项候选版本和 10 项 pip 精确豁免回归）、CMake 解析 5 项、开发安装测试 21+21+2 项、SFA 轻量回归 55 项，均须通过且无 skip。native 命令在轻量测试中使用模拟文件，不触发原生构建，不替代真实 NPU 事件回放及完整 torch/框架测试；待补项见[合入记录第 6 节](staged-sfa-event-handoff-integration.md#6-内网剩余验证)。

host 门槛仍为 16+31+39+26=112 项通过，0 fail/error/skip。pytest 已安装，不需要再次安装，部分用例仍需已有 torch。本批旧报告的 5 项错误是 `No module named 'lmcache'`：测试从 `LMCache/ascend` 启动，但需要导入父级源码包。上面的 `env PYTHONPATH=...` 只给 host 检查及其子进程添加**本批 LMCache 源码**，不安装旧插件、不跳过断言，也不改变当前 shell 的 PYTHONPATH；不要将该设置 export 到后续 wheel 安装或运行验收。路径修正是否消除全部失败，以新 112 项报告为准。

## 3. 在隔离构建容器核对依赖

可以复用当前镜像构建独立容器，不在承载服务的容器安装依赖。保留现有 torch/NPU/triton 候选制品；先由材料负责人完成其来源与原始制品 SHA-256 核对。安装记录中的哈希不替代独立制品验证。

```bash
# 仅在内网隔离构建容器；路径按实际安装核实。
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
test -f "$ASCEND_HOME_PATH/set_env.sh"
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export SOC_VERSION=ascend910b3
export VLLM_TARGET_DEVICE=ascend
export USE_MINDSPORE=0
export BUILD_WITH_HIP=0
export COMPILE_CUSTOM_KERNELS=1
export VLLM_USE_PRECOMPILED=0
export USE_HIXL=1
export BUILD_MOONCAKE=0
export MAX_JOBS=8
p1_step preflight python -B "$P1_TOOLS/preflight.py" \
  --workspace "$P1_BUILD_WORKSPACE" --output "$P1_RUN/preflight-01.json"
p1_step pip-check-before-build python -B "$P1_TOOLS/check_pip_dependencies.py" \
  --output "$P1_RUN/pip-check-before-build"
```

预检不导入 torch/NPU，不安装任何包；必须退出 0 且报告 `passed=true`，否则会停止本批流程。先在隔离构建容器补齐缺失项，再从第 1 节建立新批次复测，不能沿用旧失败报告。本轮以 `USE_HIXL=1` 构建 CANN 8.5.1 对应的 HIXL/hcomm 模块，运行时通道仍需按服务实际配置验证，不能据此认定全部传输能力通过。

上一批 `p1-repos/p1-check/preflight-01.json` 明确缺少 `aiofile`、`awscrt`、`build>=1.2`、`opentelemetry-exporter-prometheus>=0.50b0`、`redis`、`sortedcontainers`。由内网材料负责人选定兼容版本、制品哈希并补齐其依赖；这里不自动联网安装或调整既有 torch。`pytest` 与 `build` 是不同的包，pytest 已安装不代表 `python -m build` 可用。

注意：

- 依赖入口仅为两仓 `requirements/ascend.txt` 与 `requirements/build.txt`。缺失项由内网制品库或允许的 proxy 单独准备、固定版本/哈希；不执行原始全量 requirements 安装，不自动升级 torch/numpy/OpenCV。
- 上一批两仓 sdist 均因 `No module named build` 未生成，随后重建的 `FileNotFoundError` 是连带错误。本次先通过预检，再生成 sdist，源码包不存在时不进入重建。
- 最新提供的 `installed-pip-check.txt` 还包含 CANN 工具的 `absl-py`、`ml-dtypes`、`tornado` 缺失，以及以下三项标准库元数据问题。直接依赖预检通过不能替代传递依赖检查；以新报告为准，不复用旧 P0 问题数量。
- getopt、inspect、multiprocessing 属于标准库，不能从 PyPI 安装同名包补数。用户已明确批准第 3.2 节的三条精确豁免；其他包、其他 op-compile-tool 版本、其他缺依赖/冲突仍须处理，不修改 SDK 的 METADATA 来消除报告。
- 非 C8 权重兼容性仍由基线负责人验证；不要修改 quant_model_description.json 或把 GLM 权重量化改成另一种格式。
- 如实际缓存配置使用 native Mooncake L2，不能保持 BUILD_MOONCAKE=0：先提供匹配的 include/lib 和 ABI 材料，再设为 1。RemoteFill 所需 CANN Python Mooncake/服务也要独立核对，不把这个开关视作 RemoteFill 开关。
- 代理只用于批准的材料源；不接入外部大模型/Agent，不自动上传日志或模型元数据。
- 构建容器可保留 CANN SDK 所需的 Python 路径，但不得混入旧框架源码或服务环境的 editable 路径。不要为清除 host 临时路径而无差别删除 CANN 的环境设置；第 2 节的 `env` 已限制其作用域。

### 3.1 本次候选版本不匹配的重试方法

用户最新报告的实际版本为 `torch-npu 2.9.0.post2`、`transformers 5.2.0`。本次以这两个精确版本作为候选，不要求降级至旧 post1/4.x；torch 本体仍检查 `2.9.0`（接受此前的 `2.9.0+cpu`）。其他依赖、CANN/架构及构建工具门槛不变。

preflight 从 `--workspace` 指定的两仓 `pyproject.toml` 和 `requirements/ascend.txt` 读取声明；`p1_dev.py doctor` 和 native 构建也使用两仓声明/常量。必须同步两仓代码与本批 design，不能只换 preflight 脚本。旧 `p1-check/run.*/source` 是冻结快照，不会随 Git 更新；正式验收从第 1 节重新导出新批次，不修改或覆盖旧报告。若旧 wheel 已构建，也要重新构建，不能继续使用带旧 Requires-Dist 的产物。

如需先对已更新的长期 Git 克隆做一次只读依赖预检，在已有 CANN 环境的 Bash 会话执行：

```bash
set -euo pipefail
P1_ROOT=/workspace/zzj
mkdir -p "$P1_ROOT/p1-repos/p1-check"
P1_PREFLIGHT_RUN=$(mktemp -d "$P1_ROOT/p1-repos/p1-check/preflight-candidates.XXXXXXXX")
python -B "$P1_ROOT/design/p1/tools/preflight.py" \
  --workspace "$P1_ROOT/p1-repos" \
  --output "$P1_PREFLIGHT_RUN/preflight.json"
```

这次只读检查不安装包、不编译、不导入 torch/Transformers、不分配 NPU；报告新增 `workspace` 字段，便于确认实际核对的源码目录。若仍出现旧版本字符串，优先核对报告目录及该目录内依赖文件是否更新。该独立报告不替代正式批次的 preflight 门槛。

`passed=true` 仅表示直接依赖元数据匹配。Transformers 从 4.x 切换到 5.2.0 后，仍须完成 `pip check`、tokenizer/config 加载、GLM-5.2 离线/在线及 DSA/MTP 回归；torch_npu post2 的制品来源、ABI/NPU 运行结果同样待归档。本次不重写模型适配、不豁免运行验收。

### 3.2 已批准的 pip 标准库误声明豁免

仅以下三个完整诊断被批准（标准库模块本身还须能由当前解释器找到）：

```text
op-compile-tool 0.1.0 requires getopt, which is not installed.
op-compile-tool 0.1.0 requires inspect, which is not installed.
op-compile-tool 0.1.0 requires multiprocessing, which is not installed.
```

同步更新后的 `design/p1` 后，安装前可以直接执行以下只读复核；不要求重新安装依赖、重建 wheel 或覆盖前次报告：

```bash
set -euo pipefail
P1_ROOT=/workspace/zzj
mkdir -p "$P1_ROOT/p1-repos/p1-check"
P1_PIP_RUN=$(mktemp -d "$P1_ROOT/p1-repos/p1-check/pip-waiver.XXXXXXXX")
python -B "$P1_ROOT/design/p1/tools/check_pip_dependencies.py" \
  --output "$P1_PIP_RUN/before-install"
```

脚本仍运行当前解释器的 `python -m pip check`，不安装包、不访问 index、不修改环境。若只有上述三条或其中一部分，脚本退出 0，`report.json` 中为 `passed=true`、`status=passed_with_waivers`；`pip-check.txt` 保存完整原文、`pip-check.exitcode` 和 JSON 的 `raw_returncode` 仍保留 pip 的真实退出码 1。正常无问题则为 `status=passed`。不要把原始退出码改为 0。

若还有 absl-py、ml-dtypes、tornado 等缺失、版本冲突、未知警告、pip 执行异常或实际标准库模块不可发现，则脚本退出非零并列出 `blocking_errors`。不使用 `|| true` 或仅删除文本行的方式绕过检查。直接执行未经包装的 `python -m pip check` 仍会显示这三条，批准仅体现在此检查流程，不是对 pip/SDK 的全局修改。

安装后在同一新目录下用 `--output "$P1_PIP_RUN/after-install"` 再运行一次，不能复用安装前的报告。正式流程已在构建前及第 5 节安装前后接入；已有批次补做检查时，将新报告与原批次对应归档，不覆盖旧失败证据。

## 4. 构建两个候选 wheel，并从 sdist 重建

仅当前述材料、源码审计、41 项约束/预检、5 项 CMake、44 项开发安装测试、55 项 SFA、112 项 host、直接依赖预检及带精确豁免的 pip 检查全部通过，在同一环境执行。下面复核本批步骤退出码。不要执行旧的 `VLLM_TARGET_DEVICE=empty` 或 `NO_CUDA_EXT=1` 路径。本节是正式验收路线，只使用普通 wheel；editable 调测另按开发指南。

先从尚未编译的副本生成 sdist，再分别构建源码 wheel 和 sdist 重建 wheel。所有 native 构建都发生在内网；`--no-cache-dir` 避免把缓存命中的旧 wheel 当作本轮重建证据。

```bash
for p1_gate in materialize source-audit contracts cmake-entry dev-vllm dev-lmcache dev-resources sfa-light host preflight pip-check-before-build; do
  if [ ! -f "$P1_RUN/$p1_gate.exitcode" ] || \
     [ "$(< "$P1_RUN/$p1_gate.exitcode")" != 0 ]; then
    printf 'STOP: 本批前置步骤未通过：%s\n' "$p1_gate" >&2
    exit 1
  fi
done
test ! -e "$P1_BUILD_WORKSPACE/vllm/build"
test ! -e "$P1_BUILD_WORKSPACE/LMCache/build"
mkdir "$P1_RUN/wheels" "$P1_RUN/sdist" "$P1_RUN/rebuilt-wheels" "$P1_RUN/pip-tmp"
export TMPDIR="$P1_RUN/pip-tmp"

p1_step sdist-vllm python -m build --sdist --no-isolation \
  --outdir "$P1_RUN/sdist" "$P1_BUILD_WORKSPACE/vllm"
p1_step sdist-lmcache python -m build --sdist --no-isolation \
  --outdir "$P1_RUN/sdist" "$P1_BUILD_WORKSPACE/LMCache"
test -s "$P1_RUN/sdist/vllm-0.18.0+ascend.p1.tar.gz"
test -s "$P1_RUN/sdist/lmcache-0.4.3+ascend.p1.tar.gz"

p1_step build-vllm python -m pip --disable-pip-version-check --no-cache-dir \
  wheel --verbose --no-index --no-deps --no-build-isolation \
  --wheel-dir "$P1_RUN/wheels" "$P1_BUILD_WORKSPACE/vllm"
p1_step build-lmcache python -m pip --disable-pip-version-check --no-cache-dir \
  wheel --verbose --no-index --no-deps --no-build-isolation \
  --wheel-dir "$P1_RUN/wheels" "$P1_BUILD_WORKSPACE/LMCache"

p1_step rebuild-vllm python -m pip --disable-pip-version-check --no-cache-dir \
  wheel --verbose --no-index --no-deps --no-build-isolation --no-clean \
  --wheel-dir "$P1_RUN/rebuilt-wheels" "$P1_RUN/sdist/vllm-0.18.0+ascend.p1.tar.gz"
p1_step rebuild-lmcache python -m pip --disable-pip-version-check --no-cache-dir \
  wheel --verbose --no-index --no-deps --no-build-isolation --no-clean \
  --wheel-dir "$P1_RUN/rebuilt-wheels" "$P1_RUN/sdist/lmcache-0.4.3+ascend.p1.tar.gz"
```

上面任一步非零退出就停止。若仍出现编译错误，保留**本批首次失败**的完整 `.log`、`.exitcode`、`source/` 及日志标出的 native run 目录；sdist 重建通过本批 `TMPDIR` 和 `--no-clean` 留下解包/编译现场。正式验收修复后从第 1 节新建批次。开发调测可对同一源码重新运行，backend 自动隔离每次编译，旧现场不删除。目录隔离不能证明后续算子编译或 ABI 已通过。最终 wheel 为 CPython 3.11 / aarch64 内网候选产物；尚未审计 manylinux 通用兼容性。

核对内容（每个目录应恰好一对 wheel；如文件名不同先核实，不随意选择旧产物）：

```bash
p1_step wheel-contents python -B "$P1_TOOLS/inspect_wheels.py" \
  --vllm-wheel "$P1_RUN/wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --lmcache-wheel "$P1_RUN/wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P1_RUN/wheel-contents.json"
p1_step rebuilt-wheel-contents python -B "$P1_TOOLS/inspect_wheels.py" \
  --vllm-wheel "$P1_RUN/rebuilt-wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --lmcache-wheel "$P1_RUN/rebuilt-wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P1_RUN/rebuilt-wheel-contents.json"
```

检查只证明归档结构/元数据存在，不证明 ELF 依赖、ABI、资源装载或推理通过。记录本批两条构建链产物哈希；暂不要求含构建路径/时间的二进制逐字节相同，但版本、模块集合和行为必须一致：

```bash
(
  cd "$P1_RUN"
  sha256sum \
    sdist/vllm-0.18.0+ascend.p1.tar.gz \
    sdist/lmcache-0.4.3+ascend.p1.tar.gz \
    wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl \
    wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl \
    rebuilt-wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl \
    rebuilt-wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl \
    > artifacts.sha256
)
```

## 5. 仅在干净的验证容器安装

由内网负责人准备**没有旧四包、旧 editable/.pth 和工作区 PYTHONPATH**的独立验证容器，基础 torch/NPU/CANN 与候选一致。直接建 system-site-packages venv 不保证隔离旧包。不要卸载或覆盖现有 GLM 服务容器中的包。

在该新容器设置 `P1_RUN` 为**第 1 节实际生成并已完成构建的批次绝对路径**（如 `/workspace/zzj/p1-repos/p1-check/run.…`），不要重新 `mktemp`，也不要使用旧的 `p1-check/wheels`。同步该批目录及更新后的 `design/p1` 或以相同路径挂载；`P1_ROOT` 仍按实际工作区设置。检查 `PYTHONPATH` / `.pth` 没有旧框架或 host 源码路径，保留必要且经核对的 CANN SDK 设置。离开源码目录后先检查：

```bash
set -euo pipefail
: "${P1_RUN:?请先设置已构建完成的实际批次路径}"
P1_ROOT=/workspace/zzj
P1_TOOLS="$P1_ROOT/design/p1/tools"
test -f "$P1_TOOLS/check_pip_dependencies.py"
test "$(< "$P1_RUN/wheel-contents.exitcode")" = 0
test "$(< "$P1_RUN/rebuilt-wheel-contents.exitcode")" = 0
cd "$P1_RUN"
sha256sum --check artifacts.sha256
for p1_record in install.log install.exitcode installed-packages.json \
                 install-pip-check-before install-pip-check-after; do
  if [ -e "$P1_RUN/$p1_record" ]; then
    printf 'STOP: 已有安装验收记录，不覆盖：%s\n' "$p1_record" >&2
    exit 1
  fi
done
python -B - <<'PY'
from importlib import metadata
for name in ("vllm", "vllm-ascend", "lmcache", "lmcache-ascend"):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        continue
    raise SystemExit(f"STOP: validation container already contains {name}: {dist.locate_file('')}")
print("No old framework distributions found; also review .pth and PYTHONPATH.")
PY
python -B "$P1_TOOLS/check_pip_dependencies.py" \
  --output "$P1_RUN/install-pip-check-before"
if python -m pip --disable-pip-version-check install --no-index --no-deps \
  "$P1_RUN/wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  "$P1_RUN/wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  > "$P1_RUN/install.log" 2>&1; then
  printf '0\n' > "$P1_RUN/install.exitcode"
else
  p1_install_rc=$?
  printf '%s\n' "$p1_install_rc" > "$P1_RUN/install.exitcode"
  tail -n 60 "$P1_RUN/install.log" >&2
  exit "$p1_install_rc"
fi
python -m pip list --format=json > "$P1_RUN/installed-packages.json"
python -B "$P1_RUN/source/vllm/p1_dev.py" verify --mode wheel \
  --output "$P1_RUN/vllm-install-paths.json"
python -B "$P1_RUN/source/LMCache/p1_dev.py" verify --mode wheel \
  --output "$P1_RUN/lmcache-install-paths.json"
python -B "$P1_TOOLS/check_pip_dependencies.py" \
  --output "$P1_RUN/install-pip-check-after"
```

`p1_dev.py verify` 不导入框架或分配 NPU，只检查安装模式、导入位置及 native 文件，不能当作扩展加载验收。安装前后 pip 检查均使用第 3.2 节相同的精确豁免；查看各目录的 `report.json` 判定是否继续，保留真实 pip 输出/退出码，不把检查脚本的放行状态冒充 pip 原本成功。其他非零原因不得忽略。

在预约的验证资源上进行模块/ELF 与入口检查，记录 `vllm/lmcache/vllm_ascend/lmcache_ascend` 的实际导入路径和 distribution 归属，确认它们来自这两个 wheel。检查全部 .so 的依赖解析、torch C++ ABI、CANN/HIXL/hcomm、host 扩展加载，以及 custom-op 的 op_api/lib 与 kernel 资源；不得从旧源码或旧插件补 .so 来“修复”结果。

## 6. 基线负责人交接与运行验收

两种并行布局分别归档，不混用它们的性能数据。生效配置必须包括：

```bash
export VLLM_ASCEND_DSA_UNBUNDLE=1
export VLLM_ASCEND_DSA_TWO_GROUPS=1
# 在完整启动命令中保留已有配置，并合并以下值：
# --speculative-config '{"num_speculative_tokens":1,"method":"deepseek_mtp"}'
# --additional-config 中明确 "enable_sparse_c8": false
```

这只是配置核对片段，**不是完整启动脚本**。不要把新的 additional-config 覆盖掉已有图模式、layerwise 或其他必需项；不能仅靠 enable_sparse_c8=false 推断所有缓存 dtype 已关闭 C8。需提供有效 config 和运行日志中的 KV/index dtype、量化分支、MTP accepted/rejected 统计、DSA 实际分组及层数。权重名保留不变。

TP8/DP2 每节点一实例；TP4/DP4 每节点两实例、卡组不重叠。分别归档 P/D 两侧 YAML、实际 TP/DP/local DP/rank、路由、加载的 YAML 路径和内容哈希、共享索引层数、MTP 配置、通信通道、服务代码来源。TP4 的组 token 维度配置不能直接复制到 TP8；由基线负责人提供各自已验证值。

基线与 P1 使用相同 checkpoint、tokenizer、固定输入、采样参数、并发、长度和 seed，按主设计覆盖离线/在线、冷/热缓存、CPU KV、跨实例、2P2D、图 capture/replay、MTP×DSA、不等分组、RemoteFill/checkpoint、异常恢复及性能。CPU KV/故障测试不能在未预约情况下干扰现有服务。

本轮 P1 暂留 `vllm_ascend`/`lmcache_ascend`，connector 字符串沿用当前分支；不要提前替换成尚未完成的 P2/P3 原生类名。

## 7. 归档及停止条件

保存本批 `source-commits.txt`、`source-tars.sha256`、两份 `source/*/ascend/submodule-materials.json`、历史及累积更新清单、源码审计、preflight、41 项约束/预检、5 项 CMake、44 项开发安装测试、55 项 SFA、112 项 host XML/日志、完整 torch/NPU 修复回归、全部步骤 `.log`/`.exitcode`、`artifacts.sha256`、安装路径/依赖/ABI、两组配置与基线/P1 对照报告。额外保存构建前及安装前后 pip 检查目录中的 `pip-check.txt`、`pip-check.exitcode`、`report.json`，其中记录精确豁免、批准标识和剩余错误。开发路线另存 `p1_dev.py` 的 preflight/command/result/artifact JSON 和源码 diff，不混用两条路线的报告。

当前已有的 `p1-repos/p1-check/build-vllm.log` 等首批报告保持原样。后续按 `p1-repos/p1-check/run.XXXXXXXX/` 分批归档，不覆盖顶层旧报告，不合并不同批次的成功和失败结果。需要反馈时，提供实际批次名和经审核脱敏的报告/日志；`source/`、源码 tar、wheel 等大型材料通常只留内网，另有需要时再单独提供。完整原件留内网，不自动上传，不包含凭据、业务输入、敏感地址或模型权重。

以下任一情况保持未通过：基线未归档；源码/材料校验不符；缺失真实依赖；sdist 无法独立重建；旧插件或旧路径帮助导入；native/ABI 错误；任一必保场景或性能门槛失败。缺少基线数据不阻止本轮静态工作，但不能通过 P1 出口或直接推进大规模裁剪。
