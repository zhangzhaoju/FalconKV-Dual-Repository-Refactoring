# P1 编译、安装与开发调测

更新：2026-09-23。本轮在两个独立 P1 仓内提供 `p1_dev.py`，保留正式 wheel 验收，同时增加 strict editable 开发安装。**本机没有运行 CANN 编译、pip 安装或 NPU 推理；这些能力仍须在内网目标环境验证。**

## 1. 选择路线

| 路线 | 使用位置 | Python 修改 | P1 出口 |
| --- | --- | --- | --- |
| wheel 构建/安装 | 独立构建、验证容器 | 重新构建并安装 | 正式验收路线，须补 sdist 重建、ABI 和运行对照 |
| strict editable | 专用调测容器 + 长期保留的两仓源码 | 已有 Python 文件修改后重启调测进程；不必重编译 | 仅用于开发，不代替 wheel 验收 |

首次 editable 安装也会完整编译算子，没有“跳过编译”或借用旧四仓 .so 的路径。新增/重命名文件、修改原生算子、依赖或 entry point 后，重新执行 editable 安装。严格模式的链接目录和限制参考 [Setuptools 官方说明](https://setuptools.pypa.io/en/stable/userguide/development_mode.html#strict-editable-installs)。

每次编译建立新的 `build/p1-native/run-*`，完整生成并校验算子、host 扩展、CANN 资源和版本信息后才发布到 wheel 或 editable 映射。旧目录不会被删除或作为新构建输入。开发安装依赖 `build/__editable__.*` 及其指向的 native 目录；安装期间不要删除 build，也不要挪动源码目录。

## 2. 环境与源码前置条件

以下命令只在**专用隔离容器**执行，不在当前 GLM 基线服务容器执行。不创建共享旧 site-packages 的伪隔离环境，不自动卸载基线包。由内网负责人准备无旧 vllm/vllm-ascend/LMCache/LMCache-Ascend 安装及旧源码路径污染的环境；工具会拒绝覆盖检测到的旧 distribution。

沿用 910B3、aarch64/Python 3.11、CANN 8.5.1、torch 2.9.0、triton-ascend `3.2.0.dev20260322`；按用户最新内网安装信息，将候选固定为 torch_npu `2.9.0.post2`、Transformers `5.2.0`。不升级或降级现有包。构建依赖仍为两仓的 `requirements/build.txt` 和 `requirements/ascend.txt`，缺失项经内网制品库/批准的 proxy 准备；工具不下载或安装依赖。元数据匹配不代表 Transformers 5 接口或 NPU ABI 已验收。

先取得本批两个代码提交和更新后的 `design/p1`。本轮代码只在本地提交，尚未推送；内网不能用旧 main 代替。具体固定 SHA 以[主流程](intranet-next-steps.md)为准。

```bash
set -euo pipefail
export P1_ROOT=/workspace/zzj
export P1_REPOS="$P1_ROOT/p1-repos"
mkdir -p "$P1_REPOS/p1-check"
P1_DEV_RUN=$(mktemp -d "$P1_REPOS/p1-check/dev.XXXXXXXX")
export P1_DEV_RUN

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
cd "$P1_DEV_RUN"
```

保留 CANN SDK 需要的 Python/动态库路径，但剔除旧四仓 PYTHONPATH/editable 路径。不要从 `p1-repos/vllm` 或 `p1-repos/LMCache` 根目录启动推理：当前目录可能遮蔽已安装包，尤其会绕过 strict editable 的资源链接。上面已切到本批日志目录。

## 3. 一次性准备固定子模块材料

已有原四仓材料时，直接从已初始化且干净的固定子模块复制：

```bash
python -B "$P1_REPOS/vllm/p1_dev.py" materials \
  --from-submodule "$P1_ROOT/vllm-ascend/csrc/third_party/catlass"
python -B "$P1_REPOS/LMCache/p1_dev.py" materials \
  --from-submodule "$P1_ROOT/LMCache-Ascend/third_party/kvcache-ops"
```

若只保留两个新仓，也可由材料负责人先通过批准的 Git/proxy 初始化两仓自身固定 gitlink，再注册：

```bash
# 这一步可能联网；仅在来源/proxy 已批准时执行，不属于构建时自动下载。
git -C "$P1_REPOS/vllm" submodule update --init --checkout -- ascend/csrc/third_party/catlass
git -C "$P1_REPOS/LMCache" submodule update --init --checkout -- ascend/third_party/kvcache-ops
python -B "$P1_REPOS/vllm/p1_dev.py" materials \
  --from-submodule "$P1_REPOS/vllm/ascend/csrc/third_party/catlass"
python -B "$P1_REPOS/LMCache/p1_dev.py" materials \
  --from-submodule "$P1_REPOS/LMCache/ascend/third_party/kvcache-ops"
```

两种材料准备方式择一。CATLASS 固定 `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`；kvcache-ops 固定 `9f18d2339bc58a43429f7d5bdaef1628c820eff5`。工具保存 `ascend/submodule-materials.json`，验证 commit、干净状态及逐文件哈希。已有有效清单可重复核验；不覆盖已有的不同材料，不删除旧构建文件，也不会更新子模块版本。

初次交接仍按主流程检查固定版本与源码审计。调测中自行修改源码后，需要另存 diff/提交和测试记录，不能改历史来源清单让开发修改冒充原始交付。

## 4. 依赖检查与本机可用的轻量测试

```bash
python -B "$P1_REPOS/vllm/p1_dev.py" doctor --output "$P1_DEV_RUN/vllm-doctor.json"
python -B "$P1_REPOS/LMCache/p1_dev.py" doctor --output "$P1_DEV_RUN/lmcache-doctor.json"
python -B "$P1_ROOT/design/p1/tools/check_pip_dependencies.py" \
  --output "$P1_DEV_RUN/pip-check-before"

python -B "$P1_REPOS/vllm/tests/standalone/test_p1_development.py" -v
python -B "$P1_REPOS/LMCache/tests/standalone/test_p1_development.py" -v
python -B "$P1_REPOS/vllm/ascend/tests/standalone/test_p1_resources.py" -v
```

doctor 检查依赖/材料并在元数据通过后用 torch 子进程读取构建路径与 C++ ABI；不分配 NPU、不编译。必须 `passed=true` 才继续。pip 检查只豁免用户明确批准的 `op-compile-tool 0.1.0` 对 getopt/inspect/multiprocessing 的三条误声明，其他错误仍然停止，原始输出和退出码保留。须同步最新版 design 工具，详见主流程第 3.2 节。新增轻量测试为 21+21+2 项，native 命令使用模拟文件，不代表 CANN 编译成功；还须执行主流程的 41 项约束/预检、55 项 SFA 轻量回归和 112 项 host。

## 5A. wheel 构建与安装

下面是便于调测的逐仓入口。正式验收使用主流程导出的固定提交构建副本；将这里的两仓路径换成 `P1_BUILD_WORKSPACE` 即可，不能省略 sdist 重建等门槛。

```bash
python -B "$P1_REPOS/vllm/p1_dev.py" build --output "$P1_DEV_RUN/vllm-wheel"
python -B "$P1_REPOS/LMCache/p1_dev.py" build --output "$P1_DEV_RUN/lmcache-wheel"

python -B "$P1_REPOS/vllm/p1_dev.py" install --isolated-env \
  --wheel "$P1_DEV_RUN/vllm-wheel/wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P1_DEV_RUN/vllm-install"
python -B "$P1_REPOS/LMCache/p1_dev.py" install --isolated-env \
  --wheel "$P1_DEV_RUN/lmcache-wheel/wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P1_DEV_RUN/lmcache-install"

python -B "$P1_REPOS/vllm/p1_dev.py" verify --mode wheel --output "$P1_DEV_RUN/vllm-paths.json"
python -B "$P1_REPOS/LMCache/p1_dev.py" verify --mode wheel --output "$P1_DEV_RUN/lmcache-paths.json"
```

## 5B. editable 开发安装

与 5A 择一；仅在专用调测环境执行。两个仓都要安装，不再单独安装两个 Ascend distribution。

```bash
python -B "$P1_REPOS/vllm/p1_dev.py" editable --isolated-env --output "$P1_DEV_RUN/vllm-editable"
python -B "$P1_REPOS/LMCache/p1_dev.py" editable --isolated-env --output "$P1_DEV_RUN/lmcache-editable"

python -B "$P1_REPOS/vllm/p1_dev.py" verify --mode editable --output "$P1_DEV_RUN/vllm-paths.json"
python -B "$P1_REPOS/LMCache/p1_dev.py" verify --mode editable --output "$P1_DEV_RUN/lmcache-paths.json"
```

也支持标准 pip 入口；它没有脚本的隔离确认、旧包检查及日志归档，因此优先用上面的脚本：

```bash
# 只在已确认的专用调测容器；已经准备材料/依赖。
python -m pip install --no-index --no-deps --no-build-isolation \
  --config-settings editable_mode=strict -e "$P1_REPOS/vllm"
python -m pip install --no-index --no-deps --no-build-isolation \
  --config-settings editable_mode=strict -e "$P1_REPOS/LMCache"
```

`--isolated-env` 是操作人员对环境用途的明确确认，不是自动创建隔离容器。脚本还会拒绝旧四包/冲突版本，但无法判断某个进程是否正在使用同版本 P1；安装/重装前应结束当前专用调测进程，不能热替换正在使用的算子。

`verify` 只检查 distribution、导入路径、构建元数据和 native 文件是否齐全，不加载 .so、不证明 ABI 或 NPU 正确。完成 5A 或 5B 的两仓安装后，再以相同规则核对实际安装依赖：

```bash
python -B "$P1_ROOT/design/p1/tools/check_pip_dependencies.py" \
  --output "$P1_DEV_RUN/pip-check-after"
```

不能复用安装前报告或将未知错误视为标准库豁免。之后仍须按主流程执行实际扩展加载、预约卡上的事件回放和模型测试。

## 6. 重试、日志与验收边界

任意 build/editable/install 命令可加 `--dry-run` 查看完整命令；该模式不建目录、不编译、不安装。正常执行全部使用 `--no-index --no-deps`，构建使用 `--no-build-isolation`，保留当前 torch 制品。

每个 `--output` 必须是新目录。失败后创建新的输出目录重试即可，backend 自动使用新的 native 目录；不需要手工 mkdir 自定义算子目录或 rm -rf build。旧 LMCache 日志中的 `cdf.cpp.o` 被确认是 ELF EXEC，重链接已有设备产物是本次防复用处理的依据；**尚未有这版工具的内网成功构建日志，不能宣称所有 CANN 链接问题已解决**。若干净构建仍失败，保留本轮 `[P1] Fresh native build directory`、详细编译命令与对象头继续分析。

归档每次的 `preflight.json`、`command.log`、`command-result.json`，成功 wheel 的 `artifact.json`（含 SHA-256）、安装路径报告，以及两仓 `git rev-parse HEAD`/`git status`/开发 diff。失败真实退出码不改为成功；新批次不覆盖旧报告。

P1 出口保持不变：两个 wheel、sdist 独立重建、无旧插件导入污染、ABI/NPU 验证、GLM-5.2 的 TP8/DP2 与 TP4/DP4 两布局，以及 DSA 双组/MTP/C8-off、缓存与恢复能力的基线对照。editable 成功或本机轻量测试通过均不能替代这些项目。
