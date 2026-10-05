# P4 editable 安装与内网验证指南

更新：2026-10-04。本页默认验证已完成根 `ascend/` 移除的
`refactor/native-layout` 分支。它继承冻结 P4，不是继续推进 `p4` 分支。
**这次两仓都必须重新编译并安装 strict editable，不能复用旧 P4 的安装或 `.so`。**

| 目标 | vLLM / LMCache 包版本 | 配对记录与指导 |
| --- | --- | --- |
| 本次目录整改，默认验证对象 | `0.18.0+ascend.layout1` / `0.4.3+ascend.layout1` | [native-layout 配对清单](../native-layout/baseline/pair.json)，执行本页 |
| 冻结 P4，回退及对照 | `0.18.0+ascend.p4` / `0.4.3+ascend.p4` | 标签 `p4-frozen-20261004`、[冻结记录](baseline/p4-frozen-20261004.json)、[历史指南](intranet-validation-frozen-p4.md) |

本次固定 SHA：vLLM `18fde02dc54a432f81ce5350c3358429a40f4bd3`，
LMCache `e8236d8e1c41cd10752204d8cb9ef3b6074f999e`。
不要给新分支传入旧 `design/p4/baseline/p4-pair.json`，也不要混用 `.p4` 与 `.layout1`。

## 1. 上传范围和执行环境

将当前 `design/p4/`、`design/native-layout/` 及
`design/p1/tools/check_pip_dependencies.py` 上传至 `/workspace/zzj/design/`。
若远端尚未发布配对源码，由源码管理人员按
[发布指令](../native-layout/intranet-validation.md#publish) 发布两仓分支和冻结标签；
不强推，不移动已有标签，不重跑 P1～P4 的合仓、裁剪或目录迁移脚本。

使用专用验证容器，不改动正在服务的冻结 P4 环境。依赖保持：910B3、
Python 3.11.14/aarch64、CANN 8.5.1、torch 2.9.0+cpu、torch-npu 2.9.0.post2、
transformers 5.2.0、triton-ascend 3.2.0.dev20260322。
`torch` 的 `+cpu` 后缀不是 CPU 推理模式，NPU 能力由 torch_npu 提供。
pytest 继续使用内网已安装版本，不在本流程隐式升级依赖。

**四个运行容器都要安装两仓并各自核对，不能用“相同镜像”替代安装检查。**
Git 获取可走既定 proxy；构建安装不访问包索引，不向外部工具开放内网模型。

以下所有命令在同一个 Bash 会话中继续；任一步失败立即停止。默认保留原路径
`/workspace/zzj/p1-repos`，仅适用于新容器或已检出本次精确配对的独立测试目录。
若此路径仍属于旧 P4/运行中 editable 环境，先把下面 `P4_REPOS` 改为
`/workspace/zzj/layout-repos`，使用新目录，并同步修改后续启动自动化的路径。

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
P4_ROOT=/workspace/zzj
P4_REPOS="$P4_ROOT/p1-repos"
P4_DESIGN="$P4_ROOT/design"
P4_PY=/usr/local/python3.11.14/bin/python
P4_PAIR="$P4_DESIGN/native-layout/baseline/pair.json"
test -x "$P4_PY"
test -f "$P4_PAIR"
test -f "$P4_DESIGN/p4/tools/check_pair.py"
test -f "$P4_DESIGN/p1/tools/check_pip_dependencies.py"
mkdir -p -- "$P4_ROOT/p4-check"
P4_RUN=$(mktemp -d "$P4_ROOT/p4-check/layout-editable.XXXXXXXX")
printf '本容器报告目录：%s\n' "$P4_RUN"
cd /tmp

"$P4_PY" -B - <<'PY' | tee "$P4_RUN/environment-before.json"
from importlib import metadata
import json
import platform
import socket
import sys

result = {"host": socket.gethostname(), "python": sys.executable,
          "python_version": platform.python_version(),
          "machine": platform.machine(), "distributions": {}}
for name in ("vllm", "lmcache", "vllm-ascend", "lmcache-ascend",
             "torch", "torch-npu", "transformers", "triton-ascend"):
    try:
        dist = metadata.distribution(name)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        item = {"version": dist.version,
                "editable": direct.get("dir_info", {}).get("editable", False)}
        if direct.get("url", "").startswith("file:"):
            item["source_url"] = direct["url"]
        result["distributions"][name] = item
    except metadata.PackageNotFoundError:
        result["distributions"][name] = None
print(json.dumps(result, indent=2))
PY
```

## 2. 专用测试容器的旧安装处理

若上一步没有旧框架包，跳过本节。若已有同一源码路径、同一 `.layout1` 配对安装，
也不必先卸载，后面的安装脚本会重建并重装。

若存在 `.p1/.p2/.p3/.p4`、其他版本，或独立 `vllm-ascend/lmcache-ascend`，
优先使用仅含依赖的新验证容器。必须复用**已经停止本轮测试进程的专用容器**时，
在保存上一步记录、确认它不承载保留的基线服务后，才执行下面这一块：

```bash
"$P4_PY" -B -m pip --disable-pip-version-check --no-color uninstall -y \
  vllm lmcache vllm-ascend lmcache-ascend \
  2>&1 | tee "$P4_RUN/framework-uninstall.log"
```

这里只卸载列出的框架包，不卸载 torch/torch_npu、transformers 等依赖，不删除源码、
模型、缓存或 `build/`。旧环境要恢复时须按冻结配对重新安装。
不要清空全部 `PYTHONPATH`（其中可能有 CANN 路径）；只核查并移除自己添加的旧框架路径。
若卸载后仍有旧 `.pth`/导入遮蔽，先排查所属安装，不用 `rm -rf` 或强行改 `sys.path` 绕过。
`--isolated-env` 是专用环境声明，不会自动创建环境、卸载旧包或忽略版本冲突。

## 3. 检出精确配对与准备材料

下面只在目标不存在时克隆并检出固定 SHA；不会切换已有工作目录、覆盖本地修改。
已有目录必须已经是本次配对，否则身份检查会停止。不要用旧自动化脚本覆盖这里的 SHA。

```bash
P4_VLLM_REF=$("$P4_PY" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["repositories"]["vllm"]["commit"])' "$P4_PAIR")
P4_LMC_REF=$("$P4_PY" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["repositories"]["LMCache"]["commit"])' "$P4_PAIR")
mkdir -p -- "$P4_REPOS"
if [ ! -e "$P4_REPOS/vllm" ] && [ ! -L "$P4_REPOS/vllm" ]; then
  git clone --branch refactor/native-layout --single-branch \
    git@github.com:zhangzhaoju/vllm-dual.git "$P4_REPOS/vllm"
  git -C "$P4_REPOS/vllm" switch --detach "$P4_VLLM_REF"
fi
if [ ! -e "$P4_REPOS/LMCache" ] && [ ! -L "$P4_REPOS/LMCache" ]; then
  git clone --branch refactor/native-layout --single-branch \
    git@github.com:zhangzhaoju/LMCache-dual.git "$P4_REPOS/LMCache"
  git -C "$P4_REPOS/LMCache" switch --detach "$P4_LMC_REF"
fi

"$P4_PY" -B - "$P4_REPOS" "$P4_PAIR" <<'PY' | tee "$P4_RUN/source-pair.log"
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
pair = json.loads(Path(sys.argv[2]).read_text())
for name, expected in pair["repositories"].items():
    repo = root / name
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    print(name, actual, flush=True)
    if actual != expected["commit"]:
        raise SystemExit("STOP: checkout differs from native-layout pair; use a fresh directory")
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    retired = repo / "ascend"
    if retired.exists() or retired.is_symlink():
        raise SystemExit(f"STOP: retired root remains: {retired}; preserve it and use a fresh checkout")
print("PASS: paired source commits; tracked files clean; no root ascend/")
PY

git -C "$P4_REPOS/vllm" submodule update --init -- csrc/third_party/catlass
git -C "$P4_REPOS/LMCache" submodule update --init -- third_party/kvcache-ops
"$P4_PY" -B "$P4_REPOS/vllm/p1_dev.py" materials \
  --from-submodule "$P4_REPOS/vllm/csrc/third_party/catlass" \
  | tee "$P4_RUN/materials-vllm.json"
"$P4_PY" -B "$P4_REPOS/LMCache/p1_dev.py" materials \
  --from-submodule "$P4_REPOS/LMCache/third_party/kvcache-ops" \
  | tee "$P4_RUN/materials-lmcache.json"
"$P4_PY" -B "$P4_REPOS/vllm/tools/check_native_layout.py" \
  2>&1 | tee "$P4_RUN/layout-vllm.log"
"$P4_PY" -B "$P4_REPOS/LMCache/tools/check_native_layout.py" \
  2>&1 | tee "$P4_RUN/layout-lmcache.log"
```

固定材料：CATLASS `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`、kvcache-ops
`9f18d2339bc58a43429f7d5bdaef1628c820eff5`。若使用已有审核 Git 材料，跳过相应
`submodule update`，把 `--from-submodule` 改为该材料的绝对路径；脚本会核对提交和内容。
材料清单现在是两仓根 `submodule-materials.json`，不再是 `ascend/submodule-materials.json`。
内网必须登记真实材料并通过完整检查，不能加 `--source-only` 掩盖材料缺失。

## 4. 依赖门槛与两仓 editable 安装

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
export VLLM_PLUGINS=""
export VLLM_NO_USAGE_STATS=1
cd /tmp

"$P4_PY" -B "$P4_DESIGN/p1/tools/check_pip_dependencies.py" \
  --output "$P4_RUN/pip-before"
"$P4_PY" -B "$P4_REPOS/vllm/p1_dev.py" doctor \
  --output "$P4_RUN/vllm-doctor.json"
"$P4_PY" -B "$P4_REPOS/LMCache/p1_dev.py" doctor \
  --output "$P4_RUN/lmcache-doctor.json"
"$P4_PY" -B "$P4_REPOS/vllm/p1_dev.py" editable --isolated-env \
  --output "$P4_RUN/vllm-editable"
"$P4_PY" -B "$P4_REPOS/LMCache/p1_dev.py" editable --isolated-env \
  --output "$P4_RUN/lmcache-editable"
```

脚本名称仍叫 `p1_dev.py`，但当前分支已适配 `.layout1`。内部使用
`--no-index --no-deps --no-build-isolation --force-reinstall` 和 `editable_mode=strict`，
不会安装/升级依赖。pip check 仅豁免此前批准的 op-compile-tool 0.1.0 对
`getopt/inspect/multiprocessing` 的三条标准库误声明，其余错误仍阻断。

无需手工创建 `_cann_ops_custom`。每次编译使用新的 `build/p1-native/run-*`；
**安装成功后保留该目录及 `build/__editable__.*`，不能清理它们再启动服务。**
失败日志在各安装报告目录的 `command.log` / `command-result.json`，不要只截最后的 pip 错误。
重试时创建新的 `P4_RUN`；工具拒绝覆盖已有报告，不要删除失败证据以复用输出路径。

## 5. 逐容器安装身份、配对与冷启动检查

必须从源码目录外执行；不能把源码目录加入 `PYTHONPATH` 来“修复”缺模块。
以下只做安装检查、冷导入和 CLI 解析，不启动模型；导入原生库仍可能访问 CANN 运行时。

```bash
cd /tmp
"$P4_PY" -B "$P4_REPOS/vllm/p1_dev.py" verify --mode editable \
  --output "$P4_RUN/vllm-installed.json"
"$P4_PY" -B "$P4_REPOS/LMCache/p1_dev.py" verify --mode editable \
  --output "$P4_RUN/lmcache-installed.json"
"$P4_PY" -B "$P4_DESIGN/p4/tools/check_pair.py" \
  --workspace "$P4_REPOS" --pair "$P4_PAIR" \
  --output "$P4_RUN/pair-installed.json"
"$P4_PY" -B "$P4_DESIGN/p1/tools/check_pip_dependencies.py" \
  --output "$P4_RUN/pip-after"
"$P4_PY" -B "$P4_REPOS/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --check-lmcache --output "$P4_RUN/bootstrap"
"$P4_PY" -B "$P4_REPOS/vllm/tools/validate_npu_native.py" \
  --spawn --output "$P4_RUN/native-import.json"
"$P4_PY" -B "$P4_REPOS/LMCache/tools/p4_runtime_smoke.py" \
  --output "$P4_RUN/lmcache-import"
"$P4_PY" -B -m vllm.entrypoints.cli.main --help \
  2>&1 | tee "$P4_RUN/cli-help.log"
"$P4_PY" -B -m vllm.entrypoints.cli.main serve --help \
  2>&1 | tee "$P4_RUN/serve-help.log"
```

通过要求：两包版本为 `.layout1`、模式为 `editable`，导入路径位于**本次 checkout**
的 `build/__editable__.*`；配对 SHA、所有已跟踪产品 Python 文件、原生资源检查通过，
各 JSON 的 `passed=true`，所有命令退出码为 0。仅 pip 检查允许上述
`passed_with_waivers`。不应再安装或依赖 `vllm_ascend/lmcache_ascend` 独立插件包。
身份检查会归档原生库哈希，但不能单靠哈希或 `verify` 判定 ABI/NPU 可用。

需要复跑源码/主机子集时，使用**新布局**入口，不要运行旧 `design/p4/tools/verify.py`：

```bash
"$P4_PY" -B "$P4_REPOS/vllm/tools/run_layout_host_checks.py" \
  2>&1 | tee "$P4_RUN/source-host-vllm.log"
"$P4_PY" -B "$P4_REPOS/LMCache/tools/run_layout_host_checks.py" \
  2>&1 | tee "$P4_RUN/source-host-lmcache.log"
```

该入口会列出延后执行的 tensor 用例，即使环境有 torch 也不会自动补跑；不要将其计入
已通过项。它面向源码检查，不代替上面的安装路径检查或后面的设备/模型验证。

## 6. 空闲测试设备上的算子与缓存小样本

仅在已分配给本轮的空闲测试卡执行，不占用保留的基线服务设备。确认容器的设备映射，
探针使用**逻辑 NPU 0**；不要盲目将物理卡号写成逻辑号。

```bash
cd /tmp
"$P4_PY" -B "$P4_REPOS/vllm/tools/validate_npu_native.py" \
  --device-smoke --spawn --output "$P4_RUN/add-rmsnorm-bias-device.json"
"$P4_PY" -B "$P4_REPOS/LMCache/tools/p4_runtime_smoke.py" \
  --npu --output "$P4_RUN/lmcache-device"
```

要求主进程和 spawn 检查通过、`add_rms_norm_bias` 四项真实算子用例通过，
LMCache 导入及 4 KiB 注册主机内存/NPU 复制通过。未分配设备则记录“未执行”，
不要用不带设备开关的导入结果替代。不要手工指向旧包的 OPP vendor 目录绕过检查；
若报 `aclnnAddRmsNormBias`，保留探针报告和本次构建日志后排查。

## 7. 2P2D 验证、归档与回退

四个容器全部通过 §5，并在本轮测试设备上完成 §6，再按已验证的冻结 P4 配置启动新测试进程。启动脚本不应再次切到
旧 P1/P2/P3/P4 SHA，也不能在服务运行中切换 editable 源码。
路径变化只有工程布局，不应借机修改模型配置、缓存 YAML 或并发预算：

| 项目 | 本次要求 |
| --- | --- |
| 入口/示例 | 使用本次 Python 对应的 vLLM；proxy 位于 `$P4_REPOS/vllm/examples/disaggregated_prefill_v1/`，不再含根 `ascend/` |
| 模型 | 同一 GLM-5.2 checkpoint 与 W4A8 权重；DSA 双组、MTP 开启，C8 关闭；保留原 RoPE 配置 |
| 缓存连接器 | 正式 `LMCacheConnectorV1`，不再配置 `lmcache_ascend.*` 的 `kv_connector_module_path` |
| 拓扑/并发 | 4 节点×8 卡、2P2D；TP8/DP2 主验收与 TP4/DP4 回归分别执行；保留已验证的 `max_num_seqs=16` 等预算 |
| 功能 | 全部 P/D 实例加载、KV 初始化及预热成功；请求非空完成；MTP 草稿/接受计数、DSA 双组、CPU KV/RemoteFill、尾块保存、checkpoint/恢复正常 |
| 补充回归 | 离线 `LLM.generate`、无 LMCache 在线、流式/取消及既定故障/长稳；不能以单次 2P2D 启动代替 |

<a id="prefill-comparison"></a>

性能比较以**冻结 P4**为本次布局对照；P3 可作为历史参考。固定有效请求、预热与冷/热缓存
阶段，归档墙钟起止、TTFT/TPOT、吞吐、Running/Waiting、MTP 草稿与接受总数。
HTTP400、零输出、取消、未完成请求单列；不以 Waiting=0 判断 Prefill 变慢，不用请求延迟
求和当系统运行时长，也不平均各节点 MTP 百分比。详细采集规则见
[既有性能采集说明](intranet-validation-frozen-p4.md#prefill-comparison)。

每个容器归档整个 `$P4_RUN`，以及两仓根 `submodule-materials.json`、实际启动参数和
LMCache YAML（先脱敏）、checkpoint/配置/权重索引哈希、完整 P/D/proxy/客户端日志和
本轮 benchmark。`pair-installed.json` 含主机标识，不要拿一个容器的报告代表另外三个。

editable 通过不等于发行验收完成；普通 wheel、sdist 全新解包重建与 wheel 安装按
[新布局制品验证 §5](../native-layout/intranet-validation.md#wheel-sdist) 另行执行。
若在本页的同一个 shell 中继续，先映射该节所用变量，再执行其 §5 命令；无需重复克隆：

```bash
LAYOUT_ROOT="$P4_REPOS"
LAYOUT_PY="$P4_PY"
LAYOUT_DESIGN="$P4_DESIGN"
LAYOUT_REPORT="$P4_RUN"
```

失败时保留证据；回退至独立环境中的两仓 `p4-frozen-20261004` 配对安装，
不要在当前 editable checkout 上直接切回旧布局，也不要单仓回退。

后续仅修改已有 Python 文件时，检查链接和文件内容后用新进程复测；若新增/删除/移动文件、
改版本/依赖/打包规则/原生源码，必须重新安装（涉及 native 则重编译）。
本页只提供操作指令；本机文档检查不代表内网编译、NPU 或 2P2D 已通过。
