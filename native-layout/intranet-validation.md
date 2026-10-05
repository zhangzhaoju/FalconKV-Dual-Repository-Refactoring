# 根 ascend 目录整改：内网安装与回归

适用配对：`refactor/native-layout`，vLLM `0.18.0+ascend.layout1`、LMCache
`0.4.3+ascend.layout1`。精确 SHA 以同目录 [baseline/pair.json](baseline/pair.json) 为准。
该配对已完成本机源码和主机检查，原生构建、ABI 与 2P2D 留待本步骤验证。

沿用 `/workspace/zzj/p1-repos` 的逐容器 strict editable 安装、旧包处理和完整安装后检查，
优先按 [P4 editable 安装与内网验证指南](../p4/intranet-validation.md) 执行。
本页保留新目录方案及 wheel/sdist 制品验证；两页变量名不同，切换前须重新初始化，
或使用 P4 指南末尾提供的 §5 变量映射，不能直接交叉复制命令块。

<a id="publish"></a>

## 1. 发布和准备独立环境

目录整改交付时只创建了本地提交，未自动推送。若配对版本尚未发布，
由源码管理人员在本机工作区执行以下非强制推送；不要在内网重造提交：

```bash
cd /home/zhangzhaoju/Learn/code/falconkv || exit 1
git -C p1-repos/vllm push -u origin refactor/native-layout
git -C p1-repos/LMCache push -u origin refactor/native-layout
git -C p1-repos/vllm push origin refs/tags/p4-frozen-20261004
git -C p1-repos/LMCache push origin refs/tags/p4-frozen-20261004
```

Actions 保持开启。两仓工作流需要同级 checkout 配对分支；首个仓库推送时另一个分支尚未
发布可能失败，两仓发布后重跑。若另一仓为私有仓，须预先配置跨仓只读访问，不把凭据写入文件。

使用有相同 CANN/torch 依赖但未安装旧框架的专用验证容器，保留正在服务的 P4 容器。
`--isolated-env` 不会忽略旧包冲突，也不会自动卸载旧版本。每个运行容器都必须安装两仓，
不能仅在一个节点安装 LMCache。依赖基线不变：910B3、Python 3.11/aarch64、CANN 8.5.1、
torch 2.9.0、torch-npu 2.9.0.post2、transformers 5.2.0、triton-ascend 3.2.0.dev20260322。

将本目录与已有 `design/p1/tools/check_pip_dependencies.py` 一并上传到
`/workspace/zzj/design`；后者仅复用三条已批准的 pip 元数据豁免，不复用旧阶段的目录迁移工具。

## 2. 新 checkout 和固定源码身份

建议用新目录 `/workspace/zzj/layout-repos`，不要在有运行中 editable 服务的旧 `p1-repos`
里直接切分支。若自动化仍固定旧路径，需要同步调整 checkout、安装和 proxy 脚本路径。
两仓保持同级，名字分别为 `vllm`、`LMCache`。

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
LAYOUT_ROOT=/workspace/zzj/layout-repos
LAYOUT_PY=/usr/local/python3.11.14/bin/python
LAYOUT_DESIGN=/workspace/zzj/design
test ! -e "$LAYOUT_ROOT/vllm"
test ! -e "$LAYOUT_ROOT/LMCache"
mkdir -p -- "$LAYOUT_ROOT"
git clone --branch refactor/native-layout --single-branch \
  git@github.com:zhangzhaoju/vllm-dual.git "$LAYOUT_ROOT/vllm"
git clone --branch refactor/native-layout --single-branch \
  git@github.com:zhangzhaoju/LMCache-dual.git "$LAYOUT_ROOT/LMCache"

"$LAYOUT_PY" -B - "$LAYOUT_ROOT" "$LAYOUT_DESIGN/native-layout/baseline/pair.json" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
pair = json.loads(Path(sys.argv[2]).read_text())
for name, expected in pair["repositories"].items():
    actual = subprocess.check_output(
        ["git", "-C", str(root / name), "rev-parse", "HEAD"], text=True
    ).strip()
    print(name, actual)
    if actual != expected["commit"]:
        raise SystemExit("STOP: branch tip differs from the reviewed pair")
PY
LAYOUT_REPORT=$(mktemp -d "$LAYOUT_ROOT/check.XXXXXXXX")
printf '%s\n' "$LAYOUT_REPORT"
```

以下小节在同一个 shell 中继续。Git 获取可以使用既定 proxy，不配置或连接大模型服务。
若 SHA 不符，应先确认发布版本；不要强制覆盖本地改动或跳过配对检查。

## 3. 材料与依赖门槛

```bash
git -C "$LAYOUT_ROOT/vllm" submodule update --init -- csrc/third_party/catlass
git -C "$LAYOUT_ROOT/LMCache" submodule update --init -- third_party/kvcache-ops
"$LAYOUT_PY" -B "$LAYOUT_ROOT/vllm/p1_dev.py" materials \
  --from-submodule "$LAYOUT_ROOT/vllm/csrc/third_party/catlass"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/LMCache/p1_dev.py" materials \
  --from-submodule "$LAYOUT_ROOT/LMCache/third_party/kvcache-ops"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/vllm/tools/check_native_layout.py"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/LMCache/tools/check_native_layout.py"

export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export SOC_VERSION=ascend910b3
"$LAYOUT_PY" -B "$LAYOUT_ROOT/vllm/p1_dev.py" doctor \
  --output "$LAYOUT_REPORT/doctor-vllm.json"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/LMCache/p1_dev.py" doctor \
  --output "$LAYOUT_REPORT/doctor-lmcache.json"
"$LAYOUT_PY" -B "$LAYOUT_DESIGN/p1/tools/check_pip_dependencies.py" \
  --output "$LAYOUT_REPORT/pip-before"
```

材料清单变为根 `submodule-materials.json`。也可用已核验的本地 Git 材料路径代替下载，
但不能把未核验文件复制进去。固定提交仍为 CATLASS
`716fd7baa7fb7f6cac0488bb628fd1dd0e875641`、kvcache-ops
`9f18d2339bc58a43429f7d5bdaef1628c820eff5`。
以上任一门槛失败就停止，不隐式升级依赖，不用 `--source-only` 替代真实材料检查。

## 4. strict editable 调测安装

两仓都需要重新构建/安装；不能沿用 P4 `.so` 或旧 editable 链接树证明新布局通过。
脚本仍叫 `p1_dev.py`，内部已适配新目录、配对版本与独立构建 staging。

```bash
cd /tmp || exit 1
"$LAYOUT_PY" -B "$LAYOUT_ROOT/vllm/p1_dev.py" editable --isolated-env \
  --output "$LAYOUT_REPORT/editable-vllm"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/LMCache/p1_dev.py" editable --isolated-env \
  --output "$LAYOUT_REPORT/editable-lmcache"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/vllm/p1_dev.py" verify --mode editable \
  --output "$LAYOUT_REPORT/installed-vllm.json"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/LMCache/p1_dev.py" verify --mode editable \
  --output "$LAYOUT_REPORT/installed-lmcache.json"
"$LAYOUT_PY" -B "$LAYOUT_DESIGN/p1/tools/check_pip_dependencies.py" \
  --output "$LAYOUT_REPORT/pip-after"

"$LAYOUT_PY" -B "$LAYOUT_ROOT/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --check-lmcache --output "$LAYOUT_REPORT/bootstrap"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/vllm/tools/validate_npu_native.py" \
  --spawn --output "$LAYOUT_REPORT/native-import.json"
"$LAYOUT_PY" -B "$LAYOUT_ROOT/LMCache/tools/p4_runtime_smoke.py" \
  --output "$LAYOUT_REPORT/lmcache-import"
```

安装时不自动访问包索引、不重装依赖。保留每次生成的 `build/p1-native/run-*`；
strict editable 的原生库可能链接到该目录，不能清理后继续运行服务。
上述 `verify` 只检验文件/安装路径，后续 import 检查也不能代替 NPU 模型验收。
在已分配的空闲测试设备上，再分别加 `--device-smoke`、`--npu` 执行设备小样本；
不要在有基线业务的卡上自动执行。每个新容器分别归档配对 SHA 与安装/导入检查结果。

<a id="wheel-sdist"></a>

## 5. wheel 与 sdist 出口

在一个专用构建容器执行，不要求四台重复生成相同制品；四台安装后仍须各自验路径和 ABI。
下面产出普通 wheel 和带真实固定材料的 sdist，并从全新解包目录重建：

```bash
for LAYOUT_NAME in vllm LMCache; do
  "$LAYOUT_PY" -B "$LAYOUT_ROOT/$LAYOUT_NAME/p1_dev.py" build \
    --output "$LAYOUT_REPORT/wheel-$LAYOUT_NAME"
  "$LAYOUT_PY" -B -m build --sdist --no-isolation \
    --outdir "$LAYOUT_REPORT/sdist-$LAYOUT_NAME" "$LAYOUT_ROOT/$LAYOUT_NAME"
done
mkdir -p -- "$LAYOUT_REPORT/unpacked-vllm" "$LAYOUT_REPORT/unpacked-LMCache"
tar -xzf "$LAYOUT_REPORT/sdist-vllm/vllm-0.18.0+ascend.layout1.tar.gz" \
  -C "$LAYOUT_REPORT/unpacked-vllm"
tar -xzf "$LAYOUT_REPORT/sdist-LMCache/lmcache-0.4.3+ascend.layout1.tar.gz" \
  -C "$LAYOUT_REPORT/unpacked-LMCache"
LAYOUT_V_SRC="$LAYOUT_REPORT/unpacked-vllm/vllm-0.18.0+ascend.layout1"
LAYOUT_L_SRC="$LAYOUT_REPORT/unpacked-LMCache/lmcache-0.4.3+ascend.layout1"
"$LAYOUT_PY" -B "$LAYOUT_V_SRC/tools/check_native_layout.py"
"$LAYOUT_PY" -B "$LAYOUT_L_SRC/tools/check_native_layout.py"
"$LAYOUT_PY" -B "$LAYOUT_V_SRC/p1_dev.py" build --output "$LAYOUT_REPORT/rebuild-vllm"
"$LAYOUT_PY" -B "$LAYOUT_L_SRC/p1_dev.py" build --output "$LAYOUT_REPORT/rebuild-LMCache"
```

在另一个尚未启动服务的专用环境，使用相同源码的 `p1_dev.py install --isolated-env
--wheel <本次明确选定的wheel路径> --output <新目录>` 安装配对 wheel，随后执行
`p1_dev.py verify --mode wheel` 及第 4 节的导入检查。归档 `artifact.json`、完整构建日志、
材料清单与两轮 wheel/sdist 哈希。不要用宽泛 glob 误装旧轮次制品；重建成功不代表二进制
逐字节可重复，也不以两个 wheel 的 SHA 相同作为通过条件。

## 6. 2P2D 回归与结果判定

沿用冻结 P4 的模型、量化、并行和服务参数，仅调整工程/proxy 路径。
proxy 现在位于 `vllm/examples/disaggregated_prefill_v1/`，不存在 `vllm/ascend/examples/`。
保持 GLM-5.2、DSA 双组/MTP 开启、C8 关闭；不要改参数来掩盖目录回归。

- 必测离线/在线、无 LMCache 与配对缓存模式、TP8/DP2 和 TP4/DP4。
- 检查 Prefill/Decode 正常输出、缓存存取/卸载、跨实例 P/D、RemoteFill、checkpoint/恢复，
  以及此前 MTP、RoPE、AddRmsNormBias、OPP 注册与尾块保存修复未退化。
- 固定负载与冷/热阶段，使用共同墙钟统计有效请求吞吐、TTFT、TPOT、MTP 接收率。
  HTTP400、零输出、取消/未完成请求必须单独统计，不能算作成功；不要用延迟求和当压测时长。
- 每个容器归档 checkout SHA、安装路径、材料/制品哈希、最终启动参数和完整 P/D/proxy 日志。

新布局验收完成前，不移动冻结 P4，不标记“全部验收通过”。出现回归应先保留本轮日志和
构建产物，再回到独立 P4 环境；必须两仓配对回退，不在服务运行时切换 editable 源码。
旧 `design/p0` 至 `design/p4` 的裁剪/导出脚本属于历史布局，不能在新分支直接重跑。
