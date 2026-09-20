# P0 内网下一批执行手册：环境采集后的核对与回归

日期：2026-09-20。依据：已提供的 `env-P0.json`、`env-P1.json`、`env-D0.json`、`env-D1.json`，以及随后确认的版本、运行服务和“内网容器已完成 pytest 安装”的信息。

本批目标是补齐依赖、源码来源、量化摘要和 host 回归证据，不是重建或替换正在运行的服务。P0 仍处于进行中，尚未通过出口，不进入 P1。

本次修订已删除 pytest wheel 准备、哈希校验、隔离安装和专用 `PYTHONPATH` 注入步骤。直接复用容器内已安装的 pytest：第 3 节创建新批次，第 4 节刷新依赖证据，第 7 节完成现有测试工具预检后运行 host 回归；第 5–6 节尚未完成的模型和服务信息继续补齐。此前已核对且未发生变化的源码/模型证据可以引用原批次，但安装 pytest 后的包清单和依赖检查必须重新采集。

## 1. 已确认的基线与执行边界

首次采集的四份报告中，205 个 Python 包版本一致，已采集的模型元数据哈希一致。pytest 安装后的包数量、版本及节点间一致性尚未重新验证，不能继续以旧报告证明当前环境完全一致。以下是已记录的候选基线，不代表完整兼容性验收通过。

| 项目 | 已采集结果 | 本批处置 |
| --- | --- | --- |
| 硬件与部署 | 4 节点，每节点 8 张 910B3；目标 2P2D / TP8 / DP2 | 保留现有部署，进一步记录实际进程组和路由配置 |
| Python / 平台 | 3.11.14 / aarch64 / glibc 2.35 | 使用现有解释器，依赖材料按此平台准备 |
| CANN | 8.5.1 | 不更换；SDK、通信库和扩展 ABI 仍需核对 |
| torch | `2.9.0+cpu` | 保留；已通过 torch_npu 基础 NPU 探测，不因 `+cpu` 后缀而替换 |
| torch_npu | `2.9.0.post1+gitee7ba04` | 用户已确认保留，作为 P0 候选基线 |
| triton-ascend | `3.2.0.dev20260322` | 用户已确认保留，作为 P0 候选基线 |
| C++ ABI | `cxx11_abi=true` | 后续扩展构建必须使用一致的 torch 路径和 ABI |
| 编译工具 | CMake 4.3.1、GCC 11.4.0、Ninja 命令可用 | 仅确认工具存在；不据此宣称 SDK 构建兼容性已通过 |
| pytest | 用户确认内网容器已安装；具体版本和安装路径待记录 | 复用现有安装，不重新安装、升级或降级；参与测试的容器分别预检 |
| 模型 | `GlmMoeDsaForCausalLM`、`glm_moe_dsa`、78 层 | 与目标源码架构一致；不修改模型文件 |
| 模型元数据 | config、tokenizer config、generation config、量化描述哈希一致 | 完整权重分片的 revision/校验清单仍待补充 |
| 运行服务 | 用户确认卡上进程属于现有 GLM-5.2 / 2P2D 服务 | 作为待冻结的运行基线，不停止、不重启 |

采集时每卡 HBM 占用约 86%–90%。本批不启动第二套模型服务，不执行压测、故障注入、设备重置或进程清理。

其他边界：

- 内网人员或 CI 执行，不接入外部大模型/Agent，不自动上传报告。
- 不执行原始全量 requirements 安装，不升级或重装 torch、torch_npu、triton-ascend、OpenCV、numpy。
- 本批不包含任何安装或卸载命令，不修改现有 `site-packages`，不为 pytest 新建环境或追加测试专用 `PYTHONPATH`。
- 四节点核查必须在与服务相同的容器、用户和 Python 环境中进行。文件名代表人工指定的节点角色，不能单凭文件名证明实际进程拓扑。
- 本手册命令尚未在内网执行。逐节执行并检查结果，不要将全文一次性粘贴运行。

## 2. 已知问题与本批所需材料

### 2.1 已知问题

1. 当前源码的 `torch-npu==2.9.0` 不匹配现有 post1，`triton-ascend==3.2.0` 不匹配现有 dev 版本。用户已确认保留安装；制品来源、哈希和后续兼容性核对完成后，再同步修正源码声明、P0 约束及检查。不要直接编辑历史采集报告，把 `needs_review` 改成通过。
2. 当前 OpenCV 为 `4.11.0.86`，符合 Ascend 声明，但不符合 vLLM 的 `>=4.13.0`。本批先取得已安装包的实际依赖声明，不升级 OpenCV/numpy，也不仅删除一条 requirements 来掩盖问题。
3. 旧包清单未发现 pytest、iniconfig、pluggy；用户已确认随后在内网容器中安装 pytest。该缺失记录属于历史状态，本批以第 7 节的实际版本、依赖和导入预检为准，不推定所有节点已经安装了相同版本。
4. 旧包清单未发现 LMCache、LMCache-Ascend、mooncake-transfer-engine 的发行包记录。需要用本批证据区分未安装、源码路径加载和其他传输实现，不能直接判定功能缺失。
5. 旧包清单中 `build` 未安装；它影响 `python -m build`，不等于必须现在补装，也不能据此判定 `pip wheel` 不可用。

原报告中的代理环境变量均未设置，但不能据此排除 pip/Git 等工具级代理配置。本批提供的命令不需要联网或获取新的测试工具；后续若确需补齐其他构建材料，必须由内网维护者先确认批准来源、代理和企业 CA，不能默认访问公网或关闭 TLS 校验。

### 2.2 带入内网的材料

- 本 Markdown 文件。
- 已交付的 [tools/run_host_checks.py](tools/run_host_checks.py)。它不会安装依赖、构建框架或启动模型。
- 当前四仓源码及已核对的 P0 独立修复；四仓目录名保持为 `vllm`、`vllm-ascend`、`LMCache`、`LMCache-Ascend`。不要覆盖现有服务正在使用的源码。
- 供人工核对的 [源码清单](baseline/source-manifest.json)、[修复清单](changes/change-manifest.json) 和 [子模块清单](review/submodule-materials.json)。
- 与服务相同的 Python 环境，以及用户已安装的 pytest。无需另外带入 pytest、iniconfig、pluggy wheel；现有依赖是否齐全由第 7 节预检。

仅复制本 Markdown 可以执行环境取证，但不能替代 host 检查所需的脚本和四仓测试源码。缺少材料时记录缺件，不自动 clone、pull、apply patch 或切换分支。

## 3. 每节点初始化本批次目录

在 Bash 中执行。默认内网工作区为 `/workspace/zzj`；如实际不同，修改 `P0_WORKSPACE`。将 `P0_ROLE` 分别改为 `P0`、`P1`、`D0`、`D1`。

同一节点后续命令在同一个 shell 中继续执行。重新运行时创建新批次目录，不覆盖前一次证据。若重新打开 shell，重新执行本节并继续生成新批次即可。

```bash
P0_WORKSPACE=/workspace/zzj
P0_ROLE=P0

case "$P0_ROLE" in
    P0|P1|D0|D1) ;;
    *) printf '节点角色必须为 P0/P1/D0/D1\n' >&2; exit 2 ;;
esac

cd "$P0_WORKSPACE" || exit 1
umask 077
mkdir -p "$P0_WORKSPACE/p0-intranet-results" || exit 1
P0_NEXT_DIR=$(mktemp -d "$P0_WORKSPACE/p0-intranet-results/next-${P0_ROLE}.XXXXXX") || exit 1

p0_capture() {
    local p0_name="$1"
    local p0_rc
    shift
    if [[ -e "$P0_NEXT_DIR/$p0_name" ||
          -e "$P0_NEXT_DIR/$p0_name.stderr" ||
          -e "$P0_NEXT_DIR/$p0_name.rc" ]]; then
        printf '拒绝覆盖 %s；请使用新批次目录\n' "$p0_name" >&2
        return 2
    fi
    if "$@" >"$P0_NEXT_DIR/$p0_name" \
             2>"$P0_NEXT_DIR/$p0_name.stderr"; then
        p0_rc=0
    else
        p0_rc=$?
    fi
    printf '%s\n' "$p0_rc" >"$P0_NEXT_DIR/$p0_name.rc"
    printf '%s: rc=%s\n' "$p0_name" "$p0_rc"
}

p0_succeeded() {
    local p0_rc
    [[ -f "$P0_NEXT_DIR/$1.rc" ]] || return 1
    IFS= read -r p0_rc <"$P0_NEXT_DIR/$1.rc" || return 1
    [[ "$p0_rc" == 0 ]]
}

printf '节点角色：%s\n结果目录：%s\n' "$P0_ROLE" "$P0_NEXT_DIR"
```

`p0_capture` 会分别保存标准输出、标准错误及真实命令退出码。它在采集到失败后仍允许继续收集其他证据，因此 shell 继续运行不代表检查通过。以对应 `.rc` 文件和报告内容判定结果；后文 `p0_succeeded` 用于检查记录的退出码。

## 4. 第一批：依赖、解释器、源码与模块来源

执行节点：四节点分别执行。性质：只读取证，不安装、不构建、不加载模型。

使用新批次保存 pytest 安装后的包清单和依赖结果，不覆盖原始 `env-*.json`。比较 `pip-list.json`、`pip-inspect.json` 与首次报告，重点确认 torch、torch_npu、triton-ascend、numpy、OpenCV 等运行依赖没有被意外替换；若发现变化，先核对，不自动回退安装。

```bash
p0_capture pip-list.json \
    python -B -m pip --disable-pip-version-check list --format=json

p0_capture pip-check.txt \
    python -B -m pip --disable-pip-version-check check

p0_capture pip-inspect.json \
    python -B -m pip --disable-pip-version-check inspect

# 当前 Ascend 构建脚本存在硬编码 python3 的查询，排除环境混用。
p0_capture python-location.txt \
    python -B -c 'import sys; from importlib.metadata import distribution; print(sys.executable); print(distribution("torch-npu").locate_file(""))'

p0_capture python3-location.txt \
    python3 -B -c 'import sys; from importlib.metadata import distribution; print(sys.executable); print(distribution("torch-npu").locate_file(""))'

for p0_repo in vllm vllm-ascend LMCache LMCache-Ascend; do
    p0_capture "${p0_repo}-revision.txt" \
        git -C "$P0_WORKSPACE/$p0_repo" rev-parse HEAD 'HEAD^{tree}'

    p0_capture "${p0_repo}-status.txt" \
        git -C "$P0_WORKSPACE/$p0_repo" status --short

    p0_capture "${p0_repo}-submodules.txt" \
        git -C "$P0_WORKSPACE/$p0_repo" submodule status --recursive
done

# 只查询顶层模块位置，不导入框架或执行框架初始化。
p0_capture module-origins.json python -B - <<'PY'
import importlib.util
import json

result = {}
for name in ("vllm", "vllm_ascend", "lmcache", "lmcache_ascend", "mooncake", "triton"):
    spec = importlib.util.find_spec(name)
    result[name] = None if spec is None else {
        "origin": spec.origin,
        "search_locations": list(spec.submodule_search_locations or []),
    }
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
```

检查和停止条件：

- `pip-check.txt.rc` 非 0：保留问题原文，禁止自动补包或升级。`pip check` 只核对已安装包声明，不能验证源码未安装依赖、扩展 ABI 或目标模型行为。
- `pip-inspect.json.rc` 非 0：查看对应 `.stderr`，不得将空文件当作有效报告。
- `python` 和 `python3` 指向不同的 torch_npu 安装目录：暂停后续正式构建，先统一构建解释器；不修改现有服务安装。
- 四仓目录缺失、commit 不符或存在未说明的改动：记录差异，先核对源码来源；不执行 reset、强制覆盖或重复应用补丁。HEAD/tree 不包含未提交修改，存在改动时还需留存已审核补丁及其哈希。
- 子模块输出前缀 `-` 表示未初始化，`+` 表示与父仓记录不一致，`U` 表示冲突；这些状态均不能放行正式构建。本批只记录，不自动下载。
- `module-origins.json` 是本次 shell 上下文的查询结果，不是运行中服务的实际加载证明；必须与第 6 节的服务启动配置对照。模块来源为 `null` 或 namespace 目录也需要解释。

子模块的固定提交如下；不能用分支最新 HEAD 代替：

| 主仓 / 子模块 | 固定 commit |
| --- | --- |
| vllm-ascend / csrc/third_party/catlass | `716fd7baa7fb7f6cac0488bb628fd1dd0e875641` |
| LMCache-Ascend / third_party/kvcache-ops | `9f18d2339bc58a43429f7d5bdaef1628c820eff5` |

## 5. 第二批：量化摘要，不修改模型

执行节点：先在 P0 节点执行一次。其他节点已采集元数据哈希一致；如后续模型文件发生变化，需要按新批次重新采集和对比。

`quantization_config_present=false` 不代表模型未量化。当前分支支持从独立的 `quant_model_description.json` 识别 Ascend ModelSlim 路径。以下命令使用报告中已安装的 `ijson` 流式读取量化描述，不加载权重或执行 remote code。

```bash
P0_MODEL_DIR=/workspace/models/GLM-5.2-w4a8c8-0723

p0_capture model-quant-summary.json python -B - "$P0_MODEL_DIR" <<'PY'
from collections import Counter
from pathlib import Path
import json
import sys
import ijson

root = Path(sys.argv[1])
with (root / "config.json").open() as stream:
    config = json.load(stream)

fields = (
    "architectures", "model_type", "num_hidden_layers",
    "num_nextn_predict_layers", "indexer_types",
    "torch_dtype", "dtype", "quantization_config",
)
result = {"config": {k: config[k] for k in fields if k in config}}

weight_types = Counter()
quant_types = Counter()
with (root / "quant_model_description.json").open("rb") as stream:
    for key, value in ijson.kvitems(stream, ""):
        if isinstance(value, str):
            if key.endswith((".weight", ".weight_packed")):
                weight_types[value] += 1
            if key.endswith("quant_type"):
                quant_types[value] += 1
            if key in ("fa_quant_type", "indexer_quant_type"):
                result[key] = value

result["weight_type_counts"] = dict(weight_types)
result["quant_type_counts"] = dict(quant_types)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
```

检查 `model-quant-summary.json.rc` 和摘要内容。计数为空或缺少字段时，反馈格式问题，不直接认定未量化或量化验证通过。该摘要不能替代逐层回退规则、MTP 权重、C8 data/scale 的 dtype/shape/stride/布局和缓存 round trip 检查。

完整权重身份优先复用已有制品 revision、分片清单和校验清单；本批不在繁忙服务节点上全量读取大权重计算哈希。不要为适配脚本而编辑 config 或量化描述文件。

## 6. 人工冻结现有服务与二进制制品来源

由内网维护者保存现有服务启动配置、部署配置和必要日志，不能只用发行包版本号证明它与当前四仓源码一致。不要停止服务来获取这些信息，也不要外发完整环境变量或带凭据的启动命令。

使用代号形成以下摘要，未知项写“待确认”：

```text
节点角色：P0 / P1 / D0 / D1
服务代码：各组件 commit、补丁 hash 或原始 wheel 制品编号
运行来源：site-packages / editable / 源码挂载；与 module-origins 是否一致
模型身份：revision、现有权重分片校验清单编号
并行配置：TP / DP / EP / PP / CP；P、D 副本和路由关系
量化配置：quantization、kv_cache_dtype、MTP 配置
缓存配置：KV connector 类型、LMCache 配置来源、实际启用的后端
传输配置：HCCL 集合通信、缓存传输通道、Mooncake/RemoteFill 是否启用
网络策略：批准的材料来源、代理/CA 是否核对、外部大模型接入禁止策略
```

另外补充 `torch_npu 2.9.0.post1+gitee7ba04` 和 `triton-ascend 3.2.0.dev20260322` 原始制品的来源、完整文件名、SHA-256、构建 commit/补丁或内部制品编号。若原始 wheel 暂时找不到，明确记录缺失，不能用版本字符串或已安装文件的 `RECORD` 代替原始 wheel 哈希。

未发现 LMCache/Mooncake 发行包记录时，优先确认现有服务是否使用源码路径、不同容器或不同缓存实现。不能据包清单缺项直接安装原始 LMCache requirements 中的 CUDA 依赖。

## 7. 第三批：复用已安装的 pytest，运行 host 回归

执行节点：先在 P0 节点执行。若四节点源码、解释器或运行依赖不一致，按不同 profile 分别重跑；host 结果不能替代四节点 NPU/通信验收。

### 7.1 核对现有测试工具，不重新安装

用户已经完成 pytest 安装，本节只读取版本、检查 pytest 当前声明的必要依赖并确认导入和命令可用。使用 `python -m pytest`，避免 PATH 中的 `pytest` 命令属于另一套 Python 环境。

前一次本机 host 检查使用 pytest 9.0.2，仅作为结果对照信息，不要求现有内网 pytest 必须为该版本。实际安装版本由以下报告记录，不因版本不同就升级或降级。

当前预检会导入 pytest、numpy 和 packaging，不导入 torch/torch_npu，也不执行模型代码；torch 的安装身份通过 distribution 元数据读取。pytest 自身依赖按当前 Python 平台和非 extra 条件检查，不能用全环境 `pip check` 中尚待修复的框架声明冲突代替这一局部检查。

```bash
p0_capture host-test-precheck.json \
    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS= PYTEST_PLUGINS= \
    python -B - <<'PY'
from importlib import metadata
import json
import sys
import numpy
import pytest
from packaging.requirements import Requirement

dependencies = []
for raw in metadata.requires("pytest") or []:
    requirement = Requirement(raw)
    if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
        continue
    try:
        actual = metadata.version(requirement.name)
    except metadata.PackageNotFoundError:
        actual = None
    satisfied = actual is not None and requirement.specifier.contains(
        actual, prereleases=True
    )
    dependencies.append({
        "requirement": raw, "actual": actual, "satisfied": satisfied,
    })

result = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "pytest": {
        "distribution_version": metadata.version("pytest"),
        "imported_version": pytest.__version__,
        "module_file": pytest.__file__,
    },
    "numpy": {"version": numpy.__version__, "module_file": numpy.__file__},
    "torch_distribution_version": metadata.version("torch"),
    "pytest_dependencies": dependencies,
}
print(json.dumps(result, ensure_ascii=False, indent=2))
versions_agree = result["pytest"]["distribution_version"] == pytest.__version__
raise SystemExit(0 if versions_agree and all(x["satisfied"] for x in dependencies) else 1)
PY

p0_capture pytest-version.txt \
    env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTEST_ADDOPTS= PYTEST_PLUGINS= \
    python -B -m pytest --version
```

继续条件：

- `host-test-precheck.json.rc` 和 `pytest-version.txt.rc` 均为 0，pytest 的依赖检查全部满足。
- Python、pytest、numpy 的实际来源符合预期；pytest 的 distribution 版本与实际导入版本一致。
- 若曾按旧版手册手工设置测试专用 `PYTHONPATH`，先由内网人员核对，避免仍从旧测试目录加载 pytest；不要盲目清空服务/CANN 所需的环境变量。本版不再追加任何测试目录。
- 若缺失 pytest、numpy、packaging、torch 元数据，或遇到导入错误、版本冲突，预检将非 0 退出；JSON 可能为空，查看对应 `.stderr` 后停止该节点 host 检查。不自动安装或回退，其他只读取证可以继续。
- 全环境 `pip check` 的已知框架声明问题仍需单独闭环；局部预检通过不是正式构建放行。

### 7.2 直接执行固定 host 子集

复用上述同一 `python` 解释器及其已安装的 pytest，不调用全局 `pytest` 可执行文件，不修改 `PYTHONPATH`。输出目录仍使用本批次的新目录，拒绝覆盖旧结果。

```bash
if p0_succeeded host-test-precheck.json && \
   p0_succeeded pytest-version.txt && \
   [[ -f "$P0_WORKSPACE/design/p0/tools/run_host_checks.py" && \
      ! -e "$P0_NEXT_DIR/host-baseline" ]]; then
    p0_capture host-run.txt \
        python -B "$P0_WORKSPACE/design/p0/tools/run_host_checks.py" \
        --workspace "$P0_WORKSPACE" \
        --output "$P0_NEXT_DIR/host-baseline"
else
    printf '停止：现有 pytest 预检未通过、检查脚本缺失或结果目录已存在\n' >&2
fi
```

检查脚本在各仓独立进程中运行 AST/mock 子集，禁用项目 conftest 与 pytest 插件自动加载，不执行 NPU 模型或通信测试。初始化检查会间接导入现有 torch；不能将其表述为完全没有框架导入。

检查以下产物：

- `host-test-precheck.json`、`pytest-version.txt` 及各自的 `.rc` / `.stderr`。
- `host-run.txt.rc`、`host-run.txt`、`host-run.txt.stderr`。
- `host-baseline/summary.json`。
- `host-baseline/*.log`、`host-baseline/*.xml`。

当前固定子集的本机记录为 vLLM 16 项、vllm-ascend 31 项、LMCache 39 项、LMCache-Ascend 26 项，合计 112 项；此前 107 项通过，5 项因本机没有 torch 失败。内网本批目标是固定子集全部通过，并关闭这 5 项失败。

判定要求：四仓均有报告，退出码均为 0，`failures=0`、`errors=0`、`skipped=0`，测试数量与已核对的源码一致。脚本退出码为 0 并不单独证明没有跳过测试，因此还必须检查 XML/summary 中的 skipped。数量变化、缺失报告或失败时先解释原因，不减少用例、放宽断言或添加跳过来通过门槛。

## 8. 本批之后的修复顺序与构建门槛

以下是下一批源码/构建工作的顺序，不是在运行容器中立即执行的安装命令。本手册没有修改现有依赖声明或授权替换正在运行的服务。

1. 核对制品来源、哈希后，将相关 Ascend 构建/运行声明、P0 约束和检查同步对齐到已确认的候选版本：`torch-npu==2.9.0.post1+gitee7ba04`、`triton-ascend==3.2.0.dev20260322`。`torch==2.9.0` 可匹配现有 `2.9.0+cpu`，不为此更换 torch。候选版本仍需后续兼容性验收，不修改历史原始基线。
2. 结合 `pip inspect` 和文本入口实际依赖，独立处理 OpenCV 声明冲突。不能只改上限/下限或删除 requirements 一行而不验证导入路径。
3. 明确 LMCache 实际加载方式、缓存后端、Mooncake/RemoteFill 通道，确定必要 Python 包和 host 扩展。当前 `NO_CUDA_EXT=1` 会跳过全部扩展；若目标路径需要 `native_storage_ops` 或 `lmcache_mooncake` 等，必须先准备相应 host-only 构建路径，不能把空扩展 wheel 当作缓存能力齐备。
4. 补齐固定提交的子模块、CANN 头文件/编译工具、Python 开发头文件、必要系统库和通信 SDK。CANN 8.5.1 的当前 LMCache-Ascend 构建路径会选择 HIXL/hcomm one-sided；分别核对所需库和 HCCL 集合通信，不因包清单存在 `hccl` 就认定 SDK/ABI 已通过。
5. 依赖冲突、源码来源和材料缺件闭环后，在与运行服务分离的构建目录/容器中构建四仓。保留 torch 身份、非隔离构建、禁止隐式依赖升级；新产物完成扩展加载和回归后，再安排明确的替换窗口。

本批的任一通过项都不等于 P0 出口通过。GLM-5.2 的离线/在线、W4A8C8、MTP、DSA、CPU KV、跨实例缓存、P/D、RemoteFill、checkpoint/恢复及性能基线仍需按原支持矩阵逐项验收。

## 9. 反馈清单与信息边界

先反馈以下经人工审核、允许外发的材料；路径、内部制品地址可用代号替换：

- [ ] pytest 安装后四节点的包清单差异、`pip-check.txt` 问题列表和退出码。
- [ ] 与问题相关的 `pip-inspect.json` 依赖条目及来源结论；不是未经审查的完整原件。
- [ ] `python`/`python3` 是否一致，模块实际来源与服务配置是否对应。
- [ ] 四仓 commit/tree、工作区差异说明、子模块状态。
- [ ] 候选 torch_npu、triton-ascend 制品来源/哈希是否已核对。
- [ ] `model-quant-summary.json` 和现有权重制品身份结论。
- [ ] 第 6 节的服务配置摘要，特别是 LMCache/Mooncake/RemoteFill 是否实际启用。
- [ ] 实际 pytest 版本、模块来源、`host-test-precheck.json` 和 `pytest-version.txt` 的结果；是否与前次测试工具版本不同。
- [ ] host `summary.json`、退出码；如失败，附必要的最小错误片段。
- [ ] 当前未完成项和停止原因，标明 `FAIL` 或 `PENDING`，不得记作通过。

本批原件写入内网工作区的 `p0-intranet-results/next-<角色>.<随机后缀>/`，不覆盖前一批 `env-*.json`。完整日志、trace、业务输入输出、凭据、敏感地址和未审核制品来源留在内网，不自动打包上传或提交 Git。当前用于分析的 `design/p0/p0-intranet-results/` 含原始报告，不能因方便交接而默认将其加入版本控制。

## 10. 参考

以下链接供人工查阅；上述执行命令不访问这些网站。

- [原 P0 内网执行手册](intranet-runbook.md)。
- [P0 当前支持矩阵](support-matrix.json)。
- [Python 版本匹配规范](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-matching)。
- [pip check 的检查范围](https://pip.pypa.io/en/stable/cli/pip_check/)。
