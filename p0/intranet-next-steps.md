# P0 内网下一批执行手册：环境采集后的核对与回归

日期：2026-09-20。依据：已提供的 `env-P0.json`、`env-P1.json`、`env-D0.json`、`env-D1.json`，以及随后确认的版本和运行服务信息。

本批目标是补齐依赖、源码来源、量化摘要和 host 回归证据，不是重建或替换正在运行的服务。P0 仍处于进行中，尚未通过出口，不进入 P1。

## 1. 已确认的基线与执行边界

四份报告的 205 个 Python 包版本一致，已采集的模型元数据哈希一致。以下是候选基线，不代表完整兼容性验收通过。

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
| 模型 | `GlmMoeDsaForCausalLM`、`glm_moe_dsa`、78 层 | 与目标源码架构一致；不修改模型文件 |
| 模型元数据 | config、tokenizer config、generation config、量化描述哈希一致 | 完整权重分片的 revision/校验清单仍待补充 |
| 运行服务 | 用户确认卡上进程属于现有 GLM-5.2 / 2P2D 服务 | 作为待冻结的运行基线，不停止、不重启 |

采集时每卡 HBM 占用约 86%–90%。本批不启动第二套模型服务，不执行压测、故障注入、设备重置或进程清理。

其他边界：

- 内网人员或 CI 执行，不接入外部大模型/Agent，不自动上传报告。
- 不执行原始全量 requirements 安装，不升级或重装 torch、torch_npu、triton-ascend、OpenCV、numpy。
- 唯一的安装步骤是后文将三个已审核的测试工具 wheel 放入本批次独立目录；不修改现有 `site-packages`。
- 四节点核查必须在与服务相同的容器、用户和 Python 环境中进行。文件名代表人工指定的节点角色，不能单凭文件名证明实际进程拓扑。
- 本手册命令尚未在内网执行。逐节执行并检查结果，不要将全文一次性粘贴运行。

## 2. 已知问题与本批所需材料

### 2.1 已知问题

1. 当前源码的 `torch-npu==2.9.0` 不匹配现有 post1，`triton-ascend==3.2.0` 不匹配现有 dev 版本。用户已确认保留安装；制品来源、哈希和后续兼容性核对完成后，再同步修正源码声明、P0 约束及检查。不要直接编辑历史采集报告，把 `needs_review` 改成通过。
2. 当前 OpenCV 为 `4.11.0.86`，符合 Ascend 声明，但不符合 vLLM 的 `>=4.13.0`。本批先取得已安装包的实际依赖声明，不升级 OpenCV/numpy，也不仅删除一条 requirements 来掩盖问题。
3. 包清单中未发现 pytest、iniconfig、pluggy。直接运行 host 检查会失败，先按第 7 节准备测试工具。
4. 包清单中未发现 LMCache、LMCache-Ascend、mooncake-transfer-engine 的发行包记录。需要区分未安装、源码路径加载和其他传输实现，不能直接判定功能缺失。
5. `build` 未安装；它影响 `python -m build`，不等于必须现在补装，也不能据此判定 `pip wheel` 不可用。

报告中的代理环境变量均未设置，但不能据此排除 pip/Git 等工具级代理配置。本批提供的命令不需要联网；获取 wheel 等材料必须由内网维护者先确认批准来源、代理和企业 CA，不能默认访问公网或关闭 TLS 校验。

### 2.2 带入内网的材料

- 本 Markdown 文件。
- 已交付的 [tools/run_host_checks.py](tools/run_host_checks.py)。它不会安装依赖、构建框架或启动模型。
- 当前四仓源码及已核对的 P0 独立修复；四仓目录名保持为 `vllm`、`vllm-ascend`、`LMCache`、`LMCache-Ascend`。不要覆盖现有服务正在使用的源码。
- 供人工核对的 [源码清单](baseline/source-manifest.json)、[修复清单](changes/change-manifest.json) 和 [子模块清单](review/submodule-materials.json)。
- 第 7 节列出的三个已审核 wheel 及可信的 SHA-256 清单。

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

```bash
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

## 7. 第三批：隔离准备 pytest，运行 host 回归

执行节点：先在 P0 节点执行。若四节点源码、解释器或运行依赖不一致，按不同 profile 分别重跑；host 结果不能替代四节点 NPU/通信验收。

### 7.1 准备并校验三个 wheel

沿用前一次本机 host 基线的测试工具版本：

| 包 | 版本 | 预期 wheel 文件名 |
| --- | --- | --- |
| pytest | 9.0.2 | `pytest-9.0.2-py3-none-any.whl` |
| iniconfig | 2.3.0 | `iniconfig-2.3.0-py3-none-any.whl` |
| pluggy | 1.6.0 | `pluggy-1.6.0-py3-none-any.whl` |

当前 Python 3.11 环境已有 packaging 26.0、pygments 2.20.0，满足上述 pytest 的对应依赖；numpy 和 torch 也已安装。这里只准备这三个缺失的测试包，不补装整个项目的测试 requirements。

由内网管理员从批准来源准备 wheel 和可信的 `SHA256SUMS`。清单中的文件名使用相对 wheel 目录的名称。自行计算哈希可以留痕，但不能代替对制品来源和期望哈希的审核。缺少材料时停止本节，不改走未批准的公网源。

修改下面的占位路径后执行：

```bash
P0_HOST_WHEELS="/实际已审核的host-test-wheel目录"

if [[ -d "$P0_HOST_WHEELS" && -f "$P0_HOST_WHEELS/SHA256SUMS" ]]; then
    p0_capture host-wheel-checksums.txt \
        bash -c 'cd "$1" && sha256sum --check --strict SHA256SUMS' \
        p0-wheel-check "$P0_HOST_WHEELS"
else
    printf '停止：请先准备已审核的三个 wheel 和 SHA256SUMS\n' >&2
fi
```

继续前，确认校验输出包含上述三个 wheel 且全部为 `OK`，而不只是某个无关文件校验成功。

### 7.2 仅安装到本批次独立目录

此步骤有文件写入，但只写入新建的 `host-test-deps` 目录，不修改原 Python 环境，不安装框架，不重新解析或安装 torch。不要设置全局 `PYTHONPATH`、不要修改服务启动配置。

```bash
if p0_succeeded host-wheel-checksums.txt && \
   [[ ! -e "$P0_NEXT_DIR/host-test-deps" ]]; then
    p0_capture host-test-deps-install.txt \
        python -B -m pip --disable-pip-version-check install \
        --no-index \
        --find-links "$P0_HOST_WHEELS" \
        --only-binary=:all: \
        --no-deps \
        --no-cache-dir \
        --no-compile \
        --target "$P0_NEXT_DIR/host-test-deps" \
        "pytest==9.0.2" \
        "iniconfig==2.3.0" \
        "pluggy==1.6.0"
else
    printf '停止：wheel 校验未通过，或测试依赖目录已存在；检查记录或使用新批次\n' >&2
fi
```

`host-test-deps-install.txt.rc` 非 0 时，不继续运行 host 检查，不在原环境中重试全局安装。

### 7.3 执行固定 host 子集

```bash
if p0_succeeded host-test-deps-install.txt && \
   [[ -f "$P0_WORKSPACE/design/p0/tools/run_host_checks.py" && \
      ! -e "$P0_NEXT_DIR/host-baseline" ]]; then
    p0_capture host-run.txt \
        env PYTHONPATH="$P0_NEXT_DIR/host-test-deps${PYTHONPATH:+:$PYTHONPATH}" \
        python -B "$P0_WORKSPACE/design/p0/tools/run_host_checks.py" \
        --workspace "$P0_WORKSPACE" \
        --output "$P0_NEXT_DIR/host-baseline"
else
    printf '停止：测试依赖未准备成功、检查脚本缺失或结果目录已存在\n' >&2
fi
```

检查脚本在各仓独立进程中运行 AST/mock 子集，禁用项目 conftest 与 pytest 插件自动加载，不执行 NPU 模型或通信测试。初始化检查会间接导入现有 torch；不能将其表述为完全没有框架导入。

检查以下产物：

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

- [ ] 四节点 `pip-check.txt` 的问题列表和退出码。
- [ ] 与问题相关的 `pip-inspect.json` 依赖条目及来源结论；不是未经审查的完整原件。
- [ ] `python`/`python3` 是否一致，模块实际来源与服务配置是否对应。
- [ ] 四仓 commit/tree、工作区差异说明、子模块状态。
- [ ] 候选 torch_npu、triton-ascend 制品来源/哈希是否已核对。
- [ ] `model-quant-summary.json` 和现有权重制品身份结论。
- [ ] 第 6 节的服务配置摘要，特别是 LMCache/Mooncake/RemoteFill 是否实际启用。
- [ ] host `summary.json`、退出码；如失败，附必要的最小错误片段。
- [ ] 当前未完成项和停止原因，标明 `FAIL` 或 `PENDING`，不得记作通过。

本批原件写入内网工作区的 `p0-intranet-results/next-<角色>.<随机后缀>/`，不覆盖前一批 `env-*.json`。完整日志、trace、业务输入输出、凭据、敏感地址和未审核制品来源留在内网，不自动打包上传或提交 Git。当前用于分析的 `design/p0/p0-intranet-results/` 含原始报告，不能因方便交接而默认将其加入版本控制。

## 10. 参考

以下链接供人工查阅；上述执行命令不访问这些网站。

- [原 P0 内网执行手册](intranet-runbook.md)。
- [P0 当前支持矩阵](support-matrix.json)。
- [Python 版本匹配规范](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-matching)。
- [pip check 的检查范围](https://pip.pypa.io/en/stable/cli/pip_check/)。
- [pytest 9.0.2 包元数据](https://pypi.org/pypi/pytest/9.0.2/json)。
