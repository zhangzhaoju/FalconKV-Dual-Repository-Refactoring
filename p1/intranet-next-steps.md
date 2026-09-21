# P1 内网执行步骤

适用：已授权的 P1 首批双仓源码，不适用于原四仓的旧安装命令。所有编译、sdist/wheel、安装和 NPU 验证都在内网完成；本机仅提供源码和审计工具。当前服务继续保留，基线由其他开发人员验证后归档。

## 1. 交付及解包

交付以下材料：`vllm-p1-source.tar.gz`、`LMCache-p1-source.tar.gz`、`SHA256SUMS`、`delivery-manifest.json`，以及整个 `design/p1`（可排除 bundle、大型 deliveries 目录）。不要直接复制 worktree 的 .git 文件，它指向本机原仓。

本批普通源码包已经生成在 `design/p1/deliveries/source-01/`，可直接上传，无需重跑。以下记录生成方式；后续源码有变更时改用新的输出批次名，工具拒绝覆盖已有目录。这不是调用 backend 的 sdist：

```bash
python3 -B design/p1/tools/export_sources.py \
  --workspace p1-worktrees --output design/p1/deliveries/source-01
```

内网将材料放在专门的交付目录，以其为当前目录执行：

```bash
set -euo pipefail
sha256sum --check SHA256SUMS
P1_RUN=$(mktemp -d /workspace/p1-validation.XXXXXXXX)
export P1_RUN
mkdir "$P1_RUN/source"
tar -xzf vllm-p1-source.tar.gz -C "$P1_RUN/source"
tar -xzf LMCache-p1-source.tar.gz -C "$P1_RUN/source"
export P1_SOURCE_WORKSPACE="$P1_RUN/source"
export P1_TOOLS=/workspace/zzj/design/p1/tools
```

将 P1_TOOLS 改成实际上传路径；后续命令使用同一个 shell。原仓路径仍按 `/workspace/zzj/{vllm,vllm-ascend,LMCache,LMCache-Ascend}` 举例，不在其下安装或覆盖文件。P1_RUN 是全新目录。

## 2. 复用固定子模块并检查源码

该步骤只读取原始内网子模块，向新交付目录写入快照及材料清单，不联网。工具要求原子模块 HEAD 精确匹配且工作区干净；不符合时先由材料负责人核实，不能 reset/checkout 掩盖已有改动。

```bash
python -B "$P1_TOOLS/materialize_submodules.py" \
  --sources /workspace/zzj --destination "$P1_SOURCE_WORKSPACE"
python -B "$P1_TOOLS/audit_sources.py" \
  --workspace "$P1_SOURCE_WORKSPACE" \
  --manifest "$P1_TOOLS/../baseline/source-manifest.json" \
  --output "$P1_RUN/source-audit.json"
python -B -m unittest discover -s "$P1_TOOLS/../tests" -v
python -B "$P1_TOOLS/run_host_checks.py" \
  --workspace "$P1_SOURCE_WORKSPACE" --output "$P1_RUN/host"
```

固定提交：CATLASS `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`；kvcache-ops `9f18d2339bc58a43429f7d5bdaef1628c820eff5`。构建 helper 重新校验清单内全部文件，校验失败就停止。重复执行请重新解包到新目录，不覆盖已经填充的子模块。

host 门槛仍为 16+31+39+26=112 项通过，0 fail/error/skip。pytest 已安装，不需要再次安装。这个子集不初始化 NPU，但部分用例需要已有 torch；不能跳过失败来凑通过数。

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
python -B "$P1_TOOLS/preflight.py" \
  --workspace "$P1_SOURCE_WORKSPACE" --output "$P1_RUN/preflight-01.json"
```

预检不导入 torch/NPU，不安装任何包；非零退出先处理报告。本轮以 `USE_HIXL=1` 构建 CANN 8.5.1 对应的 HIXL/hcomm 模块，运行时通道仍需按服务实际配置验证，不能据此认定全部传输能力通过。

注意：

- 依赖入口仅为两仓 `requirements/ascend.txt` 与 `requirements/build.txt`。缺失项由内网制品库或允许的 proxy 单独准备、固定版本/哈希；不执行原始全量 requirements 安装，不自动升级 torch/numpy/OpenCV。
- 预检包含 sdist 前端 `build>=1.2`。它可能尚未安装；与 pytest 已安装是不同事项。按内网制品策略准备，构建步骤不自动获取。
- 最新 P0 pip check 有 18 条问题。本轮声明去除了 CUDA 专用依赖并对齐候选版本；aiofile、awscrt、redis、sortedcontainers、Prometheus exporter 和 CANN 的实际缺失依赖仍需解决。合并声明还会检查 quart 等原 Ascend 需求，以预检实际输出为准。
- getopt、inspect、multiprocessing 属于标准库，不能从 PyPI 安装同名包补数。其 CANN 错误分发声明应形成 SDK 元数据问题记录；只有经批准的明确豁免可例外，其他缺依赖/冲突必须处理。
- 非 C8 权重兼容性仍由基线负责人验证；不要修改 quant_model_description.json 或把 GLM 权重量化改成另一种格式。
- 如实际缓存配置使用 native Mooncake L2，不能保持 BUILD_MOONCAKE=0：先提供匹配的 include/lib 和 ABI 材料，再设为 1。RemoteFill 所需 CANN Python Mooncake/服务也要独立核对，不把这个开关视作 RemoteFill 开关。
- 代理只用于批准的材料源；不接入外部大模型/Agent，不自动上传日志或模型元数据。

## 4. 构建两个候选 wheel，并从 sdist 重建

仅当上述材料和预检通过，在同一已准备环境执行。每次失败修复后重新解包到新 P1_RUN；脚本拒绝复用已有 ACLNN 生成目录。不要执行旧的 `VLLM_TARGET_DEVICE=empty` 或 `NO_CUDA_EXT=1` 路径，不使用 editable 安装。

```bash
mkdir "$P1_RUN/wheels" "$P1_RUN/sdist" "$P1_RUN/rebuilt-wheels"
python -m pip --disable-pip-version-check wheel --no-index --no-deps \
  --no-build-isolation --wheel-dir "$P1_RUN/wheels" "$P1_SOURCE_WORKSPACE/vllm" \
  > "$P1_RUN/build-vllm.log" 2>&1
python -m pip --disable-pip-version-check wheel --no-index --no-deps \
  --no-build-isolation --wheel-dir "$P1_RUN/wheels" "$P1_SOURCE_WORKSPACE/LMCache" \
  > "$P1_RUN/build-lmcache.log" 2>&1
python -m build --sdist --no-isolation --outdir "$P1_RUN/sdist" \
  "$P1_SOURCE_WORKSPACE/vllm" > "$P1_RUN/sdist-vllm.log" 2>&1
python -m build --sdist --no-isolation --outdir "$P1_RUN/sdist" \
  "$P1_SOURCE_WORKSPACE/LMCache" > "$P1_RUN/sdist-lmcache.log" 2>&1

python -m pip --disable-pip-version-check wheel --no-index --no-deps \
  --no-build-isolation --wheel-dir "$P1_RUN/rebuilt-wheels" \
  "$P1_RUN/sdist/vllm-0.18.0+ascend.p1.tar.gz" \
  > "$P1_RUN/rebuild-vllm.log" 2>&1
python -m pip --disable-pip-version-check wheel --no-index --no-deps \
  --no-build-isolation --wheel-dir "$P1_RUN/rebuilt-wheels" \
  "$P1_RUN/sdist/lmcache-0.4.3+ascend.p1.tar.gz" \
  > "$P1_RUN/rebuild-lmcache.log" 2>&1
```

上面任一步非零退出就停止，保留日志，不继续安装。最终 wheel 为 CPython 3.11 / aarch64 本地候选产物；尚未审计 manylinux 通用兼容性。

核对内容（每个目录应恰好一对 wheel；如文件名不同先核实，不随意选择旧产物）：

```bash
python -B "$P1_TOOLS/inspect_wheels.py" \
  --vllm-wheel "$P1_RUN/wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --lmcache-wheel "$P1_RUN/wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P1_RUN/wheel-contents.json"
python -B "$P1_TOOLS/inspect_wheels.py" \
  --vllm-wheel "$P1_RUN/rebuilt-wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --lmcache-wheel "$P1_RUN/rebuilt-wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P1_RUN/rebuilt-wheel-contents.json"
```

检查只证明归档结构/元数据存在，不证明 ELF 依赖、ABI、资源装载或推理通过。记录两条构建链产物哈希；暂不要求含构建路径/时间的二进制逐字节相同，但版本、模块集合和行为必须一致。

## 5. 仅在干净的验证容器安装

由内网负责人准备**没有旧四包、旧 editable/.pth 和工作区 PYTHONPATH**的独立验证容器，基础 torch/NPU/CANN 与候选一致。直接建 system-site-packages venv 不保证隔离旧包。不要卸载或覆盖现有 GLM 服务容器中的包。

在该新容器设置对应的 P1_RUN/P1_TOOLS，离开源码目录后先检查：

```bash
cd "$P1_RUN"
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
python -m pip --disable-pip-version-check install --no-index --no-deps \
  "$P1_RUN/wheels/vllm-0.18.0+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  "$P1_RUN/wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl"
python -m pip list --format=json > "$P1_RUN/installed-packages.json"
python -m pip check > "$P1_RUN/installed-pip-check.txt" 2>&1
```

pip check 的任何非零退出都需要解释；不能用 `|| true` 把它变成通过。若仅剩已确认 CANN 标准库元数据错误，必须连同具体三条错误、原因和负责人豁免归档，仍保留真实退出码。

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

保存 source/delivery/submodule 清单、preflight、17 项约束测试、112 项 host XML/日志、全部构建日志、sdist/wheel 哈希、安装路径/依赖/ABI、两组配置与基线/P1 对照报告。完整原件留内网，仅人工审核脱敏后的必要结果交接；不上传凭据、业务输入、敏感地址或模型权重。

以下任一情况保持未通过：基线未归档；源码/材料校验不符；缺失真实依赖；sdist 无法独立重建；旧插件或旧路径帮助导入；native/ABI 错误；任一必保场景或性能门槛失败。缺少基线数据不阻止本轮静态工作，但不能通过 P1 出口或直接推进大规模裁剪。
