# P3 内网构建、开发态安装与 P2/P3 联合验证

本页适用于 **P3-01～05 原生源码交付及截至 2026-09-30 的 NPU 导入和 DSA KV 绑定修复**，取代首批仅配置迁移的安装说明。精确配对提交见 [最新交付清单](baseline/kv-cache-binding-fix-20260930.json)；不要仅根据相同 `+ascend.p3` 后缀混用不同批次 wheel。累计修复含上一轮新增模块，按本页重装核验；见 [修复说明](../p2/npu-bootstrap-fix.md)。源码/host 检查通过不代表 NPU 或推理验收通过。

## 1. 交接源码并保留 P2

本次没有自动推送。若通过 GitHub 交接，在外网源码机确认两个 `origin` 分别是自己的 vllm-dual / LMCache-dual 后执行：

```bash
git -C p1-repos/vllm remote get-url origin
git -C p1-repos/LMCache remote get-url origin
git -C p1-repos/vllm push origin p3
git -C p1-repos/LMCache push origin p3
```

不强推、不移动 P2/P1/main；Actions 按原设置保持开启。将本页和配对清单一起上传内网。内网可通过批准的 proxy 获取源码/依赖材料；禁止接入外部大模型或远程 Agent。

两阶段分别使用专用容器/解释器环境和源码 checkout。**不能在正在使用 editable 的 P2 checkout 中切换到 P3。** 原 P1/基线服务及其缓存不动。P2 包是 vLLM `0.18.0+ascend.p2` + 对应 P2 提交的 LMCache `0.4.3+ascend.p1`；P3 两包均为 `+ascend.p3`。

以下假设 P3 已在独立容器使用同样的 `/workspace/zzj/p1-repos` 布局。只在专属且干净的 P3 checkout 同步：

```bash
set -euo pipefail
export P3_ROOT=/workspace/zzj
export P3_REPOS="$P3_ROOT/p1-repos"
git -C "$P3_REPOS/vllm" fetch origin p3
git -C "$P3_REPOS/LMCache" fetch origin p3
```

首次检出用 `git switch --track origin/p3`；已有本地 p3 用 `git switch p3` 后 `git merge --ff-only origin/p3`。有未提交变更先归档审核，不用 reset/强制 checkout。核对下面 HEAD **必须等于配对清单**，不能仅检查有 P2 祖先：

```bash
git -C "$P3_REPOS/vllm" status --short --branch
git -C "$P3_REPOS/vllm" rev-parse HEAD
git -C "$P3_REPOS/LMCache" status --short --branch
git -C "$P3_REPOS/LMCache" rev-parse HEAD
test "$(git -C "$P3_REPOS/vllm" rev-parse HEAD)" = 7854ce2158f42a725c58cd1026b388709f320b2b
test "$(git -C "$P3_REPOS/LMCache" rev-parse HEAD)" = a4e2131e890727edadb6bb61cde369900f72c3fd
git -C "$P3_REPOS/vllm" merge-base --is-ancestor f1be323571e3ca2aab53992234045dd064d1967f HEAD
git -C "$P3_REPOS/LMCache" merge-base --is-ancestor cfe8a1754db743d41c8bb63f8d02ad7c3051948c HEAD
```

## 2. 前置检查

候选环境：910B3、4×8 卡、CANN 8.5.1、Python 3.11.14/aarch64、torch 2.9.0+cpu、torch_npu 2.9.0.post2、Transformers 5.2.0、triton-ascend 3.2.0.dev20260322。不为执行本页替换 torch。环境中不可有旧四包、旧 editable/.pth/PYTHONPATH 污染；继承旧框架的 system-site-packages 环境也会被拒绝。`--isolated-env` 只确认环境隔离，不创建环境。

```bash
mkdir -p "$P3_ROOT/p3-check"
P3_RUN=$(mktemp -d "$P3_ROOT/p3-check/run.XXXXXXXX")
export P3_RUN
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
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
cd "$P3_RUN"
python -B "$P3_REPOS/vllm/p1_dev.py" doctor --output "$P3_RUN/vllm-doctor.json"
python -B "$P3_REPOS/LMCache/p1_dev.py" doctor --output "$P3_RUN/lmcache-doctor.json"
python -B "$P3_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P3_RUN/pip-check-before"
python -B "$P3_REPOS/vllm/tools/check_npu_native.py" > "$P3_RUN/vllm-source.json"
python -B "$P3_REPOS/LMCache/tools/check_npu_native.py" > "$P3_RUN/lmcache-source.json"
```

缺少固定子模块材料时按 [材料准备说明](../p1/development-build-install.md#3-一次性准备固定子模块材料) 使用 `p1_dev.py materials` 注册/复制已审核 CATLASS、kvcache-ops；不要重新跑初始四仓合仓/物化脚本。只豁免已批准的 op-compile-tool 0.1.0 三条标准库误声明，其他错误仍阻断。

## 3. 开发态路线

P1 提供的 `p1_dev.py` 名称和 strict editable 机制继续沿用，**必须使用 P3 分支中的更新版本**，不能把 P1/P2 脚本复制过来。

```bash
python -B "$P3_REPOS/vllm/p1_dev.py" editable --isolated-env --output "$P3_RUN/vllm-editable"
python -B "$P3_REPOS/LMCache/p1_dev.py" editable --isolated-env --output "$P3_RUN/lmcache-editable"
python -B "$P3_REPOS/vllm/p1_dev.py" verify --mode editable --output "$P3_RUN/vllm-editable-paths.json"
python -B "$P3_REPOS/LMCache/p1_dev.py" verify --mode editable --output "$P3_RUN/lmcache-editable-paths.json"
```

首次会完整编译，保留命令日志、源码 checkout、`build/__editable__.*` 和 `build/p1-native/run-*/`。已有 Python 文件修改后重启本批测试进程；新增/移动文件、原生代码/依赖/资源变更后重新安装。不要借用 P2/P3-01 的 .so。

新布局只有 `vllm` 和 `lmcache` 两个根 namespace。`lmcache.c_ops`、`libcache_kernels.so`、HIXL/HCCL 和 host 扩展都安装在 `lmcache/`；没有 `lmcache_ascend/`。构建缺失文件时必须修复构建，不能通过创建空包或全局 transfer_to_npu 掩盖。

## 4. 正式 wheel / sdist 路线

正式验收使用普通 wheel，在另一个干净专用环境执行，不能用 editable 验收替代：

```bash
python -B "$P3_REPOS/vllm/p1_dev.py" build --output "$P3_RUN/vllm-wheel"
python -B "$P3_REPOS/LMCache/p1_dev.py" build --output "$P3_RUN/lmcache-wheel"
python -B "$P3_REPOS/vllm/p1_dev.py" install --isolated-env \
  --wheel "$P3_RUN/vllm-wheel/wheels/vllm-0.18.0+ascend.p3-cp311-cp311-linux_aarch64.whl" \
  --output "$P3_RUN/vllm-install"
python -B "$P3_REPOS/LMCache/p1_dev.py" install --isolated-env \
  --wheel "$P3_RUN/lmcache-wheel/wheels/lmcache-0.4.3+ascend.p3-cp311-cp311-linux_aarch64.whl" \
  --output "$P3_RUN/lmcache-install"
python -B "$P3_REPOS/vllm/p1_dev.py" verify --mode wheel --output "$P3_RUN/vllm-paths.json"
python -B "$P3_REPOS/LMCache/p1_dev.py" verify --mode wheel --output "$P3_RUN/lmcache-paths.json"
```

独立 sdist 重建验证不安装任何新依赖，不使用原生构建缓存：

```bash
python -B -m build --sdist --no-isolation --outdir "$P3_RUN/sdist" "$P3_REPOS/vllm"
python -B -m build --sdist --no-isolation --outdir "$P3_RUN/sdist" "$P3_REPOS/LMCache"
python -B -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$P3_RUN/rebuilt-vllm" "$P3_RUN/sdist/vllm-0.18.0+ascend.p3.tar.gz"
python -B -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$P3_RUN/rebuilt-lmcache" "$P3_RUN/sdist/lmcache-0.4.3+ascend.p3.tar.gz"
sha256sum "$P3_RUN"/sdist/*.tar.gz "$P3_RUN"/*/wheels/*.whl "$P3_RUN"/rebuilt-*/*.whl > "$P3_RUN/SHA256SUMS"
```

重建 wheel 也必须在单独环境使用本页 install/verify/导入检查，不止确认 pip 返回 0。不要求两个构建的 ZIP 字节 hash 一致；要求输入身份可追溯、内容/资源、依赖及运行契约一致。

## 5. 安装后导入、spawn 与 ABI

从源码目录之外执行，并确保没有手工设置的源码 PYTHONPATH。下面的检查不加载权重、不启动推理或网络服务；冷导入检查不使用已有 model-info 缓存，正常导入可加载原生库并生成既有 profiling 配置。`p3_runtime_smoke.py` 验证 LMCache 原生类和扩展来源，以及导入前后 torch CUDA API/构造函数未被全局替换。

```bash
cd "$P3_RUN"
python -B "$P3_REPOS/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --output "$P3_RUN/bootstrap"
python -B "$P3_REPOS/vllm/tools/validate_npu_native.py" \
  --output "$P3_RUN/vllm-native-import.json" --spawn --torchair-abi
python -B "$P3_REPOS/LMCache/tools/p3_runtime_smoke.py" \
  --output "$P3_RUN/lmcache-native-import"
python -B "$P3_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P3_RUN/pip-check-after"
```

`bootstrap` 七项必须全部通过，包含用显式 CPU 张量调用实际原生函数的 `kv_cache_bind` 检查；旧六项结果不能替代它。四个节点均核对修复制品和配对提交并执行安装后检查，再启动 2P2D；不要在暖进程提前导入平台来代替冷导入验收。

只有确认空闲且分配给本批测试的设备，才设置相应 `ASCEND_RT_VISIBLE_DEVICES` 并追加 `--device-smoke`（vLLM）或 `--npu`（LMCache，4 KiB 注册 host/NPU 拷贝）；生成新的报告目录。不要直接使用仍承载基线服务的卡。IPC 真正跨进程共享、通信与内核数值仍需实际场景验收，导入/spawn 不等于通过它们。

## 6. 必须调整的启动配置

P1 日志使用的是旧动态入口：
`LMCacheAscendConnectorV1Dynamic` /
`lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1`。

推荐改为 vLLM 内置正式入口，**删除原 `kv_connector_module_path` 字段**：

```json
{
  "kv_connector": "LMCacheConnectorV1",
  "engine_id": "保留原实例唯一engine_id",
  "kv_buffer_device": "npu",
  "kv_role": "保留原kv_role",
  "kv_connector_extra_config": {}
}
```

示例不是完整配置，不能覆盖原来的 extra_config、buffer、IP/port 等字段；只改入口两处，其余按已审核启动配置保留。若需要自定义 loader，则改用：

```json
{
  "kv_connector": "LMCacheConnectorV1Dynamic",
  "kv_connector_module_path": "lmcache.integration.vllm.lmcache_connector_v1"
}
```

`use_native=true` 旧 vendored adapter 路线不用于 P3，请保留/设为 false。不存在空旧包兜底。首次 P3 启动采用独立缓存 namespace、端口和 rendezvous；P2/P3 的 IPC 对象和进程不能混用，缓存持久化格式是否可复用需单独验证，不能删除旧服务的数据。

## 7. 联合验收及归档

GLM-5.2 原生文本，DSA 双组/MTP 开启，C8 关闭；4 节点×8 卡、2P2D，分别执行 TP8/DP2 和 TP4/DP4。P2/P3 使用相同 checkpoint、请求、采样、有效配置和并发档位，单独归档，不混用两个阶段的报告。

| 项目 | 必须保留的证据 |
| --- | --- |
| 无 LMCache、离线与在线生成 | 原生 NPU 平台、独立导入、实际请求成功及输出对照 |
| CPU KV 冷/热命中、跨实例/P-D | 双组实际层数（含 MTP）、缓存 key/布局、load/store/事件顺序、host registration |
| DSA staged 路径 | destination sealing、首次 prepared load、跨 chunk/两组完整性、并发请求不串数据 |
| checkpoint/RemoteFill | 抢占恢复、取消、失败回退、超时、armed 传输不可提前释放、fatal 成对重启与恢复 |
| 性能 | 相同 workload 的 benchmark 原始 JSON、成功/失败/完成数、TTFT/TPOT/吞吐；按已审核门槛比较 |
| 制品 | 两仓 HEAD/状态、镜像/软件身份、wheel/sdist SHA256、配置 hash、完整构建/导入日志 |

保留旧 enable_pd/enable_p2p 与 layerwise 的禁配规则，不将其组合成新增验收能力。历史 Ascend 独立 multiprocess GPU cache server 未实现、非目标 CacheBlend 模型不在本批入口范围。原 donor 目录中的历史测试不能全量直接作为 P3 验收脚本；本批 source/host 门禁、安装 smoke 和上述业务矩阵是明确入口。

发生构建、ABI、导入或推理错误即停在对应门槛，保留完整 traceback 和配置。不要把 P1/基线已有缺陷算作 P3 新回归，也不能忽略它们后宣告阶段通过。完成后人工审核脱敏，将新 `p3-check/run.*` 目录及模型服务/benchmark 证据交接到源码机；不自动外传原始日志。
