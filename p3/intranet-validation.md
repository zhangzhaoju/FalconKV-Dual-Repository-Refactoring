# P3 内网构建、开发态安装与 P2/P3 联合验证

本页适用于 **P3-01～05 原生源码交付及截至 2026-10-03 的 NPU 导入、DSA KV 绑定、旧启动入口迁移和 Worker 完成通知修复**，取代首批仅配置迁移的安装说明。精确配对提交见下方交接步骤；不要仅根据相同 `+ascend.p3` 后缀混用不同批次 wheel。P2 升级 P3 必须两包重装；已经正确安装 P3 的 strict editable 环境，本次只更新已有 Python 文件，无需原生重编译。源码/host 检查通过不代表 NPU 或推理验收通过。

本次 [配对清单](baseline/connector-output-fix-20261003.json)：vLLM `8767fe1121a085e8cefed03eeaa1444b55227521`，LMCache 保持 `a4e2131e890727edadb6bb61cde369900f72c3fd`。已安装正确 P3 strict editable 的环境按第 1 节同步，再执行第 6.2 节快速复测；不必重复第 3/4 节原生构建。首次安装、阶段升级和普通 wheel 更新仍走相应完整流程。

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
test "$(git -C "$P3_REPOS/vllm" rev-parse HEAD)" = 8767fe1121a085e8cefed03eeaa1444b55227521
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
  --inspect-glm --check-lmcache --output "$P3_RUN/bootstrap"
python -B "$P3_REPOS/vllm/tools/validate_npu_native.py" \
  --output "$P3_RUN/vllm-native-import.json" --spawn --torchair-abi
python -B "$P3_REPOS/LMCache/tools/p3_runtime_smoke.py" \
  --output "$P3_RUN/lmcache-native-import"
python -B "$P3_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P3_RUN/pip-check-after"
```

上述配对命令的 `bootstrap` 八项必须全部通过，包含用显式 CPU 张量调用实际原生函数的 `kv_cache_bind`，以及新增的 `lmcache_completion`；旧六/七项结果不能替代本次配对检查。四个节点均核对修复制品和配对提交并执行安装后检查，再启动 2P2D；不要在暖进程提前导入平台来代替冷导入验收。单独验证无 LMCache 的场景时不加 `--check-lmcache`，保留可选依赖边界。

2026-10-03 修复后的 `config` 子检查还须返回 `lmcache_config_migration`，其
`connector` 为 `vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector.LMCacheConnectorV1`，
`instance_created=false`。这一步使用实际安装的配置类和 Factory，仅查找类，不创建
Connector 或加载权重；同时比对安装态 `config/kv_transfer.py` 与 checkout 的内容。

新增 `lmcache_completion` 用实际安装的正式 Connector、LMCache adapter 和
`KVConnectorOutput`，在隔离的主机测试状态上检查两个连续成功冷加载和一个无效块
拒绝场景；不调用两类的初始化函数，不创建缓存引擎、网络服务或 NPU 张量。
要求 `cases_passed=3`、`kv_events_enabled=false`，并比对安装态正式 Connector
源码与 checkout。它验证完成通知、冷加载名额释放和恢复标记，不等于真实 KV
张量、图执行、并发性能或 MTP 数值验收。

只有确认空闲且分配给本批测试的设备，才设置相应 `ASCEND_RT_VISIBLE_DEVICES` 并追加 `--device-smoke`（vLLM）或 `--npu`（LMCache，4 KiB 注册 host/NPU 拷贝）；生成新的报告目录。不要直接使用仍承载基线服务的卡。IPC 真正跨进程共享、通信与内核数值仍需实际场景验收，导入/spawn 不等于通过它们。

## 6. 必须调整的启动配置

### 6.1 已交付的旧入口迁移修复

P1/P2 及 2026-10-03 的 P3 失败日志使用的是旧动态入口：
`LMCacheAscendConnectorV1Dynamic` /
`lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1`。

失败日志中四节点都已加载 `+ascend.p3` 的 vLLM，但启动参数仍包含上述两项；
错误发生在 Factory 导入旧模块时，不是要求补装独立 `lmcache-ascend`。
四个日志均在第 7 行记录旧入口，首个缺模块错误分别是 P dp0 的第 614 行、
P dp1 的第 421 行、D dp0 的第 3945 行、D dp1 的第 3752 行，
原始证据保留在 `p3_logs/logs_no_lmcache_asend/`。调用链为
`initialize_from_config → ensure_kv_transfer_initialized → KVConnectorFactory → import_module`。
这组日志尚不能证明原生 `lmcache` 安装正确；仍需四个容器各自执行安装态检查，
不能把某一节点成功作为其余节点安装成功的证据。

此前的配置修复在 `KVTransferConfig` 初始化时，对该已知旧类名配合该精确模块路径
（或未提供模块路径）的配置输出迁移告警，并转换为下面的原生正式入口。
engine ID、角色、地址、端口、buffer 和 extra_config 均不改动。
其他 `lmcache_ascend.*` 模块配置提前报错；自定义的非旧包模块不按类名强制重定向。
这仅是启动字段迁移，不恢复旧包、旧类或导入补丁。`use_native=true` 也会提前拒绝。

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

对于此前 `No module named 'lmcache_ascend'` 及本次完成通知问题的复测，四个运行容器均同步修复后的
vLLM P3 源码；已正确安装的 LMCache P3 提交不变。已有 P3 strict editable
链接到这些源码文件时，更新后重启本批测试进程即可，并从源码目录之外重新执行
第 5 节安装态检查。若仍安装 P2、链接来源不一致，或使用普通 wheel，则按第 3/4 节
重装相应制品，不能单凭 `+ascend.p3` 后缀认定已带修复。不要停止其他基线服务。

### 6.2 Decode 并发与稀疏恢复修复：本次复测入口

`p3_logs/logs_all_RunReq_less/` 中已改成正式入口，不能再归因为旧模块名。
正式 `LMCacheConnectorV1.update_connector_output()` 原先只聚合 KV events，
没有把 Worker 输出交给 adapter；`enable_kv_events=false` 时直接返回。
因此首个冷加载虽然完成，adapter 的 direct-HBM 活跃请求标记没有释放，后续请求
持续等待；首请求也没有获得 `dsa_cold_compact_resume`，稀疏恢复元数据不完整。

证据：D0 日志第 6184 行已变成 Running=0、Waiting=19、KV=0%；
D1 第 6308 行仍 Running=1、Waiting=14。D1 第 5280 行的实际执行路由为
`safe_native/dense_prefix_hit`、`cold_compact_resume_count=0`、graph `NONE`。
这不是把 `max_num_seqs=16` 调大可以解决的问题。修复现已在任何 KV-event 提前返回
之前，向 adapter 原样转发完整输出且仅转发一次，保留原事件聚合和异常传播。
不改 TP/DP、MTP 参数、KV events 开关、冷加载限流或 LMCache 状态机算法。

已按第 1 节同步、且原本正确安装 P3 strict editable 时，在四个测试容器的原
Python/CANN 环境分别执行下面的快速复测。此处只校验路径、导入和隔离的主机完成
状态，不编译、不启动服务；保留原 CANN 环境，但不能有手工源码 PYTHONPATH 污染：

```bash
set -euo pipefail
P3_ROOT=/workspace/zzj
P3_REPOS="$P3_ROOT/p1-repos"
test "$(git -C "$P3_REPOS/vllm" rev-parse HEAD)" = 8767fe1121a085e8cefed03eeaa1444b55227521
test "$(git -C "$P3_REPOS/LMCache" rev-parse HEAD)" = a4e2131e890727edadb6bb61cde369900f72c3fd
mkdir -p "$P3_ROOT/p3-check"
P3_RUN=$(mktemp -d "$P3_ROOT/p3-check/connector-output.XXXXXXXX")
cd "$P3_RUN"
python -B "$P3_REPOS/vllm/p1_dev.py" verify --mode editable --output "$P3_RUN/vllm-paths.json"
python -B "$P3_REPOS/LMCache/p1_dev.py" verify --mode editable --output "$P3_RUN/lmcache-paths.json"
python -B "$P3_REPOS/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --check-lmcache --output "$P3_RUN/bootstrap"
python -B "$P3_REPOS/LMCache/tools/p3_runtime_smoke.py" --output "$P3_RUN/lmcache-native-import"
```

四容器都通过后，重启本批 P3 测试进程，使用各实例原有 engine ID、配置和已审核
2P2D 启动脚本。继续沿用已知旧两字段时会看到迁移 WARNING；改为原生入口后不应
再出现该 WARNING。两种情况的有效 `KVTransferConfig` 都应显示
`kv_connector='LMCacheConnectorV1'`、`kv_connector_module_path=None`。
保留新日志，不覆盖这次失败日志；导入通过并不代表推理或性能验收通过。

先用 P2 已验证输入进行小规模并发冷加载，再恢复完整 benchmark：

1. 保持 GLM-5.2、DSA 双组、MTP 开启、C8 关闭和原 TP8/DP2 配置，不为绕过问题
   打开 KV events、关闭 MTP 或换回动态 Connector。确保请求的实际输入 token 数
   （含模板）加 `max_tokens` 不超过 140000；原 D1 第 6144 行的
   `130001 + 10000 > 140000` 是独立的请求边界错误，需要排除。
2. 先并发 2～4 条能完整命中双组的长前缀请求，确认两个 D 实例各自收到至少两条，
   并让第一条仍在解码时第二条完成冷加载。按不同 request ID 计数，不能把同一个
   请求 8 个 TP worker 的 ready 日志当作 8 个请求。冷加载只允许单个在途名额，
   不代表整个解码生命周期只能运行一个请求。
3. 实际冷恢复批次应出现 `cold_compact_resume_count>0`；满足原 staged graph 条件
   的批次应回到相应 eligible/PIECEWISE 路径，而不是因恢复标记丢失走
   `dense_prefix_hit`。输入供给和容量允许时 Running 应能超过 1，不能继续出现
   大量 Waiting 且 Running=0、KV=0 的持续停滞。无需要求每个采样窗口 Running
   都大于 1，正常尾部排空和真实 dense 请求仍可能走其他合法路径。
4. 先核对生成内容、请求成功数及 finish reason，再用相同输入、采样、并发和完整
   运行区间对比 P2/P3。MTP 接收率用总 accepted / 总 drafted 计算，不简单平均
   每个日志窗口的百分比。原日志低接收率与错误稀疏恢复路径同时出现，但本机
   未做 NPU 数值测试，不能提前保证接收率已恢复或给出新的性能验收结论。

归档四节点完整服务日志、proxy/client 日志、原始 benchmark JSON、有效启动
配置及本节八项检查报告；若仍异常，保留第一条失败请求及前后的冷加载/路由日志。
不要在已运行的进程里重载模块，也不要删除基线的缓存或 IPC 数据。

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
