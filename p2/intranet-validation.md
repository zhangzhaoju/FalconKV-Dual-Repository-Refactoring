# P2 内网统一验证

本次交付是完整 P2 源码候选。源码身份以 [配对提交清单](baseline/p2-native-integration-20260927.json) 为准，两仓均用 `p2`，不要使用保留的 `main`。本机检查记录见 [结果](results/native-integration-20260927/README.md)；本页命令未在本机执行构建或 NPU 验证。

## 1. 环境和源码

沿用 [P2 profile](profile.json)：Python 3.11/aarch64、910B3、CANN 8.5.1、torch 2.9.0（内网已有 2.9.0+cpu）、torch_npu 2.9.0.post2、Transformers 5.2.0、triton-ascend 3.2.0.dev20260322。精确依赖以两仓 `requirements/{build,ascend}.txt` 和已准备环境为准，不升级 torch。TorchAir 记录实际版本、安装路径和来源，另执行下方 compiler ABI/图检查。

使用专用验证容器，保留基线/P1 服务与环境；清除旧四仓 PYTHONPATH/editable 污染。LMCache 虽仍标 `0.4.3+ascend.p1`，必须配用本批新提交重新构建。`p1_dev.py`、`p1_build_info.json` 和 `build/p1-native/` 是保留的工具/记录名称，不是版本判断依据。

将下列路径替换为内网实际路径；两个仓库先检出清单中的准确 SHA，记录 `git rev-parse HEAD` 和工作树差异。

```bash
set -euo pipefail
export P2_ROOT=/workspace/zzj
export P2_REPOS="$P2_ROOT/p1-repos"
mkdir -p "$P2_ROOT/p2-check"
P2_RUN=$(mktemp -d "$P2_ROOT/p2-check/run.XXXXXXXX")
export P2_RUN
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
cd "$P2_RUN"
```

保持运行目录在源码仓之外，防止源码遮蔽 wheel 或 strict editable 资源。材料准备与依赖核对沿用已批准的 P1 工具；若已有有效材料清单，重复核验即可：

```bash
python -B "$P2_REPOS/vllm/p1_dev.py" materials \
  --from-submodule "$P2_ROOT/vllm-ascend/csrc/third_party/catlass"
python -B "$P2_REPOS/LMCache/p1_dev.py" materials \
  --from-submodule "$P2_ROOT/LMCache-Ascend/third_party/kvcache-ops"
python -B "$P2_REPOS/vllm/p1_dev.py" doctor --output "$P2_RUN/vllm-doctor.json"
python -B "$P2_REPOS/LMCache/p1_dev.py" doctor --output "$P2_RUN/lmcache-doctor.json"
python -B "$P2_ROOT/design/p1/tools/check_pip_dependencies.py" \
  --output "$P2_RUN/pip-check-before"
```

材料固定 CATLASS `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`、kvcache-ops `9f18d2339bc58a43429f7d5bdaef1628c820eff5`。如无原四仓材料，按 [材料准备说明](../p1/development-build-install.md#3-一次性准备固定子模块材料) 从批准来源准备。doctor、pip 检查通过后再构建。只沿用已经批准的 `op-compile-tool` 三条标准库误声明豁免，不扩大豁免范围。

## 2. 构建、安装和资源检查

普通 wheel 路线：

```bash
python -B "$P2_REPOS/vllm/p1_dev.py" build --output "$P2_RUN/vllm-wheel"
python -B "$P2_REPOS/LMCache/p1_dev.py" build --output "$P2_RUN/lmcache-wheel"
python -B "$P2_REPOS/vllm/p1_dev.py" install --isolated-env \
  --wheel "$P2_RUN/vllm-wheel/wheels/vllm-0.18.0+ascend.p2-cp311-cp311-linux_aarch64.whl" \
  --output "$P2_RUN/vllm-install"
python -B "$P2_REPOS/LMCache/p1_dev.py" install --isolated-env \
  --wheel "$P2_RUN/lmcache-wheel/wheels/lmcache-0.4.3+ascend.p1-cp311-cp311-linux_aarch64.whl" \
  --output "$P2_RUN/lmcache-install"
python -B "$P2_REPOS/vllm/p1_dev.py" verify --mode wheel --output "$P2_RUN/vllm-paths.json"
python -B "$P2_REPOS/LMCache/p1_dev.py" verify --mode wheel --output "$P2_RUN/lmcache-paths.json"
```

开发调测可在另一专用环境选择 strict editable；首次仍完整编译，P2 新增/移动了大量文件，必须重新安装，不能复用 P1 链接树：

```bash
python -B "$P2_REPOS/vllm/p1_dev.py" editable --isolated-env --output "$P2_RUN/vllm-editable"
python -B "$P2_REPOS/LMCache/p1_dev.py" editable --isolated-env --output "$P2_RUN/lmcache-editable"
python -B "$P2_REPOS/vllm/p1_dev.py" verify --mode editable --output "$P2_RUN/vllm-editable-paths.json"
python -B "$P2_REPOS/LMCache/p1_dev.py" verify --mode editable --output "$P2_RUN/lmcache-editable-paths.json"
```

两条路线分开归档，不能把 editable 结果替代普通 wheel 验收。构建保留全部日志、CMake 命令、材料清单和生成的 `p1_build_info.json`；不得借用旧 `.so`。vLLM wheel 只能包含原生 `vllm` namespace，包含 `vllm/_ascend_C*.so`、`libvllm_ascend_kernels.so` 和 CANN vendor 资源；没有 `vllm_ascend`、旧 Ascend entry point 或 `ascend/legacy_*`。

正式制品补充：在各仓材料准备后的副本执行 `python -m build --sdist --no-isolation --outdir <新目录>`，分别将 sdist 解包到新目录，用 `python -m pip wheel --no-index --no-deps --no-build-isolation --no-cache-dir <解包目录> --wheel-dir <新目录>` 独立重建；检查资源清单、动态库依赖、ABI 和源码身份。两条命令均使用已准备环境，不安装依赖。保留 SHA256、`pip freeze`、完整和豁免后的 pip 检查结果；不要直接运行固定 P1 donor 字节/entry-point 假设的历史 source-audit 来验收 P2。

## 3. 真实导入、子进程和基本设备检查

先在没有旧插件安装、可不安装 LMCache 的环境验证 vLLM。下列脚本会禁用可选插件，主动阻止 LMCache/旧插件导入，加载 v1/v2 Runner 和原生相关模块，检查继承链与来源。`--spawn` 在新的 spawn 子进程重复；`--torchair-abi` 检查安装的 TorchAir factory 接口。可执行一次不带设备操作的探测，再在空闲设备执行扩展和 stream/event/tensor smoke：

```bash
python -B "$P2_REPOS/vllm/tools/validate_npu_native.py" \
  --spawn --torchair-abi --output "$P2_RUN/native-import.json"
ASCEND_RT_VISIBLE_DEVICES=0 python -B "$P2_REPOS/vllm/tools/validate_npu_native.py" \
  --spawn --torchair-abi --device-smoke --output "$P2_RUN/native-device.json"
python -B "$P2_ROOT/design/p1/tools/check_pip_dependencies.py" \
  --output "$P2_RUN/pip-check-after"
```

失败时保留 JSON 的 traceback 和终端输出。TorchAir 私有接口不匹配会明确失败，不修改 site-packages 或恢复旧补丁绕过。此检查只覆盖导入、工厂、扩展加载和基础设备 API，不代替实际 kernel 数值、ACL capture/replay、npugraph_ex 编译、HCCL 或模型验证。

## 4. 测试与功能矩阵

静态检查不需要 torch。内网已有 torch 后，应运行本机未能覆盖的完整 standalone 集合；不带本机的两个 `--ignore` 和一个 `-k` 排除项：

```bash
python -B "$P2_REPOS/vllm/tools/check_npu_native.py" > "$P2_RUN/native-source.json"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest \
  "$P2_REPOS/vllm/ascend/tests/standalone" "$P2_REPOS/vllm/tests/standalone" \
  "$P2_REPOS/vllm/ascend/tests/ut/core/test_kv_connector_worker_metadata_patch.py" \
  --junitxml="$P2_RUN/vllm-standalone.xml" -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest \
  "$P2_REPOS/LMCache/tests/standalone/test_p1_development.py" \
  --junitxml="$P2_RUN/lmcache-pair.xml" -q
```

其中需补齐 `test_cold_resume_native_metadata.py`、`test_glm52_topk_ownership.py`、`test_mc2_recovery.py::test_production_draft_expansion_can_exceed_target_capacity`。已有 Ascend UT 的 mock bootstrap 和 e2e 测试入口保留在 `ascend/tests/`，已更新为原生命名空间；依既定内网测试环境执行与 GLM/DSA/MTP 相关集合，保留失败/skip 原因，不把范围外模型的历史 UT 等同于新增认证。

| 编号 | 必验场景 | 判据和记录 |
| --- | --- | --- |
| P2-I01 | wheel、sdist 重建、strict editable | 两仓 SHA/版本、唯一导入来源、扩展/资源、torch 未被替换、动态库/ABI |
| P2-I02 | 关闭可选插件与 LMCache；主进程/spawn | `NPUPlatform` 原生选择，两个 Runner 无 GPU Runner 继承，无旧 namespace/patch 导入 |
| P2-I03 | 910B3 算子与 HCCL | 原有相关 kernel UT 数值/形状/dtype；多 rank all-reduce/all-to-all/broadcast 和进程组生命周期 |
| P2-I04 | GLM-5.2 无缓存离线/在线 | 固定权重/tokenizer/采样/请求；首 token、输出 token、finish reason、流式/非流式、GLM tool parser/usage |
| P2-I05 | eager、ACL capture/replay、npugraph_ex | 实际图执行证据与 eager 数值对照；混合 prefill/decode、动态 batch、staged SFA 安全回退；字典 Triton 元数据 |
| P2-I06 | MTP/权重/采样 | 已有 speculative token=1 配置；保留支持的其他步数另测；rot.weight/量化加载、拒绝/接受 token、logprob；不默认关闭 MTP 过关 |
| P2-I07 | DSA 双组、CPU KV、跨实例缓存 | latent/indexer 布局、group metadata、native metadata/RemoteFill、event handoff、首次 load、prefetch/store/callback 完整性 |
| P2-I08 | 2P2D TP8/DP2 与 TP4/DP4 | 20K/140K 固定负载与已有短请求；冷/热缓存、并发、抢占、取消、checkpoint/cold-resume、后续再次命中 |
| P2-I09 | 故障与恢复 | 读取错误/超时、各 TP rank 一致恢复/失败、destination sealing、paired restart、资源释放；P1 共同缺陷单独标识，不误记通过 |
| P2-I10 | 性能与长稳 | 基线/P1/P2 同制品身份、负载、缓存温度；成功请求口径和墙钟吞吐，TTFT/TPOT p50/p95/p99、内存、长稳资源增长 |

目标 checkpoint 名含 w4a8c8 不表示运行 cache C8 开启：本轮仍固定 **C8 off、DSA 两组、MTP on**，归档真实 cache dtype/量化元数据和有效配置。保留已批准启动参数，模块位置改变时按 [映射](baseline/namespace-migration.json) 调整显式 Python 路径；CLI/model/connector 标识不改。v1 为现有主路线，v2 也已原生化，但不得将单独导入成功视为 v2 的 GLM/DSA 认证。非目标模型、310P 等不纳入首批认证；LoRA/其他共享功能未借本次迁移删除。

## 5. 结果交接和回退

每项记录配对提交、安装模式、制品哈希、用例/命令、配置、退出码、通过/失败/未执行及证据位置。原始日志、trace、请求数据内网保留，人工审核后交接允许外发的最小结果；验证环境无需部署 AI 客户端或远程 Agent。

本次不修复 [P1/基线已知问题](../p1/results/log-review-20260927/README.md)。后续从保留 `p1` 新建复测/修复 worktree，以共同输入和修复提交区分历史问题与 P2 引入问题，再将确需修复的提交移植到 `p2`。运行回退使用上一套配对制品、配置和兼容缓存 namespace，遵循既有在途传输和成对重启要求，不混用 P1/P2 Python 或原生库。
