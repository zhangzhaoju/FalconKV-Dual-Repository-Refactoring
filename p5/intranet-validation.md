# P5 P6 联合内网验收指导

仅在最终 `p6` 配对上进行一次功能/性能验收；不用先在 `p5` 重跑。
`p6` 完整继承 P5，唯一安装输入为 [P6 pair.json](../p6/baseline/pair.json)。
两包版本分别为 `0.18.0+ascend.p5p6rc1`、`0.4.3+ascend.p5p6rc1`。
native-layout 冻结配对作为独立对照环境保留，不在其运行中的 editable checkout 切分支。

本轮若沿用基线 strict editable 安装与原启动自动化，可直接执行两仓 README 指向的
`vllm/docs/design/baseline-validation.md` 或
`LMCache/docs/source/getting_started/baseline_validation.rst`。
两份内部指南包含相同的成对安装命令，使用本页同一 P6 配对清单；
不要求先切换 wheel/镜像才能开始功能性能对照。

## 发布与内网准备

本轮只创建本地提交和标签，没有自动推送。需要发布时，在源码工作站非强制推送两仓：

```bash
cd /home/zhangzhaoju/Learn/code/falconkv || exit 1
git -C p1-repos/vllm push origin p5 p6 refs/tags/native-layout-frozen-20261005
git -C p1-repos/LMCache push origin p5 p6 refs/tags/native-layout-frozen-20261005
```

Actions 保持开启；若另一仓对应分支尚未发布或私有仓只读授权未配置，首次 CI 可能失败，
待两仓发布完成后重跑。CI 仅作 host 检查，不代表 NPU 验收。
上传整个 `design/p5`、`design/p6`、`design/p1/tools/check_pip_dependencies.py` 和
`design/p4/tools/check_pair.py` 到 `/workspace/zzj/design`，不要上传模型/凭据到外部服务。

使用与基线相同基础依赖、无旧四包的独立测试容器。torch 不重装。
四节点每个实际运行容器均安装两仓并留存报告；同镜像不能替代逐容器安装身份核对。
以下路径继续使用 `p1-repos`，仅适用于该专用容器里的新目录；若已有目录，不覆盖，
可将 `P56_REPOS` 改成新的 `/workspace/zzj/p56-repos`。所有输出目录必须是本次新目录。

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
P56_PY=/usr/local/python3.11.14/bin/python
P56_DESIGN=/workspace/zzj/design
P56_REPOS=/workspace/zzj/p1-repos
P56_PAIR="$P56_DESIGN/p6/baseline/pair.json"
mkdir -p /workspace/zzj/p56-check
P56_REPORT=$(mktemp -d /workspace/zzj/p56-check/run.XXXXXXXX)
test ! -e "$P56_REPOS/vllm"
test ! -e "$P56_REPOS/LMCache"
mkdir -p "$P56_REPOS"
git clone --branch p6 --single-branch git@github.com:zhangzhaoju/vllm-dual.git "$P56_REPOS/vllm"
git clone --branch p6 --single-branch git@github.com:zhangzhaoju/LMCache-dual.git "$P56_REPOS/LMCache"
"$P56_PY" -B "$P56_DESIGN/p5/tools/qualify.py" source \
  --workspace "$P56_REPOS" --pair "$P56_PAIR" --output "$P56_REPORT/source.json"
```

源码门槛要求精确 HEAD/tree、候选版本、干净跟踪文件及无额外未跟踪文件，仅允许生成的
`submodule-materials.json`。分支头已变化时停止，先更新经审核的配对清单，不跳过检查。
这里不是只根据 `p6` 分支名或包版本推断安装正确。

## 材料和环境

```bash
git -C "$P56_REPOS/vllm" submodule update --init -- csrc/third_party/catlass
git -C "$P56_REPOS/LMCache" submodule update --init -- third_party/kvcache-ops
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.1
set +u
source "$ASCEND_HOME_PATH/set_env.sh"
set -u
export SOC_VERSION=ascend910b3
for P56_NAME in vllm LMCache; do
  (cd "$P56_REPOS/$P56_NAME" && \
    "$P56_PY" -B setup.py sdist --dist-dir "$P56_REPORT/sdist-$P56_NAME") \
    2>&1 | tee "$P56_REPORT/sdist-$P56_NAME.log"
  "$P56_PY" -B "$P56_REPOS/$P56_NAME/tools/check_native_layout.py"
done
"$P56_PY" -B "$P56_DESIGN/p1/tools/check_pip_dependencies.py" --output "$P56_REPORT/pip-before"
```

允许通过已审核的代理或本地 Git 材料准备固定依赖，不配置自动升级。
材料 pin 不变：CATLASS `716fd7baa7fb7f6cac0488bb628fd1dd0e875641`、kvcache-ops
`9f18d2339bc58a43429f7d5bdaef1628c820eff5`。缺材料不能用 `--source-only` 通过构建门槛。
基础环境固定 Python 3.11/aarch64、910B3、CANN 8.5.1、torch 2.9.0、torch-npu 2.9.0.post2、
Transformers 5.2.0、triton-ascend 3.2.0.dev20260322。pip 检查仅豁免已批准的三个 stdlib 错报。

材料由根 `setup.py` 在 sdist/编译时自动核验并登记，不下载材料、不覆盖漂移清单；
已移除阶段构建/安装包装脚本。直接使用 `pip install -e .`、
`python setup.py bdist_wheel` 和 `python setup.py sdist`。
下方 pip 参数使内网沿用现有依赖；裸 pip 默认仍可能创建隔离环境并解析下载依赖。

## 开发态调测

```bash
cd /tmp || exit 1
for P56_NAME in vllm LMCache; do
  (cd "$P56_REPOS/$P56_NAME" && \
    "$P56_PY" -B -m pip install -e . \
      --no-build-isolation --no-deps --no-index --force-reinstall) \
    2>&1 | tee "$P56_REPORT/editable-$P56_NAME.log"
done
for P56_NAME in vllm LMCache; do
  "$P56_PY" -B "$P56_REPOS/$P56_NAME/tools/check_native_layout.py" --installed editable \
    | tee "$P56_REPORT/installed-$P56_NAME.json"
done
"$P56_PY" -B "$P56_DESIGN/p4/tools/check_pair.py" --workspace "$P56_REPOS" \
  --pair "$P56_PAIR" --output "$P56_REPORT/installed-pair.json"
"$P56_PY" -B "$P56_DESIGN/p1/tools/check_pip_dependencies.py" --output "$P56_REPORT/pip-after"
"$P56_PY" -B "$P56_REPOS/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --check-lmcache --output "$P56_REPORT/bootstrap"
"$P56_PY" -B "$P56_REPOS/vllm/tools/validate_npu_native.py" \
  --spawn --output "$P56_REPORT/native-import.json"
"$P56_PY" -B "$P56_REPOS/LMCache/tools/p4_runtime_smoke.py" --output "$P56_REPORT/lmcache-import"
```

两个版本都变化，因此必须重建/重装，不能沿用 layout1 链接树和 `.so`。
标准 pip 不会创建专用环境或清除旧插件，旧包冲突应另备干净容器。
保留 `build/p1-native/run-*` 和 `build/__editable__.*`，不能清理正在使用的资源。
在分配的空闲设备上另跑 `validate_npu_native.py --spawn --device-smoke` 和
`p4_runtime_smoke.py --npu`，均使用新的输出路径；未指定设备 smoke 的导入检查不是 NPU 通过。

## 双包发布与镜像

基线对照可继续使用原 strict editable 环境及启动命令，先在同一安装方式下确认功能性能。
下文属于正式发布制品验证，不是本轮对比的安装前置。editable 结果要记录安装模式，
不能冒充 wheel/镜像证据；正式发布仍需验证最终 wheel/镜像和长稳，已完成的用例是否
可复用由验收负责人按来源和安装模式差异审核，不因此省略制品相关检查。
一台专用内网构建容器产出两仓制品，其余节点安装同一份制品，无需四台重复编译。

```bash
"$P56_PY" -B "$P56_DESIGN/p5/tools/qualify.py" build \
  --workspace "$P56_REPOS" --pair "$P56_PAIR" --output "$P56_REPORT/release-build"
```

该编排通过标准 `setup.py sdist`、`setup.py bdist_wheel` 完成打包，
依次核对源码、pip、材料，生成两仓 wheel/sdist，在新目录解包 sdist 再生成 wheel，
保存日志/命令/退出码/哈希。只支持内网 Python 3.11/aarch64；不会安装框架或隐式下载。
基础依赖前后只读记录版本和路径，原生 ABI 由后续设备检查证明，不混为同一种证据。
失败保留全部输出，修复后用新的输出目录；只有最后的 `artifacts.json` 表示本轮构建编排成功。

需要普通 wheel 安装时，在干净测试容器中使用本次精确路径，四节点重复安装/检查：

```bash
"$P56_PY" -B -m pip install --no-index --no-deps --force-reinstall \
  "$P56_REPORT/release-build/wheel-vllm/wheels/vllm-0.18.0+ascend.p5p6rc1-cp311-cp311-linux_aarch64.whl" \
  "$P56_REPORT/release-build/wheel-LMCache/wheels/lmcache-0.4.3+ascend.p5p6rc1-cp311-cp311-linux_aarch64.whl" \
  2>&1 | tee "$P56_REPORT/install-wheels.log"
```

跨容器复制 `release-build/` 时保留其结构并核对 `artifacts.json` 的相对路径/sha256；
同时设置该容器的 `P56_REPORT` 指向包含此目录的位置。安装后重复上节路径检查（改为
`--installed wheel`）、check_pair、pip 与真实导入/设备检查，输出路径重新命名，不覆盖 editable 报告。

镜像基础层由内网负责人提供固定 digest，必须已经预加载且无旧框架；不得用 `latest`。
以下命令要求事先设置 `P56_BASE_IMAGE` 为真实的 `仓库名@sha256:64位摘要`：

```bash
: "${P56_BASE_IMAGE:?请先设置经审核且已预加载的基础镜像 digest}"
docker image inspect "$P56_BASE_IMAGE" > "$P56_REPORT/base-image.json"
"$P56_PY" -B "$P56_DESIGN/p5/tools/prepare_image.py" --pair "$P56_PAIR" \
  --artifacts "$P56_REPORT/release-build/artifacts.json" --base-image "$P56_BASE_IMAGE" \
  --output "$P56_REPORT/image-context"
docker build --pull=false --network=none --build-arg "BASE_IMAGE=$P56_BASE_IMAGE" \
  --build-arg "PYTHON=$P56_PY" --iidfile "$P56_REPORT/image.id" \
  -t ascend-p56:rc1 "$P56_REPORT/image-context"
docker image inspect ascend-p56:rc1 > "$P56_REPORT/candidate-image.json"
```

上下文只携带精确双 wheel 和安装核对脚本，制品不包含原四仓/模型/备份。
`--network=none` 不替代镜像获取及出站策略审查；RUN 不联网，基础层获取仍由内网策略约束。
镜像默认不启动服务；通过现有已审查的 Ascend 设备挂载与 CANN 环境启动测试容器。
镜像 `installed.json` 只证明安装元数据，不替代各节点 NPU/模型检查。

## 一次联合验证

保持已验证的启动配置和 proxy 新路径 `vllm/examples/disaggregated_prefill_v1/`，不恢复根
`ascend/`。使用 `LMCacheConnectorV1`，GLM-5.2、DSA 双组/MTP 开启、C8 关闭。
冻结模型身份、有效启动参数、依赖和镜像 digest，别只记录脚本注释。

1. 执行 [A01 至 A15](../03-baseline-and-validation.md) 的安装、原生 ABI、负向边界与网络检查。
2. 执行 B01 至 B11：离线/在线、无 LMCache 与缓存模式、文本/reasoning/tool call、错误输入、
   greedy 对照、DSA/MTP 与图实际 replay。TP8/DP2、TP4/DP4 分别记录。
3. 执行 C01 至 C13：冷/热/部分 prefix、CPU/NPU round trip、不等 KV 组、RemoteFill、
   store-before-free、checkpoint/preemption、通信失败、非输出 rank 失败与成对重启。
   可复用 LMCache `tests/v1/test_glm52_group_cardinality.py`、
   `test_sparse_mla_dsa_kv_roundtrip.py`、`test_remote_fill_qualification.py` 及各仓已有设备测试，
   不把 host/mock 用例当成实际故障和数值验证。
4. 固定负载做两种拓扑、冷/热各至少三轮，预热与测量窗口分离；记录 TTFT/TPOT/ITL、
   吞吐、MTP 接收率、图 replay、CPU/HBM/共享内存及末尾排空等待。长稳建议每种拓扑至少 8 小时，
   具体门槛在测试前确认，覆盖并发、取消、缓存命中和抢占。
5. 同一批次执行 [P6 备份与成对恢复演练](../p6/README.md)，不在在线环境做故障注入或回退。

旧 native-layout 相对 P4 的尾延迟问题保留为待办。即使 P6 与 layout1 一致，也不能据此
把历史问题标为已解决；需用同负载重测并归档结论，必要时额外使用独立冻结 P4 对照。

## 性能数据格式与核对

保留原始客户端/服务端日志，并由内网采集程序生成规范化 JSON。不要将旧 JSON 的 `duration`
直接改名为墙钟，也不要从中断报告补造完成时间或 token 间隔。
顶层字段为 `schema_version=1`、`pair_sha256`（对应 pair.json 文件哈希）、`profile_sha256`
（模型/有效配置/硬件/依赖/采样，不含待比较的源码版本）、`workload_sha256`、
`topology`（`TP8/DP2` 或 `TP4/DP4`）、`cache_phase`（`cold` 或 `warm`）及 `runs`。

每个 run 记录相同的非空 `round` 标识、`completed=true`、`interrupted=false`、
`expected_requests`、同一客户端时钟的 `window_start_monotonic_ns` 与
`window_end_monotonic_ns`（从发出第一条到全部完成，含排空时间），以及 `requests`。
每条请求包括唯一 `id`、`status="ok"`、`input_sha256`、`input_tokens`、`output_tokens`、
`ttft_ms`、`e2e_ms`、`itl_ms` 数组。性能负载要求至少 2 个输出 token；单 token/错误响应另做功能用例。
`itl_ms` 是逐 token 间隔，不能把含多 token 的 SSE chunk 到达时间冒充逐 token 数据；
若采集端不能提供，保留原始数据并补齐相应采集，工具拒绝猜测。

每轮、每版本必须有相同完整请求输入集合，不能只保留成功交集。输出 token 序列正确性
由 B 类用例验证，随机输出长度与分布差异仍需人工分析。工具计算有效请求数和输出 token
数除以墙钟，输出逐轮指标和轮次中位数，P99 不会被隐藏在均值中。

例如，负责人事先确认采用 5% 门槛后，对一组拓扑/温度执行；不同组分别生成报告：

```bash
"$P56_PY" -B "$P56_DESIGN/p5/tools/compare_perf.py" \
  --baseline "$P56_REPORT/perf-tp8-cold-baseline.json" \
  --candidate "$P56_REPORT/perf-tp8-cold-candidate.json" \
  --baseline-pair "$P56_DESIGN/p5/baseline/native-layout-frozen-20261005.json" \
  --pair "$P56_PAIR" --max-regression-pct 5 \
  --output "$P56_REPORT/perf-tp8-cold-comparison.json"
```

5% 是原方案建议值，不是本次新批准的验收结论。工具对 p50/p95/p99 一并报告并应用传入门槛，
其口径须在采集前确认；不能为了通过而改阈值、关闭 MTP/DSA 或删除慢请求。

## 汇总与签核

```bash
"$P56_PY" -B "$P56_DESIGN/p5/tools/qualify.py" init --pair "$P56_PAIR" \
  --output "$P56_REPORT/qualification.json"
```

初始 45 项均为 pending。完成后由负责人逐项填写 `status`、`summary`、`reviewer` 与
`evidence`，每个 evidence 为 `{"path":"报告相对路径","sha256":"真实文件哈希"}`。
原矩阵覆盖范围不缩减，PERF 两项各附冷/热三轮对照，SOAK 两项附完整长稳，IMAGE 附基础层和
制品/镜像证明，RESTORE 附源码与模型/配置/环境恢复。不能只复制一个通用 PASS 文本填满表格。

```bash
"$P56_PY" -B "$P56_DESIGN/p5/tools/qualify.py" review --pair "$P56_PAIR" \
  --campaign "$P56_REPORT/qualification.json" --evidence-root "$P56_REPORT" \
  --output "$P56_REPORT/qualification-review.json"
```

该检查只核对覆盖、负责人声明和哈希，不自动证明日志语义，不实施生产切换。
性能比较不通过、恢复材料缺失或任何用例 pending 都不能签核完成。正式切换/原仓归档按
P6 指导另行执行；本次只准备流程。原始证据留内网，审核脱敏后再反馈。
