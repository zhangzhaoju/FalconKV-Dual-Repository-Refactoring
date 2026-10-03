# P2 P3 NPU 导入和 DSA KV 绑定修复与复测

更新：2026-09-30。本页交付包含两项修复：GLM 模型检查子进程的 `current_platform` 循环导入，以及初始化 DSA 双组缓存时 `bind_kv_cache` 抛出 `NotImplementedError`。新提交已包含上一轮导入修复，并同步到 P3；本次未推送 GitHub、未构建或操作内网服务。

## 1. 原因和修复范围

registry 的新 Python 子进程先导入 `model_executor` 的依赖链，触发 NPU 平台解析。
原 `platforms/npu.py` 在类定义完成之前导入 `config.ascend` 和 `utils.ascend`，
后者又经 `config.compilation` 请求尚未创建的 `current_platform`，形成循环。
这与上一轮自动化脚本误切到 P1 是两个不同的问题；本次不是 GLM 架构不支持或 TP/DP 配置错误。

修复将三项常量移到无依赖的 `platforms/ascend_constants.py`，保留原 utils 导出；
配置和重型工具函数改为使用时导入。保留正常 `torch_npu` 导入以注册 NPU 设备类型。
不预填假平台、不跳过模型检查、不修改已安装第三方包，也不更改 DSA/MTP/C8、量化或通信算法。

9 月 30 日新增的 KV 绑定修复针对一次迁移遗漏：旧 `patch_qwen3_next_mtp.py` 实际含有 GLM DSA 所需的通用逻辑，却被误归为非目标补丁并连同测试归档。原生 `bind_kv_cache` 现保留 NPU 的原绑定语义：按层号排序；精确 latent/indexer 兄弟对按 latent → indexer 排列；其他重复层号保留首项；所有 forward context 条目仍引用原对象。MTP 前缀、tuple cache 和跨层共享引用均保留，不复制张量、不恢复全局 patch。

只向旧判断添加 `is_npu()` 不够，因为无法保证反向输入时两组缓存的顺序。无需关闭 DSA/MTP 或更改 TP/DP；websockets 弃用与 DeepGEMM 警告不是该 `NotImplementedError` 的直接原因。

## 2. 准确配对身份

冻结的 `p2` 不前移，**P2 复测必须改用修复分支/提交，不能继续检出旧 `p2`**。
两套环境和源码 checkout 分开；不要在运行中的 editable 服务下切换源码。

| 验证阶段 | vLLM 分支与提交 | 配对 LMCache 提交（本次未修改） |
| --- | --- | --- |
| P2 修复 | `fix/p2-npu-bootstrap`：`e1b35d4f12fdc0cd975b8e990e146507c459d094` | `p2`：`cfe8a1754db743d41c8bb63f8d02ad7c3051948c` |
| P3 | `p3`：`7854ce2158f42a725c58cd1026b388709f320b2b` | `p3`：`a4e2131e890727edadb6bb61cde369900f72c3fd` |

精确 tree、版本与检查范围见 [P2 修复清单](baseline/kv-cache-binding-fix-20260930.json)
和 [P3 修复清单](../p3/baseline/kv-cache-binding-fix-20260930.json)。包版本不变：
P2 为 `vllm 0.18.0+ascend.p2` / `lmcache 0.4.3+ascend.p1`；P3 两包均为 `+ascend.p3`。
因此不能仅根据 `pip show` 或包版本判断修复是否生效。

若选择通过 GitHub 交接，由源码机执行（本次未代为执行；不要强推）：

```bash
git -C p1-repos/vllm remote get-url origin
git -C p1-repos/LMCache remote get-url origin
# 确认是自己的 vllm-dual / LMCache-dual 后执行。
git -C p1-repos/vllm push origin fix/p2-npu-bootstrap p3
# LMCache 没有新增提交；仅在此前未交接相应分支时执行。
git -C p1-repos/LMCache push origin p2 p3
```

### P2 内网同步

仅在专用验证容器执行。先确认没有进程使用该 checkout，并保留日志、材料和本地改动。
下面使用准确 SHA 的 detached checkout；这是有意的，不是切回 P1。
自动化脚本也要固定相同 SHA，并在安装前和启动前再次校验。

```bash
set -euo pipefail
export P2_ROOT=/workspace/zzj
export P2_REPOS="$P2_ROOT/p1-repos"
export P2_FIX=e1b35d4f12fdc0cd975b8e990e146507c459d094
export P2_LM=cfe8a1754db743d41c8bb63f8d02ad7c3051948c
git -C "$P2_REPOS/vllm" status --short --branch
git -C "$P2_REPOS/LMCache" status --short --branch
git -C "$P2_REPOS/vllm" diff --quiet
git -C "$P2_REPOS/vllm" diff --cached --quiet
git -C "$P2_REPOS/LMCache" diff --quiet
git -C "$P2_REPOS/LMCache" diff --cached --quiet
git -C "$P2_REPOS/vllm" fetch origin fix/p2-npu-bootstrap
git -C "$P2_REPOS/LMCache" fetch origin p2
git -C "$P2_REPOS/vllm" switch --detach "$P2_FIX"
git -C "$P2_REPOS/LMCache" switch --detach "$P2_LM"
test "$(git -C "$P2_REPOS/vllm" rev-parse HEAD)" = "$P2_FIX"
test "$(git -C "$P2_REPOS/LMCache" rev-parse HEAD)" = "$P2_LM"
```

有冲突即停止，不执行 `reset --hard` / `clean`，不删除原始日志或子模块材料。
P3 按 [P3 指导](../p3/intranet-validation.md) 在另一环境同步，并核对上表 SHA。

## 3. 重装及复测

9 月 29 日导入修复增加了 Python 模块，尚未刷新过链接树的 strict editable 必须重装；9 月 30 日的运行代码仅修改已有 Python 文件。本页继续使用原安装流程重新生成安装及来源记录，不让旧同版本安装混入本次验收。
构建脚本及其完整原生构建流程不变，不手工复用其他分支的 `.so`。
普通 wheel 用户按 [P2 构建/安装步骤](intranet-validation.md#2-构建安装和资源检查)
重新构建并安装 vLLM；不能用旧同版本 wheel。

以下为 P2 开发态示例。先按 [P2 环境步骤](intranet-validation.md#1-环境和源码)
初始化 CANN/SOC/构建变量并准备固定材料。复用既有 torch 2.9.0+cpu、torch_npu 2.9.0.post2、
Transformers 5.2.0，不安装或升级这些依赖。`--isolated-env` 是操作者的隔离确认，不会创建环境。

```bash
mkdir -p "$P2_ROOT/p2-check"
P2_RUN=$(mktemp -d "$P2_ROOT/p2-check/npu-bootstrap.XXXXXXXX")
export P2_RUN
cd "$P2_RUN"
python -B "$P2_REPOS/vllm/p1_dev.py" doctor --output "$P2_RUN/vllm-doctor.json"
python -B "$P2_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P2_RUN/pip-before"
python -B "$P2_REPOS/vllm/p1_dev.py" editable --isolated-env --output "$P2_RUN/vllm-editable"
python -B "$P2_REPOS/vllm/p1_dev.py" verify --mode editable --output "$P2_RUN/vllm-paths.json"
python -B "$P2_ROOT/design/p1/tools/check_pip_dependencies.py" --output "$P2_RUN/pip-after"
```

LMCache 若已有可追溯的上表 P2 配对安装，本次不要求重建；否则按原指南重新构建安装配对版本。
在源码外运行，去除旧 PYTHONPATH/.pth 污染，保留安装后的 checkout 和 strict editable 构建资源。

```bash
cd "$P2_RUN"
test "$(git -C "$P2_REPOS/vllm" rev-parse HEAD)" = "$P2_FIX"
test "$(git -C "$P2_REPOS/LMCache" rev-parse HEAD)" = "$P2_LM"
python -B "$P2_REPOS/vllm/tools/check_npu_bootstrap.py" \
  --inspect-glm --output "$P2_RUN/bootstrap"
python -B "$P2_REPOS/vllm/tools/validate_npu_native.py" \
  --spawn --torchair-abi --output "$P2_RUN/native-import.json"
python -B -m vllm.entrypoints.cli.main serve --help > "$P2_RUN/serve-help.log" 2>&1
```

本批检查预期 **7 项 PASS**：平台、配置、registry、Ascend utils 四种冷导入顺序，
真实 registry 子进程协议、`kv_cache_bind` 绑定检查，以及不使用 model-info 缓存的 GLM 模型类检查。
`kv_cache_bind` 使用安装后的实际函数和少量显式 CPU tensor view，检查正反输入顺序、普通/MTP 层、consumer、单缓存和空输入共六个场景；只证明绑定顺序和引用，不分配 NPU 缓存或执行设备内核。
工具不加载权重或运行推理；正常导入可加载原生库并生成既有 profiling 配置。
不需要手动删除 model-info 缓存，也不要空输入直接执行 `python -m ...registry`。
工具只在自身子进程禁用可选插件；默认单项超时 120 秒，失败返回非零并保留 traceback。
检查会核对包版本和七个导入/绑定源码文件（含 `v1/worker/utils.py`），不替代全部源码/制品身份或 ABI 检查。旧工具仅六项通过不能替代新 `kv_cache_bind` 检查。

每个参与 2P2D 的节点都必须使用相同修复制品/配对身份并通过上述安装后检查。
通过后再按原审核配置启动专用测试服务：DSA 双组/MTP 开启、C8 关闭，先原失败用例，再 TP8/DP2、TP4/DP4。
本次修复不要求改变启动参数。若再失败，归档 `bootstrap/report.json`、对应 `.log`、
两仓 HEAD、安装路径报告及完整服务 traceback，不通过预导入平台或关闭特性绕过。

## 4. 已完成和待验证

- P2 最新 [主机日志](results/kv-cache-binding-20260930/p2-final-host.log) 和 [JUnit](results/kv-cache-binding-20260930/p2-final-host.xml)：187 项通过、311 个 subtest 通过；新增 12 项绑定回归，修复前已复现同一 `NotImplementedError`。
- P3 最新 [配对报告](../p3/results/kv-cache-binding-20260930/final/verification.json)：409 项通过（vLLM 196、LMCache 115、RemoteFill 98），为提交后的源码/主机验证；本次四个修复文件在两个分支逐字一致。
- 9 月 29 日 [P2](results/npu-bootstrap-20260929/README.md) / [P3](../p3/results/npu-bootstrap-20260929/README.md) 结果保留为历史，不能证明本次 KV 绑定已通过。
- 本机未安装 torch/CANN、构建框架、连接 NPU 或验证 2P2D；上述源码/host 结果不能视为阶段运行验收通过。
