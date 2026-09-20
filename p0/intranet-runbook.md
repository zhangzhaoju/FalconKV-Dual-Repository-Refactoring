# P0 内网执行与人工反馈

适用范围：内网人员/CI 执行；不接入外部大模型/Agent，不自动外发日志。以下命令本轮未在内网执行。当前仍是四仓 P0 基线，不是已经合并好的双仓。

## 1. 先检查来源与材料

对照 `baseline/source-manifest.json` 核对四仓原始 commit/tree 和 bundle SHA-256。将四个仓库恢复到并列目录，保留现有环境/部署，不覆盖正在使用的目录。P0 差异位于 `changes/`，只适用于清单中的原始 commit；先用 `git apply --check` 核对再由内网人员应用，已有改动则先人工合并，不强制覆盖。

两个子模块当前缺少源码：按 `review/submodule-materials.json` 的 URL 和 gitlink commit，通过内网源或明确配置的代理补齐，并核对实际 HEAD。不要使用 `.gitmodules` 的分支最新 HEAD 代替固定 commit，也不要认为父仓 bundle 自动包含子模块对象。补齐后记录源码 hash、嵌套子模块/LFS/第三方资源和版本元数据。

公网资源可经代理获取；Python/CMake/Git/容器各工具分别核对代理、企业 CA 和固定来源。回环、模型 API、P/D、缓存及 HCCL 等内网通信按实际拓扑直连，不走公网代理。不要在源码、shell 历史或反馈材料中包含代理口令；不关闭 TLS 校验。

## 2. 采集环境与模型元数据

使用现有已激活的目标 Python 环境（torch 2.9.0 已安装），不新建环境、不安装 torch，不使用会隐式同步依赖的执行命令。仅 `collect_intranet_env.py` 可以独立复制到内网，它只使用标准库（Python 3.10+）。其他本机审计工具使用本机现有 Python 3.12。

在内网工作区根目录运行，替换模型路径占位符：

```bash
python -B design/p0/tools/collect_intranet_env.py \
  --output ./p0-intranet-results/env-P0.json \
  --model 'glm52=/实际本地模型目录' \
  --probe-runtime --probe-npu
```

四个节点分别采集，输出名称改为 `env-P0.json`、`env-P1.json`、`env-D0.json`、`env-D1.json`，并记录节点角色。该命令不联网、不安装、不构建、不加载权重、不执行 remote code；显式的两个 probe 开关会导入已安装 torch/torch_npu、查询设备可用性并运行 `npu-smi info`。不带 probe 时只读包元数据，不能视为运行时验证。

报告默认权限为 0600、拒绝覆盖同名文件；代理只记录是否配置，不记录地址/凭据。报告仍可能含本地路径、设备信息和工具输出，属于内网原件，不能未经审核直接外发。软件元数据匹配不等于 CANN/驱动/扩展 ABI 通过。

另请内网补充这些字段（允许使用代号）：

- GLM-5.2 权重 revision/校验清单、`architectures`、`model_type`、config/tokenizer/量化描述 hash。脚本不读取大权重内容，完整权重身份需由现有制品清单提供。
- W4A8C8 的量化提供方、逐层量化/回退规则、MTP 权重格式；C8 是哪些 latent/index 层、数据与 scale 的 dtype/shape/stride/布局。
- Python/CPU 架构/OS、CANN/driver/firmware、编译器、Ascend kernels、triton-ascend、transformers、compressed-tensors、通信 SDK 和镜像 digest。
- P0/P1/D0/D1 的 TP8/DP2 进程组/路由关系，EP/PP/CP 设置，HCCL 集合通信与缓存传输分别选用的通道和 SDK。
- 内网代理与出站策略已检查的结论；无需向外部模型服务发请求来证明其被禁止。

## 3. 依赖核对与构建前置门槛

`constraints-intranet.txt` 只是现有目标版本的约束，不是安装清单。不要执行原始 CUDA/XPU/ROCm 或完整多模型测试 requirements 来补齐环境。当前至少存在已知 OpenCV 版本区间冲突以及 LMCache 的非目标设备依赖，详见 `findings.md`；先核对现有包和目标导入闭包，再形成独立修复，不靠一次 `pip install -U` 解决。

P0 已修正的是 build-system 的 torch 声明，不是已经解决全部运行依赖或完成 ABI 适配。`--no-build-isolation` 和 `--no-deps` 用于保护已准备的环境，不能使错误的依赖元数据变正确。须记录 `pip check` 发现的问题并按目标 profile 处置。

材料齐备后的 P0 构建配置沿用当前四仓策略：vLLM 主体为 `VLLM_TARGET_DEVICE=empty`，LMCache 主体为 `NO_CUDA_EXT=1`，两个 Ascend 包使用各自现有 NPU 构建入口。这里的 empty/NO_CUDA_EXT 是合仓前基线策略，不是最终原生双仓架构。先核对必要 host/Mooncake 扩展是否齐备，不能用空扩展安装冒充完整缓存能力。

构建和安装由内网执行，显式非隔离构建、不自动替换 torch；所有产物记录源 commit + P0 patch SHA-256、编译参数、SOC、C++ ABI、动态库、wheel/sdist hash。必要的固定依赖可以经代理获取。构建前后比对 torch 的版本/路径/构建信息；不得因编译问题改成 2.10 或关闭必保功能。当前尚未提供可直接投产的启动命令，因为实际模型量化/通道/环境尚未冻结。

## 4. 先重跑 host 检查，再采集 NPU 基线

当前四仓仍保留在并列目录时，可在内网重复同一 host 子集：

```bash
python -B design/p0/tools/run_host_checks.py \
  --workspace . --output ./p0-intranet-results/host-baseline
```

依赖现有 pytest/numpy，初始化检查还间接依赖 torch。四仓在独立进程中执行；关闭项目 conftest 与插件自动加载，仅核对源码提取/mock 逻辑。本机版本记录是辅助证据，不能替代目标环境结果。还需按 `03-baseline-and-validation.md` 和实际依赖环境运行 GLM52 分组、shared indexer、NPU round trip、RemoteFill、HCCL/HIXL/hcomm 等集成测试。

NPU/模型基线按以下顺序冻结，每项对应 `support-matrix.json` 中的场景编号：

1. TP8 主模型加载，核对实际 W4A8/C8/MTP 权重格式和内存；无缓存离线/在线文本推理、stream/cancel、parser。
2. 固定单副本输入和采样，分别记录 MTP 开/关、Eager/实际 ACL replay、DSA/shared indexer/不等 KV 组行为。C8 data 与 scale 都进入搬运和恢复检查。
3. 开启 CPU KV/共享缓存，冷/热/部分命中与无缓存对照，检查实际加载字节、store 完成、释放和输出，不以重算掩盖缓存失效。
4. 扩展到完整 2P2D/TP8/DP2，覆盖两个 P 和两个 D 的路由与跨实例传输；单独确认 HCCL 集合通信和所选缓存通道。
5. checkpoint、抢占、取消、超时、目的端重启及 paired restart；必须覆盖非输出 TP rank 和目的地址代际。
6. 同配置固定预热、并发、输入/输出长度、缓存温度，至少 3 次重复；采集吞吐、TTFT/TPOT/ITL、p95、HBM/RSS/共享内存、图 replay、缓存字节/命中和异常。8 小时长稳按审核设计执行。

负载建议先使用可公开的合成文本，greedy、seed=0、固定输出上限；短/中/长上下文和并发值在实际权重可运行范围内冻结。将 tokenized 输入、采样、输出上限、模型 hash、MTP 候选宽度、图模式及并行/缓存配置留存在内网，随后基线与候选复用同一份材料。不能事后放宽数值或性能门槛。

## 5. 人工反馈与下一批修复

反馈最少包含：批次号、四仓 commit/P0 patch hash、环境是否匹配、模型配置关键字段及 hash、32 卡拓扑解释是否正确、用例编号、通过/失败/未执行、错误类型与最小堆栈、必要指标。

原始命令、完整日志/trace、业务输入输出、凭据和敏感地址留在内网。仅由人员审核脱敏后提供内部策略允许外发的部分；不可外发的材料在内网分析后反馈结论。脚本不会上传报告，当前会话不会直接登录验证环境。P0 未获得模型/缓存/恢复/性能对照和补丁依赖核对结果前保持未完成。
