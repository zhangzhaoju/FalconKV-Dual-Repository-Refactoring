# 当前代码依据、迁移重点和验收标准

调研日期：2026-09-17。本文区分本轮静态观察与未来执行验收。本轮只读取源码、配置和 Git 状态，未安装依赖、编译扩展、启动服务或运行 NPU 测试。

模型范围确认（2026-09-17）：用户明确仅保留 DeepSeek / GLM 原生文本生成及已有 MTP、DSA，删除多模态和基于 Qwen/Llama 的蒸馏模型。以下验收按此范围更新，代码裁剪尚未执行。

必保能力确认（2026-09-17）：保留当前分支目标模型相关能力，明确包括 DSA、CPU KV 卸载、跨实例缓存和 Prefill/Decode 分离；相关索引共享、分组、checkpoint、RemoteFill 及恢复路径进入回归验收。

首批验收环境确认（2026-09-17）：Ascend 910B3、CANN 8.5.1、torch 2.9.0、torch_npu 2.9.0。这里只记录用户指定目标，尚无该组合的本次构建或 NPU 测试结果；具体模型权重、量化、卡数/拓扑及其余环境信息仍待补齐。

构建方式确认（2026-09-17）：构建在无互联网连接的局域网中执行，本机不构建；用户确认内网 torch 已安装 2.9.0。后续复用该环境，源码构建声明中的 2.10.0 需修改，但不要求重装 torch。本文构建、安装和 NPU 验收证据均应来自内网执行环境。

## 1. 当前基线

| 仓库 | 当前分支 | HEAD |
| --- | --- | --- |
| `vllm` | `dsa-two-groups` | `764dea93053f7bfd24fdb78b6ca6996350cb9636` |
| `vllm-ascend` | `sparse` | `326dd77b004a7fc1f393e80e39780adadc4b14cd` |
| `LMCache` | `dev-qzy` | `4353ac9bf0b47f48868e1099103c6417ae9a6b84` |
| `LMCache-Ascend` | `dev-qzy` | `07ebf5c4f7bc622bfce9f3a06fb01c20682ca3aa` |

读取时四仓 `git status --short --branch` 均无工作区改动。当前提交说明包含 GLM 5.2、group-aware cache、RemoteFill 并发 discovery 和 destination sealing 后首次 prepared load 的改动，因此它们必须进入合仓影响分析。

辅助规模统计使用 `rg --files --hidden`，排除 `.git` 和 `__pycache__`，仍遵循 ignore 规则；以下不是严格的 Git tracked 文件数，也不是删除比例预测。

| 仓库 | 扫描可见文件 | Python 文件 | 其他观察 |
| --- | ---: | ---: | --- |
| vllm | 4288 | 2701 | `vllm/model_executor/models/` 下 267 个 Python 文件 |
| vllm-ascend | 1524 | 665 | `patch/` 下 54 个 Python 文件，含初始化模块 |
| LMCache | 1186 | 645 | 同时有 Python、CUDA/C++、Rust 和 Operator |
| LMCache-Ascend | 294 | 192 | 安装 patch 目录 7 个 Python 文件，另有顶层运行时注入 |

静态调研覆盖关键入口和耦合点；完整依赖闭包、全部补丁行为和硬件支持矩阵是实施 P0 的产物，不能把本表当成可直接批量删除的文件清单。

## 2. 关键事实与代码证据

链接指向本地当前源码，重构后以本节 HEAD 和迁移映射回溯；符号比未来变化的行号更稳定。

| 观察 | 证据 | 设计影响 |
| --- | --- | --- |
| Ascend 通过平台和 general plugins 接入 | [vllm-ascend/setup.py](../vllm-ascend/setup.py)，`entry_points`；[包入口](../vllm-ascend/vllm_ascend/__init__.py) | 合仓后改为原生平台和显式注册 |
| NPU 当前被标记为 OOT，启动时应用全局 patch | [platform.py](../vllm-ascend/vllm_ascend/platform.py)，`NPUPlatform` / `pre_register_and_update` | 平台能力与配置逻辑直接合入 |
| NPU 主 Runner 继承 GPU Runner，并替换 CUDA stream/event | [model_runner_v1.py](../vllm-ascend/vllm_ascend/worker/model_runner_v1.py)，`NPUModelRunner` / `_torch_cuda_wrapper` | 先提取继承的有效行为，再删除 GPU 实现 |
| v2 Runner 同样继承 GPU Runner | [worker/v2/model_runner.py](../vllm-ascend/vllm_ascend/worker/v2/model_runner.py) | 不能通过切到 v2 自动解决原生化 |
| 平台和 worker 初始化会导入多个非目标模型 patch | [平台 patch 入口](../vllm-ascend/vllm_ascend/patch/platform/__init__.py)、[worker patch 入口](../vllm-ascend/vllm_ascend/patch/worker/__init__.py) | 删除模型必须同步处理初始化链 |
| GLM 依赖 Llama 完整模型与组件 | [glm.py](../vllm/vllm/model_executor/models/glm.py)、[glm4.py](../vllm/vllm/model_executor/models/glm4.py) | 先抽公共 decoder/MLP，保持权重映射 |
| MTP 使用 Eagle proposer，后者导入 Llama Eagle3 | [spec_decode 入口](../vllm-ascend/vllm_ascend/spec_decode/__init__.py)、[eagle_proposer.py](../vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py) | 不能按 Eagle/Llama 名称直接删除 |
| DeepSeek 与 GLM DSA 共用模型实现，包含结构化共享 indexer | [registry.py](../vllm/vllm/model_executor/models/registry.py)、[deepseek_v2.py](../vllm/vllm/model_executor/models/deepseek_v2.py) | 架构名与文件名非一一对应，保留 shared/full 和权重过滤语义 |
| registry 存在通用 Transformers fallback | [registry.py](../vllm/vllm/model_executor/models/registry.py)，`_try_resolve_transformers` / `resolve_model_cls` | 只删注册项不足以限制模型支持 |
| LMCache-Ascend import 时重建配置类、替换工厂/类和模块 | [lmcache_ascend/__init__.py](../LMCache-Ascend/lmcache_ascend/__init__.py) | 消除 import-order 耦合，逐项合入主体实现 |
| 包含安装后 patch 机制 | [setup.py](../LMCache-Ascend/setup.py)，`run_patches`；[apply_patch.py](../LMCache-Ascend/lmcache_ascend/integration/patch/apply_patch.py) | 构建/安装不得改写已安装的其他包源码；需在 P0 核对实际触发路径 |
| CacheEngine、adapter 都有主体/Ascend 子类叠加 | [Ascend cache_engine.py](../LMCache-Ascend/lmcache_ascend/v1/cache_engine.py)、[Ascend vllm_v1_adapter.py](../LMCache-Ascend/lmcache_ascend/integration/vllm/vllm_v1_adapter.py) | 不能直接覆盖主体文件，必须合并继承契约 |
| 两个 KV 组具有独立层集合 | [metadata.py](../LMCache/lmcache/v1/metadata.py)、[kv_format.py](../LMCache-Ascend/lmcache_ascend/v1/kv_format.py)、[不等组测试](../LMCache-Ascend/tests/v1/test_glm52_group_cardinality.py) | 维持 runtime group layer counts/names；覆盖 79/22 fixture |
| 旧 PD/P2P backend 与 layerwise 不兼容 | [storage_backend/__init__.py](../LMCache-Ascend/lmcache_ascend/v1/storage_backend/__init__.py) | 原限制进入能力矩阵，和 RemoteFill 路径分开验证 |
| RemoteFill 有额外能力/生命周期接口 | [LMCacheAscendConnectorV1Dynamic](../LMCache-Ascend/lmcache_ascend/integration/vllm/lmcache_ascend_connector_v1.py) | 保留 handoff、sealing、placement、metrics、paired restart 等语义 |
| staged SFA 文档记录已实现内容和仍待资格验证内容 | [STAGED_SFA_GRAPH_PRODUCTION.md](../vllm-ascend/vllm_ascend/distributed/kv_transfer/sparse_offload/STAGED_SFA_GRAPH_PRODUCTION.md) | 文档中的历史结果不能代替本次图执行和故障验收 |
| torch 声明不一致 | [vllm/pyproject.toml](../vllm/pyproject.toml)、[LMCache/pyproject.toml](../LMCache/pyproject.toml)、[Ascend requirements](../vllm-ascend/requirements.txt) | 按用户指定 torch/torch_npu 2.9.0 统一构建和运行声明，验证 API/ABI |
| CANN 8.5+ 默认选择 HIXL/hcomm one-sided 构建 | [LMCache-Ascend/setup.py](../LMCache-Ascend/setup.py)，`_is_cann_85_or_later` 和扩展构建流程 | 在无显式覆盖的 8.5.1 目标上核对 SDK/库及 910B3 实测结果，不把构建选择等同于能力验证 |
| LMCache 默认依赖含 cufile/NIXL/nvtx/CuPy CUDA | [requirements/common.txt](../LMCache/requirements/common.txt) | 清理安装元数据，不能只删 Python 分支 |
| 当前构建文件存在公网 wheel 获取和 CMake FetchContent | [vllm/setup.py](../vllm/setup.py)、[CMakeLists.txt](../vllm/CMakeLists.txt)、[external_projects](../vllm/cmake/external_projects) | 删除非目标依赖的下载路径；目标构建链所需材料必须离线齐备，pip 禁索引不能代替完整检查 |
| vLLM 有 RLHF 协作路由和 weight transfer | [RLHF router](../vllm/vllm/entrypoints/serve/rlhf/api_router.py)、[weight_transfer](../vllm/vllm/distributed/weight_transfer) | 按训练协作用途裁剪，保留正常推理权重加载 |

## 3. 模块迁移清单

这是模块级清单，具体文件/符号映射在 P0 补齐；“删除”均以前置依赖已解除和范围审核为条件。

| 来源 | 处理 | 目标/条件 |
| --- | --- | --- |
| vllm-ascend `platform.py` / `ascend_config.py` / `envs.py` | 合并 | vLLM 平台/配置；去插件发现、保留初始化顺序 |
| vllm-ascend `worker/` | 合并/收敛 | 原生 NPU Worker/Runner、input batch 和公共状态 |
| vllm-ascend `attention/` / `ops/` / `compilation/` / `sample/` | 合并 | 对应语义模块，只保留目标模型所需实现 |
| vllm-ascend `spec_decode/` | 提取并合并 | MTP 及批准保留的 draft 路径；清理无关模型导入 |
| vllm-ascend `quantization/` / `csrc/` | 选择性合并 | Ascend 构建、已批准量化和目标算子；保留必要 host glue |
| vllm-ascend `distributed/` / `sparse_offload/` / `live_source_handoff.py` | 合并 | HCCL/并行、推理侧 KV 与事件所有权 |
| vllm-ascend `patch/` | 内联有效行为后删除 | 每项 patch 都有归属和回归证据 |
| vllm-ascend `_310p/` | 首批 910B3 验收之外，是否裁剪待总体范围审核 | 首批认证型号不自动授权删除其他 Ascend 型号 |
| vllm 非 NPU 平台/设备内核/通信 | 提取共用逻辑后删除 | GPU/CPU 计算后端退出；主机服务功能保留 |
| vllm `models/`、config/tokenizer/parser/template 注册 | 白名单裁剪 | 先抽 GLM/MTP 共用依赖，关闭通用模型 fallback |
| LMCache-Ascend 顶层注入 | 逐项内联后删除 | 正式配置、工厂、工具和接口 |
| LMCache-Ascend `v1/cache_engine.py` / `integration/vllm/` | 合并 | 保留主体当前分支的恢复和分组语义 |
| LMCache-Ascend `npu_connector/` / `kv_format.py` / `kv_layer_groups.py` | 合并 | 原生 NPU 搬运、独立组元数据 |
| LMCache-Ascend `remote_fill*` / `local_checkpoint.py` / `preemption_checkpoint.py` | 合并 | 与主体协议和生命周期按职责整合 |
| LMCache-Ascend `transfer_channel/` / `storage_backend/` / `csrc/` | 选择性合并 | 只构建选定 profile 的 NPU 通道和必要内存/序列化实现 |
| LMCache GPU/XPU/HPU Connector、CUDA/HIP/GDS/NIXL | 删除 | 显式 NPU/CPU host 能力已接管 |
| SGLang、MindSpore、CacheBlend、Operator、未使用云后端 | 建议删除，待范围确认 | 属于额外产品面，不能归入“其他设备”一概删除 |
| 两侧测试/示例/CI/文档 | 合并并裁剪 | 覆盖全部必保功能，删除非目标依赖和运行入口 |

## 4. 验收矩阵

### 4.1 静态、构建和入口

| 编号 | 验证 | 通过条件 |
| --- | --- | --- |
| A01 | 仓库/包数量 | 两个活动 Git 仓、两个分发包，无嵌套 Ascend 仓库或独立插件安装要求 |
| A02 | Python import 和动态注册闭包 | 白名单模块全部可导入，动态路径/entry point/包资源无悬空引用 |
| A03 | 原生化 | 生产不依赖 `_ascend` 包、`sys.modules` 重定向、跨包猴子补丁、全局 CUDA API 模拟或安装后源码改写 |
| A04 | 安装 | 内网无旧四包污染且复用既有基础环境的 sdist、wheel、editable install 可复现；不联网补依赖、不替换 torch，依赖检查通过，版本来源唯一 |
| A05 | 设备依赖 | 当前产品源码/构建产物无其他计算设备实现，部署依赖无非 Ascend 专用 runtime；不能将第三方基础库内部代码视为本项目必须重写的范围 |
| A06 | 原生扩展 | pybind 导入名、共享库、RPATH、ABI 匹配；所有保留 op 注册存在 |
| A07 | 初始化 | config/tokenizer/CLI 不意外初始化设备；spawn worker 的 rank/device、CANN 初始化顺序正确 |
| A08 | 负向用例 | 非白名单架构、CUDA/CPU 执行、已删训练路由和未支持组合明确拒绝；无 Transformers fallback 绕过 |
| A09 | 配置 | 每个保留选项有效；删除或改名的配置有明确提示，禁配规则仍生效 |
| A10 | 模型裁剪完整性 | 产品源码和构建产物不含多模态、Qwen/Llama 蒸馏及其他非目标模型的完整实现/注册；GLM/MTP 必要复用代码已抽为公共组件；离线、在线和 draft 加载均拒绝范围外模型 |
| A11 | 文本输入边界 | 离线和在线入口对图像、音频、视频等输入明确报错，不静默丢弃；正常文本 chat、reasoning 和 tool call 通过 |
| A12 | 首批环境一致性 | 910B3、CANN 8.5.1、torch/torch_npu 2.9.0 与内网设备、构建和运行记录一致；使用已安装 torch，默认非隔离构建，安装/CI/镜像不改用目标外版本 |
| A13 | 无互联网构建 | 全新构建目录在无公网出口且无历史下载缓存时可生成两个产物；Python/backend、CMake、Git、系统包和镜像均无在线获取前提；离线材料版本与校验值完整 |
| A14 | 离线运行与证据 | 模型/config/tokenizer/必要 remote code 和测试数据由内网材料提供；无模型 Hub 自动下载；报告能对应源码和产物版本，本机检查不冒充内网构建或 NPU 验收 |

`torch.cuda`、`nvcc`、`vllm_ascend` 等扫描是候选定位工具，还需区分生产实现、文档、许可证和负向测试。不能用字符串零命中代替运行闭包验证，也不能仅用“当前没执行到”证明代码已经裁剪。

### 4.2 目标模型和服务

| 编号 | 场景 | 必测内容 |
| --- | --- | --- |
| B01 | 每个批准文本架构 | config/tokenizer/权重加载、参数映射、短/长 prompt、EOS/stop、logprobs、连续生成 |
| B02 | Dense GLM / ChatGLM | 抽公共 decoder/MLP 后结果与基线对照，TP 分片和特殊 token 一致 |
| B03 | DeepSeek/GLM MoE、MLA、DSA | expert routing、共享专家、indexer、top-k、mask、长上下文和 KV layout |
| B04 | GLM 结构化共享 indexer | shared 层无自有 indexer 的构造/加载、producer/consumer 对应、组内层映射 |
| B05 | 量化 | 对每个目标权重实际 dtype/量化 profile 验证加载与数值；不把 CUDA 专用 FP8/AWQ/GPTQ kernel 当作 NPU 已支持 |
| B06 | 离线入口 | 单/多 prompt、`generate`/文本 `chat`、复用实例、销毁/重建、缓存开关 |
| B07 | 在线入口 | models、completion、chat、流式拼接、非流式、并发、请求取消/断连、超长输入、错误响应、优雅退出 |
| B08 | 模型交互 | DeepSeek/GLM chat template、reasoning、tool call、目标结构化输出；与纯文本采样结果区分验证 |
| B09 | 调度与并行 | 批处理/混合 prefill+decode、prefix hit、抢占、TP/EP/PP/DP 中批准组合；空 rank 和进程退出 |
| B10 | 图模式 | eager vs ACL capture/replay；实际 replay 计数/trace、bucket 边界、batch 重排和无效行，不只检查成功 capture |
| B11 | MTP | 开关、候选宽度、全接收/部分接收/全拒绝、accepted frontier、KV 回退、MTP × 图 × DSA 关键组合 |

代码搬迁且数值路径未变的确定性用例，首先要求固定输入与 greedy token 序列一致。遇到设备非确定性时，保留差异证据并使用预先约定的 logits/attention 误差和任务指标；不能事后放宽阈值掩盖错误。随机采样不要求跨执行完全相同，但分布和质量不能出现系统性偏移。

### 4.3 KV、RemoteFill 和恢复

本节覆盖的目标模型现有链路为必保回归范围，不能以单实例或无缓存 smoke 代替跨实例缓存、P/D 及故障恢复验证；各组合继续遵守当前代码的支持限制。

| 编号 | 场景 | 关键不变量 |
| --- | --- | --- |
| C01 | 无缓存/冷缓存/热缓存 | 命中 token 数、输出与无缓存对照，部分 prefix、不完整 chunk、cache miss |
| C02 | CPU/NPU KV round trip | Dense、MLA、DSA、dtype、shape/stride、block/chunk 边界、部分页和搬运字节正确 |
| C03 | 不等 KV 分组 | latent/index 层集合独立；覆盖现有 79/22 fixture，以及小型不等组和不合法元数据 |
| C04 | 共享 CPU cache | rank0 与 passive rank、NUMA、host registration、handle/offset/generation、退出后清理 |
| C05 | DSA selective load | top-k 对应原 token、producer/consumer 正确、index readiness、scratch slot 不冲突、MTP 行顺序 |
| C06 | store-before-free | required group 和 owner 全部完成才释放；慢 store、失败/取消、部分完成不能释放唯一副本 |
| C07 | checkpoint/preemption | 命中/缺失/重试、部分页、分配预算、恢复后再次抢占、idle control path、回收无悬挂 |
| C08 | RemoteFill | 冷启动、discovery 并发、sealing 后首次 prepared load、目的地址代际、warm reuse、多请求完成与释放 |
| C09 | 通信失败 | 超时、对端退出、延迟完成、传输已武装后的错误、paired restart；资源和错误状态可收敛 |
| C10 | 图和 layerwise 回调 | 检索 split 前后事件顺序、callback exactly-once、无错误游标推进、replay 不使用旧地址 |
| C11 | 旧 PD/P2P backend | 批准的非 layerwise 配置正常；与 layerwise 同开时仍明确拒绝 |
| C12 | 缓存格式 | 错模型 revision/量化/分组/namespace/schema 拒绝或按规定 miss；不能读错缓存返回正常输出 |
| C13 | 首批通信组合 | 在 910B3/CANN 8.5.1 上验证最终选定的缓存传输通道、SDK/动态库及跨实例/P/D 链路，单独验证 HCCL 集合通信；不通过禁用必保功能绕过不兼容 |

数据尚未写入时可以选择已有的安全回退；数据/索引已改变后的错误，按当前协议做失败/重算/成对重启。不能在部分状态已写入后无条件重新执行一次 native forward。故障测试必须覆盖非输出 TP rank，验证不会无限等待；已有 bounded timeout 不自动等于成功路径增加全局同步的授权。

## 5. 可复用的现有测试入口

以下文件已在当前树中找到；尚未在本轮执行。迁移时保持用例语义并更新 import。部分 standalone 测试采用源码提取或 mock，能检查结构/状态约束，但不能替代 NPU 数值和集成测试。

| 能力 | 当前入口 |
| --- | --- |
| 模型侧 GLM shared indexer | `vllm/tests/model_executor/test_glm52_shared_indexer.py` |
| 推理侧 KV 生命周期 | `vllm/tests/standalone/test_kv_transfer_lifetime.py` |
| Runner KV spec | `vllm-ascend/tests/ut/worker/test_glm52_shared_indexer_spec.py` |
| top-k 所有权和恢复 | `vllm-ascend/tests/standalone/test_glm52_topk_ownership.py`、`test_preemption_dispatch.py`、`test_checkpoint_graph_route.py`、`test_mc2_recovery.py` |
| Cache 元数据和 checkpoint | `LMCache/tests/standalone/test_glm52_metadata.py`、同目录 `test_checkpoint_*.py` |
| NPU 组数量和 round trip | `LMCache-Ascend/tests/v1/test_glm52_group_cardinality.py`、`test_sparse_mla_dsa_kv_roundtrip.py`、`test_mla_dsa_two_groups.py` |
| RemoteFill 与 prepared load | `LMCache-Ascend/tests/v1/test_remote_fill_qualification.py`、`test_remote_fill_final_batch.py`、`test_prepared_sparse_retrieve.py`、`test_sparse_destination_layout.py` |
| checkpoint 与异常完成 | `LMCache-Ascend/tests/standalone/test_glm52_dispatch.py`、`test_local_checkpoint_restore.py`、`test_cold_abort_completion.py` |
| 通信通道 | `LMCache-Ascend/tests/v1/transfer_channel/` 中 HCCL/HIXL/hcomm 测试 |

实施 P0 时，在内网各仓已准备好的依赖环境中分别运行对应测试，不把四个仓库的全部 `tests/` 在一个 pytest 进程中混跑。下面直接使用已激活环境的解释器，不创建新环境或在线同步依赖；模型及数据需预先齐备。本机只运行已有工具可支持且不需要构建的 host 逻辑检查。例：

```bash
# 在内网 vllm 仓库、已准备并激活的环境中执行；本轮未运行
python -m pytest tests/model_executor/test_glm52_shared_indexer.py tests/standalone/test_kv_transfer_lifetime.py
```

```bash
# 在内网 LMCache-Ascend 仓库、已准备并激活的 Ascend 环境中执行；本轮未运行
python -m pytest tests/v1/test_glm52_group_cardinality.py tests/v1/test_sparse_mla_dsa_kv_roundtrip.py tests/v1/test_remote_fill_qualification.py
```

完成迁移后的两仓测试命令根据新路径生成并写入交付文档，不假设旧路径一直存在。优先复用现有行为测试，新增测试聚焦依赖脱钩、导入顺序、白名单绕过、异常释放和跨仓契约等真实风险。

## 6. 性能与长稳门槛

以下是建议门槛，需在 P0 固定噪声水平后写入验收基线；这些数值不是已测结果。

| 维度 | 建议标准 |
| --- | --- |
| 固定并发的端到端吞吐 | 相对同环境基线回退不超过 5% |
| TTFT / TPOT / ITL | 分别记录中位数和 p95；必保场景回退不超过 5%，低时延场景同时判断绝对差和噪声 |
| 峰值 NPU 内存 | 同配置不出现未解释的超过 5% 增长；原可运行的最大验收上下文仍可运行 |
| CPU RSS / shared memory | 无持续增长、遗留共享对象和资源泄漏；稳定值与基线比较 |
| 缓存 | 命中、实际加载字节、store 完成和回收正确；不能以额外重算换取表面输出正常 |
| 图路径 | 目标 key 实际 replay，无意外长期退回 eager；数值与资源状态通过 |
| 长稳 | 建议至少 8 小时混合并发/取消/缓存命中/抢占负载；关键部署配置按环境安排更长验证 |

性能对比使用同硬件、权重、输入 token、输出限制、采样、并发、缓存温度、图模式和并行策略。至少 3 轮独立重复，先预热再统计，给出波动；冷启动单独记录。不能只用整体平均掩盖 decode-window save 边界的时延尖峰。

若基线没有稳定复现或噪声大于门槛，应先改善测量并重新固定基线；不能把“未能测量”记为“没有回归”。结构精简本身不保证性能提升，本轮只要求保留既有能力与可接受的性能。

## 7. 主要风险及处理

| 风险 | 处理 | 阻止发布的证据 |
| --- | --- | --- |
| 仅搬目录，运行仍依赖 patch | 建立 patch-to-source 映射并检查新环境导入 | 需要旧插件包或特定 import 顺序才能工作 |
| GPU Runner 被删，继承行为丢失 | 提取依赖，逐模式验证 | async output、状态、capture 或销毁路径异常 |
| 目标模型依赖被误删 | AST + 动态 registry + 实际加载闭包 | GLM/MTP 需要恢复完整非目标模型才能启动 |
| 依赖版本/ABI 冲突 | 锁定统一 profile，干净构建安装 | 安装时替换 torch、加载扩展失败或隐含 CUDA runtime |
| 离线构建隐含下载 | 完整离线材料与目标平台标签，检查 backend/CMake/Git/镜像，内网无缓存重建 | 需要公网、临时替换 torch、缺失源码/资源，或只能靠开发机缓存成功 |
| GLM index/latent 组被同构化 | 独立层集合与映射测试 | 错搬运字节、producer 错配、缺失组仍被视为命中 |
| 异步 store/callback 生命周期改变 | 事件/所有权测试与异常注入 | 早释放、重复完成、悬挂请求或错误回退 |
| 稀疏路径被错误 FULL graph 化 | 保留 host retrieve split，检查 replay trace | 图内 Python 回调假设或 replay 错地址 |
| 只更新一个仓库/缓存格式 | 配对版本、namespace、兼容验证 | 新旧读写不兼容却仍读取对象 |
| 可选裁剪超出用户目标 | README 决策表审核，未确认保留边界 | 删除仍被需求指定的模型/硬件/推理能力 |
| 旧文档或小型 mock 被当作硬件证明 | 标注证据级别，实测关键模式 | 缺乏代表性模型和 NPU 运行结果 |

## 8. 最终验收清单

- [ ] 只有两个活动仓库和两个构建/安装单元，原始四仓基线可恢复。
- [ ] 原生 NPU 主路径完整，生产不再依赖 Ascend 插件、GPU Runner 或运行时补丁。
- [ ] 模型、设备、任务、算子、量化、Connector 注册与批准范围一致；负向测试通过。
- [ ] DeepSeek / GLM 原生文本 checkpoint 有逐项结果，已有 MTP、DSA 保留并验收；LoRA、首批之外的型号等其余决策有明确记录。
- [ ] 多模态和基于 Qwen/Llama 的蒸馏模型实现及专用依赖已移除；范围外模型和非文本输入的拒绝用例通过。
- [ ] 离线、在线、采样/解析、并行和缓存开关通过；启用的 DSA/MTP/图组合有实际执行证据。
- [ ] GLM 结构化 indexer、不等 KV 组、store-before-free、checkpoint、RemoteFill 及失败恢复通过。
- [ ] CPU KV 卸载、跨实例缓存和既有 P/D 分离保留，并按实际部署拓扑完成联合及故障回归。
- [ ] 首批结果来自指定的 910B3 / CANN 8.5.1 / torch 2.9.0 / torch_npu 2.9.0，配套环境和构建 ABI 有可复现记录。
- [ ] 构建与安装在无互联网的内网完成，已安装 torch 2.9.0 未被替换；离线材料完整，源码/产物校验值和回传日志可对应。
- [ ] 依赖/扩展/镜像干净可复现，无意外 GPU runtime，无旧包残留帮助运行。
- [ ] 性能、内存、长稳达到固定门槛，未测或不支持的组合明确列出。
- [ ] 新旧路径、配置和版本配对有记录；LICENSE/NOTICE/版权来源保留。
- [ ] 最终双仓从干净环境复现成功，完成成对回退演练和旧仓可恢复归档。
