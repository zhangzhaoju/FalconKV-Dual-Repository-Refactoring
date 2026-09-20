# P0 基线问题与核对记录

所有“源码发现”均非 NPU 实测结果。当前模型仅 GLM-5.2-w4a8c8，硬件为 910B3 / 32 卡 / 2P2D / TP8/DP2。

| 编号 | 发现与证据 | 当前处置 | 退出条件 |
| --- | --- | --- | --- |
| P0-F01 | vLLM/LMCache 原始 build-system pin torch 2.10.0；LMCache-Ascend 未固定 torch/torch_npu | 独立对齐 2.9.0，保留原始 bundle；静态元数据测试通过 | 内网非隔离构建、扩展导入和图/MTP/DSA API/ABI 实测通过；不能仅改版本号即销项 |
| P0-F02 | 10 个 standalone 测试引用不存在的 `LMCache-NPU`；修复前 host 子集 27 项失败 | 仅改同级目录为 `LMCache`，不改断言；复测 22 项转为通过，5 项暴露本机 torch 缺失 | 内网重跑剩余测试及完整相关恢复测试 |
| P0-F03 | 本机未安装 torch；5 个初始化检查经 `runpy` 间接导入它 | 明确保留失败结果，不在本机安装或用假模块掩盖 | 使用已安装 torch 2.9.0 的内网环境复测 |
| P0-F04 | `catlass` 和 `kvcache-ops` 为未初始化 gitlink | 固定 commit、URL 和缺件记录已生成；当前 bundle 不含载荷 | 经允许来源取得精确 commit，核对完整源码及其嵌套依赖，保留可复现材料 |
| P0-F05 | vLLM `requirements/common.txt` 要求 `opencv-python-headless>=4.13.0`，Ascend `requirements.txt` / build-system 要求 `<=4.11.0.86` | 确认版本区间无交集，禁止直接把原始清单合并安装；本轮尚未改该运行依赖 | 核对内网现有版本和目标文本导入闭包；独立修复临时兼容依赖或移除非目标媒体依赖，记录原因，不任意选版本 |
| P0-F06 | LMCache common 依赖仍含 cufile/NIXL/nvtx/CuPy CUDA；vLLM 默认可探测 CUDA/CPU | 记录当前包装风险，P0 安装不能直接解析所有原始依赖；最终在合仓/裁剪阶段消除 | 目标 profile 无其他设备 runtime，依赖检查通过；`--no-deps` 不能用于永久掩盖错误元数据 |
| P0-F07 | W4A8_DYNAMIC、KV C8、sparse C8 源码存在，但实际 GLM-5.2 配置/量化描述未取得 | 支持矩阵标记源码存在、运行未验证；不从模型名推定所有层 dtype | 确认模型提供方、逐层量化/回退、MTP 权重、latent/index 数据与 scale 布局；数值与缓存 round trip 通过 |
| P0-F08 | CANN 8.5.1 默认进入 HIXL/hcomm one-sided 构建路径，USE_HIXL 可覆盖 | 不预先更改通道或通过关闭跨实例/P/D 规避；HCCL 集合通信单独保留 | 实际 SDK、动态库、910B3 单/跨节点通道和异常恢复通过 |
| P0-F09 | 静态索引包含 784 个赋值/调用候选；可能含普通成员赋值，动态加载和 try/except 路径未完整解析 | 保留源位置、词法条件和候选性质；未宣称补丁语义/动态闭包已完成 | 按 target profile 逐项确认是否启用、实际替换、新行为、所属模块、测试和执行顺序 |

## torch 2.9.0 初步源码核对

除构建声明外，当前源码已包含若干较旧 torch 的兼容分支：

- `vllm/vllm/compilation/wrapper.py` 对 `torch.compiler.skip_all_guards_unsafe` 和 AOT 配置使用 `hasattr`；缺失时有替代/禁用路径。
- `vllm/vllm/compilation/compiler_interface.py` 对 `<2.10.0` 的 standalone cache 写入及 AOT 参数作版本判断。
- `vllm/vllm/compilation/decorators.py` 对部分 2.10 新选项作版本判断。
- `vllm_ascend/patch/platform/patch_fusion_matcher_compat_ops.py` 为缺失的上游 op 提供占位对象；若实际执行会报错，不能当作可用算子实现。

这些观察只说明存在兼容处理，不证明所有 2.9.0 API/ABI 闭包成立。完整线索保存在 `baseline/dependency-findings.json`；内网必须覆盖实际导入、编译、ACL replay、W4A8C8、MTP 与缓存组合。CUDA/ROCm/XPU 专用 requirements 和 CMake 中的 2.10 常量未在 P0 批量改成 2.9，以免制造这些路径受支持的假象；最终随非目标设备裁剪。

## 安装 patch 的触发事实

`LMCache-Ascend/setup.py` 定义了 `run_patches()`，但当前扩展构建末尾的调用被注释为暂时禁用；`ascend_extension()` 注册的是 `build_py` / `build_ext`。不能把函数存在解释为当前安装必然执行源码改写。顶层 `lmcache_ascend/__init__.py` 的运行时注入依然存在，需与安装 patch 分开核对、迁移和验收。

## 功能组合限制

旧 PD/P2P backend 与 layerwise 的禁配规则继续保留；RemoteFill 和旧 backend 不能混作同一个已支持组合。4 节点 2P2D 先冻结实际通道和路由，基线需覆盖两个 P、两个 D、副本间路由、非输出 TP rank 及一侧故障；不以单节点 TP8 smoke 替代 32 卡联合验收。
