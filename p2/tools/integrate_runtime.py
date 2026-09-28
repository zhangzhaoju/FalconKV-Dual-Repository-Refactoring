"""Fold process/configuration and remaining non-compiler P1 adapters into owners."""

import ast
import textwrap

from source_edit import ROOT, add_imports, append_class, definition, find, prepend_body, read, replace, replace_text, write

path = "vllm/model_executor/layers/fused_moe/oracle/unquantized.py"
replace_text(path, "return UnquantizedMoeBackend.NPU, None", "backend = UnquantizedMoeBackend.NPU")
write(path, read(path).replace("        # The Ascend MoE implementation is still migrated in a later P2 batch.\n", ""))

path = "vllm/v1/worker/runner_output.py"
add_imports(path, "from vllm.distributed.parallel_state import Handle\nfrom vllm.v1.sample.rejection_sampler import RejectionSampler")
path = "vllm/v1/worker/npu_runner_state.py"
add_imports(path, "from vllm.v1.worker.runner_output import _copy_pooler_output_to_cpu")
write(path, read(path) + "\n\n" + definition("vllm/v1/worker/gpu_model_runner.py", "EncoderTimingStats"))
path = "vllm/v1/worker/npu/v2/runner_state.py"
write(path, read(path) + "\n\n" + definition("vllm/v1/worker/gpu/model_runner.py", "ExecuteModelState"))
for path in ("vllm/v1/worker/npu/v2/model_runner.py", "vllm/v1/worker/npu_worker.py"):
    write(path, read(path).replace("torch.cuda", "torch.npu").replace("torch.accelerator", "torch.npu"))
source = read("vllm/v1/spec_decode/ngram_proposer_gpu.py").replace("torch.cuda", "torch.npu").replace("NgramProposerGPU", "NgramDeviceProposer")
write("vllm/v1/spec_decode/ascend/ngram_device_proposer.py", source)
path = "vllm/v1/worker/npu_runner_state.py"
write(path, read(path).replace("vllm.v1.spec_decode.ngram_proposer_gpu", "vllm.v1.spec_decode.ascend.ngram_device_proposer").replace("NgramProposerGPU", "NgramDeviceProposer"))

path = "vllm/utils/mem_utils.py"
source = read(path).replace("torch.accelerator.", "_memory_backend().")
write(path, source + '''
def _memory_backend():
    """torch.accelerator does not delegate all memory APIs to PrivateUse1."""
    return torch.npu if current_platform.is_npu() else torch.accelerator
''')

path = "vllm/distributed/utils.py"
add_imports(path, "from vllm.platforms import CpuArchEnum, Platform")
replace_text(path, '''USE_SCHED_YIELD = (sys.version_info[:3] >= (3, 11, 1)) or (
    sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8
)''', '''USE_SCHED_YIELD = (
    (sys.version_info[:3] >= (3, 11, 1))
    or (sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8)
) and Platform.get_cpu_architecture() != CpuArchEnum.ARM''')

# Native cache spec identity is now stable in every process, including spawn/unpickle.
path = "vllm/v1/core/sched/recompute_scheduler.py"
replace(path, "register_ascend_mla_spec_in_manager", "")
source = read(path)
begin = source.index("# `spec_manager_map`")
end = source.index("@dataclass\nclass RecomputeSchedulerConfig")
source = source[:begin] + source[end:]
source = source.replace("        register_ascend_mla_spec_in_manager()\n", "")
write(path, source)
replace(path, "RecomputeSchedulerConfig", "")
path = "vllm/config/scheduler.py"
append_class(path, "SchedulerConfig", '''
mc2_recovery_token_budget: int | None = None
"""Native NPU decoder recovery budget; populated during platform validation."""
SLO_limits_for_dynamic_batch: float = -1
"""Native NPU dynamic batch latency limit; negative disables adaptation."""
''')
path = "vllm/platforms/npu.py"
source = read(path)
begin = source.index("        if ascend_config.recompute_scheduler_enable:")
end = source.index("        # Extend original scheduler_config", begin)
source = source[:begin] + '''        if ascend_config.recompute_scheduler_enable:
            from vllm.v1.core.mc2_recovery import decoder_recovery_budget

            scheduler = vllm_config.scheduler_config
            suffix = "AsyncRecomputeScheduler" if scheduler.async_scheduling else "RecomputeScheduler"
            scheduler.scheduler_cls = "vllm.v1.core.sched.recompute_scheduler." + suffix
            scheduler.mc2_recovery_token_budget = (
                decoder_recovery_budget(vllm_config) if is_moe_model(vllm_config) else None
            )

''' + source[end:]
write(path, source)

patch = "ascend/legacy_patches/platform/patch_balance_schedule.py"
source = read(patch)
imports = "\n".join(ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, (ast.Import, ast.ImportFrom)))
write("vllm/v1/core/sched/balance_scheduler.py", imports + "\n\n" + definition(patch, "BalanceScheduler"))
write("vllm/v1/engine/balance_core.py", imports + "\n\n" + definition(patch, "BalanceDPEngineCoreProc"))
path = "vllm/config/scheduler.py"
prepend_body(path, "get_scheduler_cls", '''
from vllm import envs_ascend
if self.scheduler_cls is None and envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING and not self.async_scheduling:
    from vllm.v1.core.sched.balance_scheduler import BalanceScheduler
    return BalanceScheduler
''', "SchedulerConfig")
path = "vllm/v1/engine/core.py"
replace_text(path, "engine_core = DPEngineCoreProc(*args, **kwargs)", '''from vllm import envs_ascend
                if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
                    from vllm.v1.engine.balance_core import BalanceDPEngineCoreProc
                    engine_core = BalanceDPEngineCoreProc(*args, **kwargs)
                else:
                    engine_core = DPEngineCoreProc(*args, **kwargs)''')
path = "vllm/v1/core/sched/scheduler.py"
append_class(path, "Scheduler", definition(patch, "balance_gather", "BalanceScheduler"))
add_imports(path, "import torch\nimport torch.distributed as dist")
# Subclasses that override scheduling retain their P1 algorithm, while DP load gathering
# is available regardless of which scheduler configuration owns the request state.
prepend_body(path, "__init__", '''
from vllm import envs_ascend
if envs_ascend.VLLM_ASCEND_BALANCE_SCHEDULING:
    self.balance_queue = [
        torch.tensor([0], dtype=torch.int, device="cpu")
        for _ in range(vllm_config.parallel_config.data_parallel_size)
    ]
''', "Scheduler")

# Quarot model weight handling is instance-local. The original target config is
# already passed to Eagle3LlamaForCausalLM by the model loader.
patch = "ascend/legacy_patches/worker/patch_draft_quarot.py"
source = read(patch).replace("from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM\n", "")
write("vllm/model_executor/model_loader/ascend/quarot.py", source)
replace("vllm/model_executor/model_loader/ascend/quarot.py", "patch_load_weights", "")
path = "vllm/model_executor/models/llama_eagle3.py"
prepend_body(path, "__init__", '''
self._quarot_weight_loader = None
from vllm.platforms import current_platform
if current_platform.is_npu() and vllm_config.quant_config is not None:
    from pathlib import Path
    from vllm.model_executor.model_loader.ascend.quarot import get_rotation_path, make_load_weights
    rotation_path = get_rotation_path(vllm_config)
    if rotation_path is not None:
        self._quarot_weight_loader = make_load_weights(Path(vllm_config.model_config.model), rotation_path)
''', "Eagle3LlamaForCausalLM")
prepend_body(path, "load_weights", '''
if self._quarot_weight_loader is not None:
    return self._quarot_weight_loader(self, weights)
''', "Eagle3LlamaForCausalLM")
path = "vllm/v1/worker/npu_model_runner.py"
source = read(path).replace("from vllm.patch.worker.patch_draft_quarot import patch_load_weights\n", "").replace("from vllm.patch.worker.patch_module import patch_torch_npu_argsort\n", "")
source = source.replace("                if self.vllm_config.quant_config is not None:\n                    patch_load_weights(self.vllm_config)\n", "")
source = source.replace("                patch_torch_npu_argsort()", "                # Boolean sort is handled directly by the attention metadata builder.")
write(path, source)
path = "vllm/v1/attention/backends/gdn_attn.py"
replace_text(path, "torch.argsort(spec_token_masks, stable=True)", "torch.argsort(spec_token_masks.to(torch.int32), stable=True)")

print("Integrated native process/configuration, sampling state, memory and weight adapters")
