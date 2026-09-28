"""Migrate patch tests to native interfaces; preserve off-profile tests as history."""

import ast
import re

from source_edit import ROOT, definition, find, read, replace, segment, start, write

for p in (ROOT / "ascend/tests").rglob("*.py"):
    source = p.read_text().replace("forward_oot", "forward_npu")
    source = re.sub(r"^from vllm.utils.ascend import (adapt_patch|register_ascend_customop).*\n", "", source, flags=re.M)
    source = re.sub(r"^(\s*)adapt_patch\([^\n]*\)\n", "", source, flags=re.M)
    source = source.replace("register_ascend_customop()", "initialize_native_ops()")
    if "initialize_native_ops()" in source:
        source = "from vllm.model_executor.layers.ascend import initialize_native_ops\n" + source
    p.write_text(source)

path = "ascend/tests/ut/worker/test_worker_v1.py"
source = read(path)
source = re.sub(r'^    @patch\("vllm\.(utils.ascend.adapt_patch|v1.worker.npu_worker.register_ascend_customop)"\)\n', "", source, flags=re.M)
source = re.sub(r"^        mock_(adapt_patch|register_ascend_customop),\n", "", source, flags=re.M)
source = re.sub(r"^        mock_(adapt_patch|register_ascend_customop).assert_called_once\(\)\n", "", source, flags=re.M)
source = source.replace("register_dummy_fusion_op", "initialize_native_ops")
write(path, source)

path = "ascend/tests/ut/worker/test_model_runner_v1.py"
source = read(path).replace("model_runner_module.GPUModelRunner", "model_runner_module.NPUModelRunnerState")
lines = source.splitlines(True)
offsets = [0]
for line in lines:
    offsets.append(offsets[-1] + len(line))
edits = []
for node in ast.walk(ast.parse(source)):
    if isinstance(node, ast.Call) and any(isinstance(a, ast.Constant) and a.value in ("_torch_cuda_wrapper", "_replace_gpu_model_runner_function_wrapper") for a in node.args):
        edits.append((offsets[node.lineno - 1] + node.col_offset, offsets[node.end_lineno - 1] + node.end_col_offset, "nullcontext()"))
for first, last, replacement in sorted(edits, reverse=True):
    source = source[:first] + replacement + source[last:]
write(path, source)

path = "ascend/tests/ut/test_platform.py"
for name in ("test_pre_register_and_update_with_parser", "test_pre_register_and_update_without_parser", "test_pre_register_and_update_with_parser_no_quant_action", "test_pre_register_and_update_with_existing_ascend_quant"):
    replace(path, name, f'''
def {name}(self):
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    parser = MagicMock()
    choices = list(QUANTIZATION_METHODS)
    parser._option_string_actions = {{"--quantization": MagicMock(choices=choices)}}
    with patch("vllm.platforms.npu.config_deprecated_logging") as configure_logging:
        self.platform.pre_register_and_update(parser)
    configure_logging.assert_called_once()
    self.assertIn("ascend", choices)
    self.assertEqual(choices, QUANTIZATION_METHODS)
''', "TestNPUPlatform")
source = read(path).replace("vllm.v1.core.sched.recompute_scheduler.RecomputeSchedulerConfig.initialize_from_config", "vllm.v1.core.mc2_recovery.decoder_recovery_budget")
write(path, source)

path = "ascend/tests/ut/test_utils.py"
source = read(path).replace("from vllm.utils.ascend import REGISTERED_ASCEND_OPS\n", "")
source = source.replace("from vllm.model_executor.layers.ascend import initialize_native_ops\n", "")
write(path, source)
owner = next(n.name for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and any(getattr(m, "name", "") == "test_register_ascend_customop" for m in n.body))
replace(path, "test_register_ascend_customop", '''
def test_native_layer_binding_is_immutable(self):
    from vllm.model_executor.layers.ascend.registry import NPU_LAYERS
    self.assertIn("RMSNorm", NPU_LAYERS)
    with self.assertRaises(TypeError):
        NPU_LAYERS["RMSNorm"] = ("wrong", "wrong")
''', owner)

path = "ascend/tests/ut/ops/test_gate_linear.py"
replace(path, "test_gate_linear_oot_registration_instantiates_ascend_gate", '''
def test_gate_linear_native_binding_instantiates_ascend_gate() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")))
    with mock.patch("vllm.config.get_current_vllm_config", return_value=config), mock.patch.dict(op_registry_oot, {}, clear=True):
        gate = GateLinear(input_size=16, output_size=4, bias=False)
        assert not op_registry_oot
    assert type(gate) is AscendGateLinear
''')
source = read(path)
node = find(source, "test_gate_linear_registration_is_model_specific")
fn = segment(source, node)
fn = fn[:fn.index("    previous_registered")] + '''    from vllm.model_executor.layers.ascend.registry import get_npu_layer_class
    with mock.patch("vllm.config.get_current_vllm_config", return_value=vllm_config), mock.patch("vllm.utils.ascend.is_310p", return_value=False):
        selected = get_npu_layer_class("GateLinear")
    assert selected is (AscendGateLinear if expected_registered else None)
'''
replace(path, node.name, fn)

path = "ascend/tests/ut/_310p/ops/test_mm_encoder_attention_310.py"
replace(path, "test_register_customop_overrides_mm_encoder_attention_for_310p", '''
def test_native_mm_encoder_attention_selects_310p():
    from vllm.model_executor.layers.ascend.registry import get_npu_layer_class
    with mock.patch("vllm.utils.ascend.is_310p", return_value=True):
        assert get_npu_layer_class("MMEncoderAttention") is AscendMMEncoderAttention310
''')

for p in (ROOT / "ascend/tests/ut/patch").rglob("*.py"):
    source = p.read_text()
    if p.name in {"test_patch_qwen3_next_mtp.py", "test_patch_gdn_attn.py"}:
        target = ROOT / "ascend/legacy_tests" / p.relative_to(ROOT / "ascend/tests/ut")
        target.parent.mkdir(parents=True, exist_ok=True)
        p.rename(target)
        continue
    source = source.replace("vllm.patch.worker.patch_logprobs", "vllm.v1.sample.ops.logprobs")
    source = source.replace("vllm.patch.worker.patch_routed_experts_capturer", "vllm.model_executor.layers.fused_moe.routed_experts_capturer")
    source = source.replace("from vllm.patch.worker.patch_distributed import GroupCoordinatorPatch", "from vllm.distributed.parallel_state import GroupCoordinator as GroupCoordinatorPatch")
    p.write_text(source)

path = "ascend/tests/ut/patch/platform/test_patch_minimax_usage_accounting.py"
# Off-profile reasoning parser behavior stays with the archived MiniMax patch.
original = read(path)
archive = ROOT / "ascend/legacy_tests/patch/platform/test_patch_minimax_reasoning.py"
archive.parent.mkdir(parents=True, exist_ok=True)
archive.write_text(original)
replace(path, "test_count_reasoning_tokens", "")
source = read(path).replace("from vllm.patch.platform import patch_minimax_usage_accounting as minimax_usage_patch", "from vllm.entrypoints.openai.chat_completion import serving as minimax_usage_patch")
source = source.replace("minimax_usage_patch._make_usage_info", "minimax_usage_patch.OpenAIServingChat._make_usage_info")
write(path, source)

path = "ascend/tests/ut/core/test_kv_connector_worker_metadata_patch.py"
write(path, '''# SPDX-License-Identifier: Apache-2.0
"""Metadata-before-output ordering is also exercised by the host native contract suite."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

def test_worker_metadata_precedes_stock_scheduler_update():
    source = Path(__file__).resolve().parents[4] / "vllm/v1/core/sched/scheduler.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    fn = next(node for node in cls.body if getattr(node, "name", None) == "update_from_output")
    calls = []
    connector = MagicMock()
    connector.update_connector_worker_metadata.side_effect = lambda *_: calls.append("metadata")
    self = SimpleNamespace(connector=connector, requests={"active": SimpleNamespace(is_finished=lambda: False), "finished": SimpleNamespace(is_finished=lambda: True)})
    metadata = object()
    model_runner_output = SimpleNamespace(kv_connector_output=SimpleNamespace(kv_connector_worker_meta=metadata))
    # Execute the production prefix and inspect its position before stock output processing.
    prefix = ast.Module(body=fn.body[:2], type_ignores=[])
    exec(compile(prefix, str(source), "exec"), {"self": self, "model_runner_output": model_runner_output})
    assert calls == ["metadata"]
    connector.update_connector_worker_metadata.assert_called_once_with(metadata, {"active"})
''')
print("Updated native UT bootstrap, layer selection, metadata and Runner contracts")
