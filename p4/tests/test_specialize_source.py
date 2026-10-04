# SPDX-License-Identifier: Apache-2.0
"""Semantic regressions for the review-only P4 branch specializer."""

import runpy
import textwrap
from pathlib import Path
from types import GeneratorType
from types import SimpleNamespace as NS

import pytest

SPECIALIZE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/specialize_source.py")
)["specialize"]


def compare(source: str, *arguments) -> str:
    """Compare return values, raised errors and mutations before/after folding."""
    source = textwrap.dedent(source)
    specialized = SPECIALIZE(source)
    results = []
    for code in (source, specialized):
        namespace = {"current_platform": NS(is_npu=lambda: True)}
        exec(compile(code, "<specializer fixture>", "exec"), namespace)
        events = []
        try:
            result = namespace["choose"](*arguments, events)
            if isinstance(result, GeneratorType):
                result = list(result)
        except ValueError as error:
            result = (type(error), str(error))
        results.append((result, events))
    assert results[0] == results[1], specialized
    assert SPECIALIZE(specialized) == specialized
    return specialized


@pytest.mark.parametrize("method", ["mtp", "unknown"])
def test_mtp_tail_else_remains_conditional(method):
    """Deleting retired methods must not unconditionally execute the error tail."""
    source = """
        def choose(config, events):
            if config.method == "ngram":
                events.append("retired")
            elif config.use_eagle():
                events.append("mtp")
            elif config.method == "medusa":
                events.append("retired")
            elif config.method == "extract_hidden_states":
                events.append("retired")
            else:
                raise ValueError("unknown method")
            events.append("rejection sampler")
    """
    compare(source, NS(method=method, use_eagle=lambda: method == "mtp"))


@pytest.mark.parametrize("has_drafter", [False, True])
@pytest.mark.parametrize("method", ["mtp", "unknown"])
def test_removed_middle_elif_keeps_previous_branch_exclusive(has_drafter, method):
    """A no-drafter branch must not fall through to the later MTP branch."""
    source = """
        def choose(drafter, config, events):
            if not drafter:
                result = None
            elif config.method == "ngram":
                result = "retired"
            elif config.use_eagle():
                events.append("mtp")
                result = "draft"
            else:
                raise ValueError("unknown method")
            return result
    """
    compare(source, has_drafter, NS(method=method, use_eagle=lambda: method == "mtp"))


@pytest.mark.parametrize("first", [False, True])
def test_true_elif_becomes_else_not_unconditional_body(first):
    source = """
        def choose(first, events):
            if first:
                events.append("first")
            elif current_platform.is_npu():
                events.append("npu")
            else:
                events.append("retired")
            return "done"
    """
    compare(source, first)


@pytest.mark.parametrize("first", [False, True])
def test_false_final_elif_without_else_is_removed(first):
    source = """
        def choose(first, config, events):
            if first:
                events.append("first")
            elif config.method == "ngram":
                events.append("retired")
            return "done"
    """
    compare(source, first, NS(method="mtp"))


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_else_containing_if_is_not_mistaken_for_elif(first, nested):
    source = """
        def choose(first, nested, config, events):
            if first:
                events.append("first")
            elif config.method == "ngram":
                events.append("retired")
            else:
                if nested:
                    events.append("nested")
                events.append("tail")
    """
    compare(source, first, nested, NS(method="mtp"))


@pytest.mark.parametrize(
    "kind",
    ["default", "linear", "yarn", "deepseek_yarn", "deepseek_llama_scaling", "unknown"],
)
def test_rope_factory_keeps_one_dispatch_chain_and_reachable_cache(kind):
    source = """
        def choose(scaling_type, events):
            if scaling_type == "default":
                result = "default"
            elif scaling_type == "llama3":
                result = "retired"
            elif scaling_type == "linear":
                result = "linear"
            elif scaling_type == "ntk":
                result = "retired"
            elif scaling_type == "yarn":
                result = "yarn"
            elif scaling_type in ("deepseek_yarn", "deepseek_llama_scaling"):
                result = "deepseek"
            elif scaling_type == "longrope":
                result = "retired"
            else:
                raise ValueError("unknown rope")
            events.append("cache")
            return result
    """
    compare(source, kind)


@pytest.mark.parametrize("strategy", ["eager", "lazy", "prefetch"])
def test_weight_iterator_does_not_yield_twice(strategy):
    source = """
        def choose(safetensors_load_strategy, events):
            if safetensors_load_strategy == "eager":
                events.append("eager")
                yield "weight"
            elif safetensors_load_strategy == "torchao":
                yield "retired"
            else:
                events.append("safe_open")
                yield "weight"
    """
    compare(source, strategy)


@pytest.mark.parametrize("last", [False, True])
def test_final_prefill_does_not_fall_through_to_chunk_alignment(last):
    source = """
        def choose(last, config, events):
            if last:
                events.append("full prompt")
            elif config.enable_blending:
                events.append("retired")
            else:
                events.append("aligned prompt")
    """
    compare(source, last, NS(enable_blending=False))
