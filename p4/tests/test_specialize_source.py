# SPDX-License-Identifier: Apache-2.0
"""Semantic regressions for the review-only P4 branch specializer."""

import runpy
import textwrap
from pathlib import Path
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
