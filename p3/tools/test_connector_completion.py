# SPDX-License-Identifier: Apache-2.0
"""Paired production completion bodies on host fixtures, not runtime/NPU tests.

This workspace gate deliberately reads both checkouts. Product standalone tests
remain independent of a sibling repository. The same probe runs actual installed
classes in the intranet via check_npu_bootstrap.py --check-lmcache.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[3]
VLLM = WORKSPACE / "p1-repos/vllm"
LMCACHE = WORKSPACE / "p1-repos/LMCache"
CONNECTOR = "distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"


def source_class(path, name, methods=None):
    """Execute complete selected method bodies; no device imports or patches."""
    cls = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    if methods is not None:
        cls.bases = []
        cls.decorator_list = []
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
        assert {node.name for node in cls.body} == set(methods)
    namespace = {"dataclass": dataclasses.dataclass, "field": dataclasses.field}
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    code = ast.fix_missing_locations(ast.Module(body=[prefix, cls], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def paired_classes():
    wrapper = source_class(
        VLLM / "vllm" / CONNECTOR,
        "LMCacheConnectorV1",
        ["update_connector_output"],
    )
    adapter = source_class(
        LMCACHE / "lmcache/integration/vllm/vllm_v1_adapter.py",
        "LMCacheConnectorV1Impl",
        [
            "update_connector_output",
            "_common_update_connector_output",
            "_clear_request_marker",
            "_discard_request_set",
            "_take_completed_cold_load",
        ],
    )
    output = source_class(VLLM / "vllm/v1/outputs.py", "KVConnectorOutput")
    return wrapper, adapter, output


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location(
        "paired_bootstrap", VLLM / "tools/check_npu_bootstrap.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert CONNECTOR in tool.IDENTITY_FILES
    assert "lmcache_completion" not in tool.ORDERS
    return tool.check_lmcache_completion


def test_real_paired_completion_state_machine(probe, paired_classes):
    result = probe(*paired_classes)
    assert result["cases_passed"] == 3
    assert result["kv_events_enabled"] is False
    assert result["constructors_called"] is False
    assert result["device_allocation"] is False


def test_probe_rejects_missing_formal_delegation(probe, paired_classes):
    wrapper, adapter, output = paired_classes

    class OldWrapper(wrapper):
        def update_connector_output(self, result):
            # Prior formal wrapper discarded controls when KV events were off.
            if not result.kv_cache_events:
                return
            raise AssertionError("this fixture must not generate KV events")

    with pytest.raises(RuntimeError, match="did not release the cold-load slot"):
        probe(OldWrapper, adapter, output)


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("sticky-slot", "did not release the cold-load slot"),
        ("lost-completion", "validation/resume state"),
        ("missing-resume-marker", "sparse resume markers"),
        ("invalid-block-promoted", "validation/resume state"),
    ],
)
def test_probe_rejects_incorrect_adapter_state(probe, paired_classes, fault, expected):
    wrapper, adapter, output = paired_classes

    class BrokenAdapter(adapter):
        def update_connector_output(self, result):
            active = getattr(self, "_dsa_group1_direct_hbm_active_req_id", None)
            super().update_connector_output(result)
            if fault == "sticky-slot" and active is not None:
                self._dsa_group1_direct_hbm_active_req_id = active
            if fault == "lost-completion":
                self._dsa_cold_loaded_req_ids = set()
            if fault == "invalid-block-promoted":
                self.__dict__.setdefault("_dsa_cold_loaded_req_ids", set()).update(
                    (result.finished_recving or set()) & self.load_specs.keys()
                )

        def _take_completed_cold_load(self, req_id, load_spec):
            ready = super()._take_completed_cold_load(req_id, load_spec)
            if ready and fault == "missing-resume-marker":
                del load_spec.dsa_cold_compact_resume
            return ready

    with pytest.raises(RuntimeError, match=expected):
        probe(wrapper, BrokenAdapter, output)
