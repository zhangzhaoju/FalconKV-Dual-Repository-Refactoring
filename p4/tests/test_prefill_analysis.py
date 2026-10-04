# SPDX-License-Identifier: Apache-2.0
"""CPU/stdlib fixtures for bounded 2P2D log evidence, not runtime tests."""

import json
from pathlib import Path
import runpy

import pytest

TOOL = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/analyze_prefill.py")
)


def event(kind="scheduler_lookup", time=0, **kwargs):
    row = {
        "schema": 1,
        "event": kind,
        "clock_domain": "node:boot",
        "pid": 42,
        "monotonic_ms": time,
        "req_id": "request",
        "prompt_tokens": 100,
        "vllm_cached_tokens": 0,
        "lmcache_cached_tokens": 20,
        "resumed": False,
        "query_scope": "all_tokens",
        "transfer_id": "transfer",
        "outcome": "PERSISTENT_ONLY",
        "persistent_common_end": 100,
        "required_store_end": 100,
        **kwargs,
    }
    return "\x1b[32m[LMCACHE_COLD_PERF] " + json.dumps(row) + " (source.py:10)\x1b[0m\n"


def metric(second, running=1, waiting=0):
    return (
        f"INFO 10-04 15:00:{second:02d} [loggers.py] "
        f"Avg prompt throughput: 0.0 tokens/s, "
        f"Running: {running} reqs, Waiting: {waiting} reqs\n"
    )


def fixture_run(directory, duration=20):
    directory.mkdir()
    for rank in (0, 1):
        (directory / f"host_P_dp{rank}_port7910.log").write_text(
            event()
            + event("remote_fill_producer_terminal", 5000)
            + event(time=20000, req_id="later", lmcache_cached_tokens=90)
        )
        (directory / f"host_D_dp{rank}_port7920.log").write_text(
            metric(0)
            + metric(5, running=4, waiting=1)
            + metric(15, running=9, waiting=6)
            + metric(duration)
        )
    return directory


def test_matched_window_does_not_compare_late_baseline_queue(tmp_path):
    baseline = fixture_run(tmp_path / "baseline")
    candidate = fixture_run(tmp_path / "candidate")
    for path in candidate.glob("*_D_*.log"):
        path.write_text(metric(0) + metric(5, running=4, waiting=1) + metric(10))
    report = TOOL["analyze"](baseline, candidate, 12)
    node = report["nodes"]["D0"]
    assert node["matched_window_seconds"] == 10
    assert node["baseline"]["full_capture"]["max_waiting"] == 6
    for label in ("baseline", "candidate"):
        assert node[label]["matched_window"]["max_waiting"] == 1
        assert node[label]["matched_window"]["positive_waiting_samples"] == 1
    assert "passed" not in report
    assert report["warnings"]


def test_prefill_counts_terminals_not_rank_duplicates_and_uses_latest_lookup(tmp_path):
    path = tmp_path / "p.log"
    path.write_text(
        event()
        + event(time=1000, lmcache_cached_tokens=30)
        + event("remote_fill_producer_terminal", 5000)
        + event("remote_fill_producer_terminal", 5000, pid=43)
        + event(time=20000, req_id="later", lmcache_cached_tokens=90)
    )
    node = TOOL["read_node"](path, "P")
    report = TOOL["prefill"](node, 10)
    assert report["terminal_transfers"] == 1
    assert report["duplicate_terminal_events"] == 1
    assert report["completed_uncached_tokens"] == 70
    assert report["closed_request_token_rate_estimate"] == 7
    assert report["early_lookup_cache"]["lookup_events"] == 2
    assert report["early_lookup_cache"]["event_weighted_hit_fraction"] == 0.25
    assert report["later_lookup_cache"]["event_weighted_hit_fraction"] == 0.9


def test_physical_line_numbers_preserve_carriage_returns_and_ansi(tmp_path):
    path = tmp_path / "p.log"
    path.write_bytes(("progress\rprogress\n" + event()).encode())
    assert TOOL["read_node"](path, "P")["events"][0]["line"] == 2


@pytest.mark.parametrize(
    "extra, message",
    [
        (event(time=10, clock_domain="other:boot"), "mixed clock"),
        (event(time=10, pid=43), "multiple scheduler"),
        (event(time=float("nan")), "invalid monotonic"),
        ("[LMCACHE_COLD_PERF] {broken\n", "malformed"),
    ],
)
def test_invalid_event_stream_fails_closed(tmp_path, extra, message):
    path = tmp_path / "p.log"
    path.write_text(event() + extra)
    with pytest.raises(ValueError, match=message):
        TOOL["read_node"](path, "P")


def test_unrelated_events_need_no_clock(tmp_path):
    path = tmp_path / "p.log"
    path.write_text('[LMCACHE_COLD_PERF] {"event":"metadata"}\n' + event())
    assert len(TOOL["read_node"](path, "P")["events"]) == 1


def test_failed_or_incomplete_terminals_are_not_successes(tmp_path):
    path = tmp_path / "p.log"
    path.write_text(
        event()
        + event("remote_fill_producer_terminal", 1000, outcome="FAILED")
        + event(
            "remote_fill_producer_terminal",
            2000,
            transfer_id="two",
            persistent_common_end=90,
        )
        + event("remote_fill_producer_terminal", 3000, req_id="unknown")
    )
    report = TOOL["prefill"](TOOL["read_node"](path, "P"), 10)
    assert report["terminal_transfers"] == 0
    assert report["unsupported_terminals"] == 2
    assert report["unmatched_terminals"] == 1


def test_missing_and_duplicate_nodes_fail(tmp_path):
    baseline = fixture_run(tmp_path / "baseline")
    missing = baseline / "host_D_dp1_port7920.log"
    missing.rename(baseline / "saved.txt")
    with pytest.raises(ValueError, match="require exactly"):
        TOOL["read_run"](baseline)
    (baseline / "another_P_dp0_port7910.log").write_text(event())
    with pytest.raises(ValueError, match="duplicate P0"):
        TOOL["read_run"](baseline)


def test_identical_auxiliary_artifact_detected_without_reading_it_as_new_result(
    tmp_path,
):
    baseline = fixture_run(tmp_path / "baseline")
    candidate = fixture_run(tmp_path / "candidate")
    for directory in (baseline, candidate):
        (directory / "benchmark-old.json").write_text('{"old": true}')
    (candidate / "benchmark-new.json").write_text('{"new": true}')
    before = {p: p.read_bytes() for d in (baseline, candidate) for p in d.iterdir()}
    result = TOOL["analyze"](baseline, candidate, 10)
    assert len(result["identical_auxiliary_files_excluded"]) == 1
    assert all(p.read_bytes() == content for p, content in before.items())


def test_cli_never_overwrites_existing_file(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("original")
    with pytest.raises(SystemExit):
        TOOL["main"](
            [
                "--baseline",
                str(tmp_path),
                "--candidate",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
    assert output.read_text() == "original"


def test_cache_counts_and_metric_time_must_be_valid(tmp_path):
    path = tmp_path / "p.log"
    path.write_text(event(lmcache_cached_tokens=101))
    with pytest.raises(ValueError, match="cache counts"):
        TOOL["prefill"](TOOL["read_node"](path, "P"), 10)
    path.write_text(metric(10) + metric(5))
    with pytest.raises(ValueError, match="non-monotonic"):
        TOOL["read_node"](path, "D")


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan")])
def test_invalid_window_rejected(tmp_path, seconds):
    with pytest.raises(ValueError, match="finite and positive"):
        TOOL["analyze"](tmp_path, tmp_path, seconds)
