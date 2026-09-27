#!/usr/bin/env python3
"""Reproduce the 2026-09-26 baseline/P1 log comparison using the standard library."""

import hashlib
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
RUNS = {
    "baseline": "基线_2P2D_20K_140K_测试",
    "p1": "P1_2P2D_20K_140K_测试",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
EVENT = re.compile(r"\[(LMCACHE_COLD_PERF|LMCACHE_REMOTE_FILL_DIAGNOSTIC)\] (\{.*\})")
METRICS = ("ttft_ms", "tpot_ms", "itl_ms", "latency_seconds")


def quantile(values, fraction):
    """Use linear interpolation at (n - 1) * fraction, matching these reports."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values):
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "p50": quantile(values, 0.50),
        "p95": quantile(values, 0.95),
        "p99": quantile(values, 0.99),
    }


def evidence(lines, predicate):
    hits = [{"line": i, "text": text[:1400]}
            for i, text in enumerate(lines, 1) if predicate(text)]
    return {"line_count": len(hits), "first": hits[:1], "last": hits[-1:]}


def normalized_args(lines):
    args = next((line.split("non-default args: ", 1)[1] for line in lines
                 if "non-default args: " in line), None)
    if args is None:
        return None
    return re.sub(r"torch_profiler_dir='[^']*'", "torch_profiler_dir='<PATH>'", args)


def main():
    result = {
        "scope": "Only the two 20K_140K directories; no NPU rerun or deployment.",
        "quantiles": "Linear interpolation at (n-1)*p; ITL is per-request itl_ms.",
        "runs": {},
        "manifest": [],
    }
    rows_by_run, arguments = {}, {}
    for name, directory in RUNS.items():
        folder = ROOT / "p1_logs" / directory
        benchmark_path, = folder.glob("*benchmark*.json")
        client_path, = folder.glob("*ansible*.log")
        benchmark = json.loads(benchmark_path.read_text())
        rows = benchmark["results"]
        assert len({row["id"] for row in rows}) == len(rows)
        assert sum(row["success"] for row in rows) == benchmark["completed"]
        assert sum(not row["success"] for row in rows) == benchmark["failed"]
        for token_type in ("input", "output"):
            assert sum(row[f"{token_type}_tokens"] for row in rows) == benchmark[f"total_{token_type}_tokens"]
        rows_by_run[name] = {row["id"]: row for row in rows}
        client_lines = client_path.read_text().splitlines()
        planned = int(re.search(r"Loaded \d+ files, (\d+) total requests", "\n".join(client_lines))[1])
        valid_rows = [row for row in rows if row["input_tokens"] > 0 and row["output_tokens"] > 0]
        run = {
            "benchmark_file": str(benchmark_path.relative_to(ROOT)),
            "reported": {key: value for key, value in benchmark.items() if key != "results"},
            "planned_requests": planned,
            "recorded_fraction_percent": len(rows) / planned * 100,
            "sum_request_latency_seconds": sum(row["latency_seconds"] for row in rows),
            "zero_token_success_ids": [row["id"] for row in rows if row["success"] and row["input_tokens"] == row["output_tokens"] == 0],
            "positive_token_rows": len(valid_rows),
            "positive_input_range": [min(row["input_tokens"] for row in valid_rows), max(row["input_tokens"] for row in valid_rows)],
            "input_files": [line.removeprefix("Loading: ") for line in client_lines if line.startswith("Loading: ")],
            "interrupt_evidence": evidence(client_lines, lambda line: "Received Ctrl+C" in line or "Interrupted!" in line),
            "all_recorded_stats": {key: describe([row[key] for row in rows if row[key] is not None]) for key in METRICS},
            "servers": {},
        }
        run["duration_minus_sum_latency_seconds"] = benchmark["duration"] - run["sum_request_latency_seconds"]
        # Cross-check the reported aggregates; these checks do not validate the benchmark's measurement semantics.
        for metric in METRICS[:3]:
            for reported_key, calculated_key in (("mean", "mean"), ("median", "p50"), ("p99", "p99")):
                assert abs(benchmark[metric][reported_key] - run["all_recorded_stats"][metric][calculated_key]) < 0.011
        arguments[name] = {}
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            raw = path.read_bytes()
            result["manifest"].append({"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
            if path.suffix != ".log" or "ansible" in path.name:
                continue
            lines = ANSI.sub("", raw.decode("utf-8", errors="replace")).splitlines()
            arguments[name][path.name] = normalized_args(lines)
            markers = {
                "fatal": "EngineCore encountered a fatal error.",
                "sample_timeout": "TimeoutError: RPC call to sample_tokens timed out.",
                "worker_exception": "WorkerProc hit an exception.",
                "shared_error": "ValueError: Shared CPU cache rank0 error envelope:",
                "http400": '"POST /v1/chat/completions HTTP/1.1" 400 Bad Request',
                "context_rejection": "VLLMValidationError: This model's maximum context length",
                "cpu_allocation_warning": "Failed to batched allocate",
                "profiler_error": "Call tensorboard_trace_handler failed.",
                "scheduler_at_failure": "Dumping scheduler stats:",
                "generated_activity": "Activity: generated",
                "no_engine_updates": "Activity: no_engine_updates",
                "layout": "LMCache NPU payload layout:",
                "groups": "runtime_kv_group_layer_counts=",
                "sampling": "SamplingParams(",
            }
            server = {key: evidence(lines, lambda line, marker=marker: marker in line) for key, marker in markers.items()}
            diagnostics, bad_gets, routes = [], [], {}
            passive_ranks = set()
            for line_number, line in enumerate(lines, 1):
                if markers["shared_error"] in line:
                    rank = re.search(r"Worker_DP\d+_TP(\d+)", line)
                    if rank:
                        passive_ranks.add(int(rank[1]))
                match = EVENT.search(line)
                if not match:
                    continue
                event = json.loads(match[2])
                if match[1] == "LMCACHE_REMOTE_FILL_DIAGNOSTIC":
                    diagnostics.append({"line": line_number, "event": event})
                if event.get("event") == "mooncake_page_get" and event.get("status") != "ok":
                    bad_gets.append({"line": line_number, "event": event})
                if event.get("event") == "decoder_execution_route":
                    route = str((event.get("runtime_graph_mode"), event.get("staged_action")))
                    routes[route] = routes.get(route, 0) + 1
            server.update(diagnostics=diagnostics, bad_page_gets=bad_gets,
                          passive_error_ranks=sorted(passive_ranks), execution_route_log_counts=routes)
            run["servers"][path.name] = server
        result["runs"][name] = run
    baseline, p1 = rows_by_run["baseline"], rows_by_run["p1"]
    common = sorted(baseline.keys() & p1.keys())
    result["comparison"] = {
        "common_ids": len(common),
        "baseline_only_ids": sorted(baseline.keys() - p1.keys()),
        "p1_only_ids": sorted(p1.keys() - baseline.keys()),
        "common_row_mismatch_counts": {key: sum(baseline[i][key] != p1[i][key] for i in common)
                                       for key in ("source_file", "input_tokens", "output_tokens", "context_chars", "context_turns", "num_messages", "has_tools")},
        "same_input_file_list": result["runs"]["baseline"]["input_files"] == result["runs"]["p1"]["input_files"],
        "same_nondefault_args_except_profiler_path": {name: value == arguments["p1"].get(name)
                                                     for name, value in arguments["baseline"].items() if value is not None},
        "paired_positive_output_stats": {},
    }
    for metric in METRICS:
        ids = [i for i in common if all(rows[i][metric] is not None and rows[i]["output_tokens"] > 0 for rows in (baseline, p1))]
        stats = {name: describe([rows[i][metric] for i in ids]) for name, rows in (("baseline", baseline), ("p1", p1))}
        stats["change_percent"] = {key: (stats["p1"][key] / stats["baseline"][key] - 1) * 100 for key in ("mean", "p50", "p95", "p99")}
        result["comparison"]["paired_positive_output_stats"][metric] = stats
    output = Path(__file__).with_name("comparison.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {output}; checked {len(result['manifest'])} input files and {len(common)} common request IDs.")


if __name__ == "__main__":
    main()
