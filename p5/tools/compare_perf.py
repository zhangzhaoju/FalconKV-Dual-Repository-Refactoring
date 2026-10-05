#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare completed, repeated fixed-workload runs using measured wall time.

Input is the normalized schema in p5/intranet-validation.md, not the historical
benchmark JSON whose duration summed request latencies. No model calls are made.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from qualify import digest, save


def positive(value: object, name: str, *, zero: bool = False) -> float:
    """Validate finite measurements; do not accept strings, bools or NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}: expected numeric measurement")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not zero):
        raise ValueError(f"{name}: invalid measurement")
    return number


def percentile(values: list[float], q: float) -> float:
    """Linearly interpolate the percentile, including the one-sample case."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def measure(run: dict) -> tuple[dict, dict]:
    """Reject incomplete/error runs and return metrics plus input identities."""
    if run.get("completed") is not True or run.get("interrupted") is not False:
        raise ValueError("Interrupted/incomplete runs cannot qualify performance")
    start = run["window_start_monotonic_ns"]
    end = run["window_end_monotonic_ns"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("Record a real monotonic wall-clock interval")
    wall = (end - start) / 1e9
    requests = run["requests"]
    if (
        type(run["expected_requests"]) is not int
        or not requests
        or len(requests) != run["expected_requests"]
    ):
        raise ValueError("Missing/extra requests in the measured window")
    ids, ttft, tpot, e2e, itl, output_tokens = {}, [], [], [], [], 0
    for row in requests:
        key = row["id"]
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError("Request identities must be nonempty and unique")
        if row.get("status") != "ok":
            raise ValueError("Errors/cancellations/HTTP400 must not count as success")
        count = row["output_tokens"]
        if type(count) is not int or count < 2:
            raise ValueError(
                "Performance workload needs >=2 output tokens; "
                "zero output is not success"
            )
        if type(row["input_tokens"]) is not int or row["input_tokens"] <= 0:
            raise ValueError("Invalid input token count")
        input_hash = row["input_sha256"]
        if (
            not isinstance(input_hash, str)
            or len(input_hash) != 64
            or any(c not in "0123456789abcdef" for c in input_hash)
        ):
            raise ValueError("Record the canonical request input hash")
        ids[key] = (input_hash, row["input_tokens"])
        first = positive(row["ttft_ms"], "ttft_ms", zero=True)
        elapsed = positive(row["e2e_ms"], "e2e_ms")
        if elapsed < first or elapsed > wall * 1000 + 1:
            raise ValueError("Request latency lies outside the measured wall interval")
        intervals = [positive(x, "itl_ms", zero=True) for x in row["itl_ms"]]
        if len(intervals) != count - 1:
            raise ValueError(
                "ITL must contain one interval per subsequent token, not per SSE chunk"
            )
        if sum(intervals) > elapsed - first + 1:
            raise ValueError("ITL sum exceeds post-first-token latency")
        ttft.append(first)
        e2e.append(elapsed)
        tpot.append((elapsed - first) / (count - 1))
        itl.extend(intervals)
        output_tokens += count
    metrics = {
        "request_throughput": len(requests) / wall,
        "output_token_throughput": output_tokens / wall,
    }
    for name, values in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2e", e2e)):
        for label, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
            metrics[f"{name}_{label}_ms"] = percentile(values, q)
    return metrics, ids


def compare(baseline: dict, candidate: dict, limit: float) -> dict:
    """Compare >=3 paired rounds, report every round and median regressions."""
    positive(limit, "regression limit", zero=True)
    if limit >= 100:
        raise ValueError("Percentage limit must be below 100")
    for key in (
        "schema_version",
        "profile_sha256",
        "workload_sha256",
        "topology",
        "cache_phase",
    ):
        if key not in baseline or baseline.get(key) != candidate.get(key):
            raise ValueError(f"Non-comparable campaigns: {key}")
    if baseline["schema_version"] != 1 or baseline["topology"] not in (
        "TP8/DP2",
        "TP4/DP4",
    ):
        raise ValueError("Unsupported schema/topology")
    if baseline["cache_phase"] not in ("cold", "warm"):
        raise ValueError("Separate cold and warm campaigns")
    for document in (baseline, candidate):
        for key in ("profile_sha256", "workload_sha256", "pair_sha256"):
            value = document[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError(f"Missing identity: {key}")
    left, right = baseline["runs"], candidate["runs"]
    if len(left) < 3 or len(left) != len(right):
        raise ValueError("At least three paired independent repetitions required")
    samples = []
    run_ids = set()
    workload = None
    for before, after in zip(left, right, strict=True):
        run_id = before["round"]
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id != after["round"]
            or run_id in run_ids
        ):
            raise ValueError("Rounds must match and be unique")
        run_ids.add(run_id)
        old, old_ids = measure(before)
        new, new_ids = measure(after)
        if old_ids != new_ids or (workload is not None and old_ids != workload):
            raise ValueError("Input IDs/tokens differ between rounds or versions")
        workload = old_ids
        samples.append({"round": run_id, "baseline": old, "candidate": new})
    metrics = {}
    for name in samples[0]["baseline"]:
        old = statistics.median(row["baseline"][name] for row in samples)
        new = statistics.median(row["candidate"][name] for row in samples)
        # Higher throughput is better, lower latency is better.
        regression = old - new if "throughput" in name else new - old
        percent = (regression / old * 100) if old else (0.0 if new == 0 else None)
        metrics[name] = {
            "baseline_median": old,
            "candidate_median": new,
            "regression_pct": percent,
            "passed": percent is not None and percent <= limit,
        }
    return {
        "scope": "fixed_workload_performance_not_functionality_or_cutover",
        "pair_sha256": candidate["pair_sha256"],
        "baseline_pair_sha256": baseline["pair_sha256"],
        "topology": baseline["topology"],
        "cache_phase": baseline["cache_phase"],
        "regression_limit_pct": limit,
        "rounds": samples,
        "metrics": metrics,
        "passed": all(row["passed"] for row in metrics.values()),
    }


def main() -> int:
    """Read normalized evidence; the threshold must be chosen before the runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--baseline-pair", type=Path, required=True)
    parser.add_argument("--max-regression-pct", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        before = json.loads(args.baseline.read_text())
        after = json.loads(args.candidate.read_text())
        if before["pair_sha256"] != digest(args.baseline_pair) or after[
            "pair_sha256"
        ] != digest(args.pair):
            raise ValueError("Evidence does not identify the selected source pairs")
        report = compare(before, after, args.max_regression_pct)
        report["input_sha256"] = {
            "baseline": digest(args.baseline),
            "candidate": digest(args.candidate),
        }
        save(args.output, report)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
