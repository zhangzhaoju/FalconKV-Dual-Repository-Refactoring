#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare 2P2D log windows, not kernel performance or acceptance criteria.

Read-only, stdlib-only. P clocks are aligned to each node's first lookup; D
clocks to each node's first metric sample. Neither is a shared system window.
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


NODE = re.compile(r"_([PD])_dp([01])_port\d+\.log$")
METRIC = re.compile(
    r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Avg prompt throughput:.*"
    r"Running: (\d+) reqs, Waiting: (\d+) reqs"
)
EVENTS = {"scheduler_lookup", "remote_fill_producer_terminal"}


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def read_node(path, role):
    events, metrics = [], []
    with path.open(encoding="utf-8", errors="strict", newline="\n") as stream:
        for number, line in enumerate(stream, 1):
            if role == "D" and (match := METRIC.search(line)):
                # Logs omit the year. Cross-year or restarted streams must be
                # separated before analysis, not silently joined.
                stamp = datetime.strptime("2000-" + match[1], "%Y-%m-%d %H:%M:%S")
                metrics.append(
                    {
                        "time": match[1],
                        "seconds": (stamp - datetime(2000, 1, 1)).total_seconds(),
                        "running": int(match[2]),
                        "waiting": int(match[3]),
                        "line": number,
                    }
                )
            if role != "P" or "[LMCACHE_COLD_PERF]" not in line:
                continue
            try:
                event, _ = json.JSONDecoder().raw_decode(line[line.index("{") :])
            except ValueError as error:
                raise ValueError(
                    f"{path}:{number}: malformed cold-perf JSON"
                ) from error
            if event.get("event") in EVENTS:
                if not event.get("clock_domain") or not event.get("req_id"):
                    raise ValueError(f"{path}:{number}: missing event identity/clock")
                value = event.get("monotonic_ms")
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{path}:{number}: invalid monotonic_ms")
                events.append({**event, "line": number})
    if role == "P":
        if not events or len({e["clock_domain"] for e in events}) != 1:
            raise ValueError(f"{path}: missing events or mixed clock domains")
        lookups = [e for e in events if e["event"] == "scheduler_lookup"]
        if not lookups or len({e.get("pid") for e in lookups}) != 1:
            raise ValueError(f"{path}: missing lookups or multiple scheduler processes")
    elif not metrics or any(
        b["seconds"] < a["seconds"] for a, b in zip(metrics, metrics[1:])
    ):
        raise ValueError(f"{path}: missing metrics or non-monotonic log timestamps")
    return {"file": identity(path), "events": events, "metrics": metrics}


def read_run(directory):
    nodes = {}
    for path in sorted(directory.glob("*.log")):
        match = NODE.search(path.name)
        if not match:
            continue
        name = match[1] + match[2]
        if name in nodes:
            raise ValueError(f"{directory}: duplicate {name} logs; select one run")
        nodes[name] = read_node(path, match[1])
    if set(nodes) != {"P0", "P1", "D0", "D1"}:
        raise ValueError(
            f"{directory}: require exactly P0/P1/D0/D1, found {sorted(nodes)}"
        )
    return nodes


def queue_summary(samples):
    return {
        "samples": len(samples),
        "first_time": samples[0]["time"] if samples else None,
        "last_time": samples[-1]["time"] if samples else None,
        "max_running": max((s["running"] for s in samples), default=None),
        "max_waiting": max((s["waiting"] for s in samples), default=None),
        "positive_waiting_samples": sum(s["waiting"] > 0 for s in samples),
        "first_positive_waiting": next((s for s in samples if s["waiting"] > 0), None),
    }


def decode(node, seconds):
    samples = node["metrics"]
    start = samples[0]["seconds"]
    return {
        "file": node["file"],
        "coverage_seconds": samples[-1]["seconds"] - start,
        "full_capture": queue_summary(samples),
        "matched_window": queue_summary(
            [s for s in samples if 0 <= s["seconds"] - start < seconds]
        ),
    }


def lookup_tokens(event):
    values = tuple(
        event.get(k)
        for k in ("prompt_tokens", "vllm_cached_tokens", "lmcache_cached_tokens")
    )
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError(f"Invalid lookup token counts on line {event['line']}")
    prompt, local, cached = values
    if local + cached > prompt:
        raise ValueError(f"Overlapping/invalid cache counts on line {event['line']}")
    return prompt, cached, prompt - local - cached


def cache_summary(lookups):
    tokens = [lookup_tokens(e) for e in lookups]
    prompt = sum(t[0] for t in tokens)
    cached = sum(t[1] for t in tokens)
    return {
        "lookup_events": len(tokens),
        "prompt_tokens": prompt,
        "lmcache_cached_tokens": cached,
        "event_weighted_hit_fraction": cached / prompt if prompt else None,
    }


def prefill(node, seconds):
    events = sorted(node["events"], key=lambda e: e["monotonic_ms"])
    lookups = [e for e in events if e["event"] == "scheduler_lookup"]
    start = lookups[0]["monotonic_ms"]
    end = start + seconds * 1000
    latest, seen, completions = {}, set(), []
    unmatched = unsupported = duplicates = 0
    outcomes = Counter()
    for event in events:
        req = event["req_id"]
        if event["event"] == "scheduler_lookup":
            latest[req] = event
            continue
        if not start <= event["monotonic_ms"] < end:
            continue
        key = (req, event.get("transfer_id"))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        outcomes[event.get("outcome", "MISSING")] += 1
        lookup = latest.get(req)
        if lookup is None:
            unmatched += 1
            continue
        # This bounded review supports the full-prompt producer terminal in
        # these logs, not arbitrary retries/resume/direct-transfer protocols.
        prompt, _, uncached = lookup_tokens(lookup)
        if (
            event.get("outcome") != "PERSISTENT_ONLY"
            or event.get("required_store_end") != prompt
            or event.get("persistent_common_end", -1) < prompt
            or lookup.get("resumed")
            or lookup.get("query_scope") != "all_tokens"
        ):
            unsupported += 1
            continue
        completions.append(
            {
                "req_id": req,
                "lookup_line": lookup["line"],
                "terminal_line": event["line"],
                "prompt_tokens": prompt,
                "uncached_tokens": uncached,
            }
        )
    uncached = sum(e["uncached_tokens"] for e in completions)
    return {
        "file": node["file"],
        "clock_domain": lookups[0]["clock_domain"],
        "anchor_lookup_line": lookups[0]["line"],
        "coverage_seconds": (events[-1]["monotonic_ms"] - start) / 1000,
        "window_seconds": seconds,
        "terminal_transfers": len(completions),
        "completed_prompt_tokens": sum(e["prompt_tokens"] for e in completions),
        "completed_uncached_tokens": uncached,
        "closed_request_token_rate_estimate": uncached / seconds,
        "duplicate_terminal_events": duplicates,
        "unmatched_terminals": unmatched,
        "unsupported_terminals": unsupported,
        "terminal_outcomes": dict(outcomes),
        "early_lookup_cache": cache_summary(
            [e for e in lookups if e["monotonic_ms"] < end]
        ),
        "later_lookup_cache": cache_summary(
            [e for e in lookups if e["monotonic_ms"] >= end]
        ),
        "completions": completions,
    }


def analyze(baseline, candidate, seconds=250):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("window-seconds must be finite and positive")
    runs = {"baseline": read_run(baseline), "candidate": read_run(candidate)}
    result = {
        "schema_version": 1,
        "scope": "log_evidence_not_performance_acceptance",
        "requested_window_seconds": seconds,
        "nodes": {},
        "warnings": [],
        "limits": [
            "Windows start independently per node; no cross-host clock subtraction.",
            "D samples are instantaneous and may have gaps; not queue duration.",
            "P terminal is persistence completion, not pure model/kernel timing.",
            "Closed-request estimates omit in-flight boundary work and may count retries.",
            "Lookup hit ratios are event-weighted, not deduplicated request hit rates.",
            "No performance PASS/FAIL or full workload completion is inferred.",
        ],
    }
    for name in ("P0", "P1", "D0", "D1"):
        if name.startswith("D"):
            coverage = [
                r[name]["metrics"][-1]["seconds"] - r[name]["metrics"][0]["seconds"]
                for r in runs.values()
            ]
            window = min(seconds, *coverage)
            if window <= 0:
                raise ValueError(f"{name}: insufficient metric coverage")
            if window < seconds:
                result["warnings"].append(
                    f"{name}: shortened common D window to {window}s"
                )
            summaries = {
                label: decode(run[name], window) for label, run in runs.items()
            }
        else:
            coverage = []
            for run in runs.values():
                events = run[name]["events"]
                anchor = min(
                    e["monotonic_ms"]
                    for e in events
                    if e["event"] == "scheduler_lookup"
                )
                coverage.append(
                    (max(e["monotonic_ms"] for e in events) - anchor) / 1000
                )
            window = min(seconds, *coverage)
            if window <= 0:
                raise ValueError(f"{name}: insufficient P event coverage")
            if window < seconds:
                result["warnings"].append(
                    f"{name}: shortened common P window to {window}s"
                )
            summaries = {
                label: prefill(run[name], window) for label, run in runs.items()
            }
            for label, summary in summaries.items():
                if summary["unmatched_terminals"] or summary["unsupported_terminals"]:
                    result["warnings"].append(
                        f"{name}/{label}: excluded terminal events; inspect counts"
                    )
        result["nodes"][name] = {"matched_window_seconds": window, **summaries}
    result["identical_auxiliary_files_excluded"] = []
    for path in sorted(candidate.iterdir()):
        if (
            path.is_file()
            and ("benchmark" in path.name or "ansible" in path.name)
            and (baseline / path.name).is_file()
        ):
            item = identity(path)
            if item["sha256"] == identity(baseline / path.name)["sha256"]:
                result["identical_auxiliary_files_excluded"].append(item)
    if result["identical_auxiliary_files_excluded"]:
        result["warnings"].append(
            "Identical auxiliary artifacts cannot establish a new candidate run"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=250)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    try:
        result = analyze(args.baseline, args.candidate, args.window_seconds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote evidence (not performance acceptance): {args.output}")


if __name__ == "__main__":
    main()
