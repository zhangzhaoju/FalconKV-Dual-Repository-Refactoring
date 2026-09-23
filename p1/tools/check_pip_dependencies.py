# SPDX-License-Identifier: Apache-2.0
"""Run read-only pip check with three explicitly approved SDK metadata waivers.

Only op-compile-tool 0.1.0's missing getopt/inspect/multiprocessing declarations
are waived. Raw output/return code are retained; all other diagnostics fail.
Does not install packages, edit distribution metadata, or initialize an NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib import util
from pathlib import Path

STDLIB_MODULES = ("getopt", "inspect", "multiprocessing")
APPROVED_MESSAGES = {
    f"op-compile-tool 0.1.0 requires {name}, which is not installed.": name
    for name in STDLIB_MODULES
}
SUCCESS_MESSAGE = "No broken requirements found."


def stdlib_availability() -> dict[str, bool]:
    """Check module discovery, not distribution metadata, without importing them."""
    available = {}
    for name in STDLIB_MODULES:
        try:
            available[name] = (
                name in sys.stdlib_module_names and util.find_spec(name) is not None
            )
        except (ImportError, ValueError):
            available[name] = False
    return available


def assess(returncode: int | None, output: str, available: dict[str, bool]) -> dict:
    """Accept exact approved missing-distribution messages, never other failures."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    clean = returncode == 0 and lines == [SUCCESS_MESSAGE]
    waived = [
        line
        for line in lines
        if returncode == 1
        and line in APPROVED_MESSAGES
        and available.get(APPROVED_MESSAGES[line], False)
    ]
    errors = [] if clean else [line for line in lines if line not in waived]
    if returncode not in (0, 1):
        errors.append(f"pip check did not complete normally: returncode={returncode}")
    elif not clean and not lines:
        errors.append("pip check returned no dependency diagnostics")
    if returncode == 0 and not clean:
        errors.append("Unexpected output with successful pip check exit code")
    passed = clean or (returncode == 1 and bool(waived) and not errors)
    return {
        "scope": "installed_dependency_metadata_not_ABI_or_runtime",
        "approval": "user_approved_op_compile_tool_0.1.0_stdlib_20260923",
        "approved_messages": list(APPROVED_MESSAGES),
        "stdlib_available": available,
        "raw_returncode": returncode,
        "waived_messages": waived,
        "blocking_errors": errors,
        "status": "passed_with_waivers"
        if passed and waived
        else "passed"
        if passed
        else "failed",
        "passed": passed,
    }


def run_check(output: Path) -> dict:
    """Check the current interpreter; create a new report directory, never overwrite."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-B",
        "-m",
        "pip",
        "--disable-pip-version-check",
        "--no-color",
        "check",
    ]
    process_error = None
    try:
        process = subprocess.run(
            command,
            cwd=output,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
        raw, code = process.stdout, process.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        raw = exc.output or b"" if isinstance(exc, subprocess.TimeoutExpired) else b""
        code, process_error = None, str(exc)
    with (output / "pip-check.txt").open("xb") as stream:
        stream.write(raw)
    with (output / "pip-check.exitcode").open("x", encoding="utf-8") as stream:
        stream.write(f"{code if code is not None else 'unavailable'}\n")
    report = assess(code, raw.decode("utf-8", errors="replace"), stdlib_availability())
    if process_error:
        report["blocking_errors"].append(process_error)
    report.update(
        {
            "schema_version": 1,
            "command": command,
            "python": sys.executable,
            "output": str(output),
            "raw_output_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    """Return the policy verdict; retain pip's separate raw exit code in the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new report directory"
    )
    args = parser.parse_args(argv)
    try:
        report = run_check(args.output)
    except OSError as exc:
        print(f"P1 STOP: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
