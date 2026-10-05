#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare/review a paired release campaign; never start services or mark tests run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

DESIGN = Path(__file__).resolve().parents[2]
CASE_IDS = (
    [f"A{i:02}" for i in range(1, 16)]
    + [f"B{i:02}" for i in range(1, 12)]
    + [f"C{i:02}" for i in range(1, 14)]
    + ["PERF-TP8", "PERF-TP4", "SOAK-TP8", "SOAK-TP4", "IMAGE", "RESTORE"]
)
FOUNDATION = ("torch", "torch-npu", "transformers", "triton-ascend")


def read(path: Path) -> dict:
    """Load an explicitly selected JSON record."""
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    """Hash a file without loading the whole artifact into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path: Path, data: dict) -> None:
    """Create a new report, refusing to overwrite previous evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def git(root: Path, *args: str) -> str:
    """Run a read-only Git query and fail on errors."""
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_check(workspace: Path, pair_path: Path) -> dict:
    """Bind a clean checkout and release profile to the reviewed pair, without NPU."""
    pair = read(pair_path)
    if set(pair["repositories"]) != {"vllm", "LMCache"}:
        raise ValueError("Pair must contain exactly vllm and LMCache")
    result = {
        "scope": "source_identity_not_build_or_runtime",
        "repositories": {},
        "pair_sha256": digest(pair_path),
        "errors": [],
    }
    for name, expected in pair["repositories"].items():
        root = workspace / name
        head = git(root, "rev-parse", "HEAD")
        tree = git(root, "rev-parse", "HEAD^{tree}")
        dirty = git(root, "status", "--porcelain", "--untracked-files=no")
        untracked = git(root, "ls-files", "--others", "--exclude-standard").splitlines()
        unexpected = [path for path in untracked if path != "submodule-materials.json"]
        version = tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "version"
        ]
        profile = read(root / "release-profile.json")
        package = "lmcache" if name == "LMCache" else "vllm"
        if (
            head != expected["commit"]
            or tree != expected["tree"]
            or dirty
            or unexpected
            or version != expected["version"]
            or profile["versions"][package] != version
            or profile["candidate"] != pair["candidate"]
        ):
            result["errors"].append(f"{name}: source/profile differs from pinned pair")
        result["repositories"][name] = {
            "commit": head,
            "tree": tree,
            "version": version,
        }
    result["passed"] = not result["errors"]
    return result


def new_campaign(pair_path: Path) -> dict:
    """Create a pending human-reviewed acceptance matrix; never prefill passes."""
    return {
        "schema_version": 1,
        "pair_sha256": digest(pair_path),
        "candidate": read(pair_path)["candidate"],
        "reviewer": "",
        "scope": "human_attested_intranet_evidence_not_automatic_test_execution",
        "cases": {
            case: {"status": "pending", "summary": "", "evidence": []}
            for case in CASE_IDS
        },
    }


def evidence_path(root: Path, row: dict) -> Path:
    """Resolve a nonempty, hashed evidence file strictly inside its evidence root."""
    relative = Path(row["path"])
    target = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not target.is_relative_to(root.resolve())
    ):
        raise ValueError("Evidence path must stay inside the evidence root")
    if (
        not target.is_file()
        or target.stat().st_size == 0
        or digest(target) != row["sha256"]
    ):
        raise ValueError(f"Missing, empty or changed evidence: {relative}")
    return target


def review(campaign: dict, pair_path: Path, evidence_root: Path) -> dict:
    """Check coverage, hashes and attestation, not the truth of arbitrary log prose."""
    errors = []
    if campaign.get("schema_version") != 1 or campaign.get("pair_sha256") != digest(
        pair_path
    ):
        errors.append("Campaign is not bound to this exact pair manifest")
    if campaign.get("candidate") != read(pair_path)["candidate"]:
        errors.append("Wrong candidate")
    if (
        not isinstance(campaign.get("reviewer"), str)
        or not campaign["reviewer"].strip()
    ):
        errors.append("Named intranet reviewer is required")
    cases = campaign.get("cases", {})
    if set(cases) != set(CASE_IDS):
        errors.append(
            "Case set differs from the required matrix; no omitted cases or waivers"
        )
    for case in CASE_IDS:
        row = cases.get(case, {})
        if (
            not isinstance(row, dict)
            or row.get("status") != "pass"
            or not isinstance(row.get("summary"), str)
            or not row["summary"].strip()
        ):
            errors.append(f"{case}: pending/failed or missing human conclusion")
            continue
        if not isinstance(row.get("evidence"), list) or not row["evidence"]:
            errors.append(f"{case}: no evidence")
            continue
        for evidence in row["evidence"]:
            try:
                evidence_path(evidence_root, evidence)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{case}: {exc}")
    return {
        "scope": "coverage_hashes_and_human_attestation_not_independent_runtime_proof",
        "pair_sha256": digest(pair_path),
        "errors": errors,
        "passed": not errors,
        "production_cutover_performed": False,
        "old_repositories_retired": False,
    }


def run(command: list[str], output: Path, name: str) -> None:
    """Execute a build/check with an immutable command, log and exit-status record."""
    with (output / f"{name}.log").open("x") as log:
        process = subprocess.run(
            command, cwd=output, stdout=log, stderr=subprocess.STDOUT
        )
    save(
        output / f"{name}.command.json",
        {"argv": command, "returncode": process.returncode},
    )
    if process.returncode:
        raise RuntimeError(
            f"{name} failed ({process.returncode}); see {output / (name + '.log')}"
        )


def unpack(archive: Path, destination: Path, expected_root: str) -> Path:
    """Extract a self-contained sdist, rejecting traversal, links and special files."""
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        seen = set()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != expected_root
                or member.name in seen
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe/duplicate sdist member: {member.name}")
            seen.add(member.name)
        tar.extractall(destination, members=members, filter="data")
    source = destination / expected_root
    if not (source / "pyproject.toml").is_file():
        raise ValueError("sdist lacks the root project")
    return source


def foundation() -> dict:
    """Record protected installed distributions without importing torch or NPU."""
    return {
        name: {
            "version": metadata.version(name),
            "path": str(metadata.distribution(name).locate_file("")),
        }
        for name in FOUNDATION
    }


def build(workspace: Path, pair_path: Path, output: Path) -> None:
    """Build two wheels/sdists and rebuild from sdists; intranet aarch64 only.

    Uses existing p1_dev.py and the narrowly approved pip waiver checker. Does not
    install frameworks, fetch dependencies, access a device or delete any files.
    """
    if sys.version_info[:2] != (3, 11) or platform.machine() != "aarch64":
        raise ValueError(
            "Build only in prepared Python 3.11/aarch64 intranet container"
        )
    report = source_check(workspace, pair_path)
    if not report["passed"]:
        raise ValueError(report["errors"])
    if output == workspace or output.is_relative_to(workspace):
        raise ValueError("Keep build reports outside the paired source workspace")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "source.json", report)
    before = foundation()
    save(output / "foundation-before.json", before)
    command = [sys.executable, "-B"]
    pip_check = str(DESIGN / "p1/tools/check_pip_dependencies.py")
    run(
        [*command, pip_check, "--output", str(output / "pip-before")],
        output,
        "pip-before",
    )
    artifacts = {}
    for name, expected in read(pair_path)["repositories"].items():
        source = workspace / name
        package = "lmcache" if name == "LMCache" else "vllm"
        run(
            [*command, str(source / "tools/check_native_layout.py")],
            output,
            f"materials-{name}",
        )
        run(
            [
                *command,
                str(source / "p1_dev.py"),
                "build",
                "--output",
                str(output / f"wheel-{name}"),
            ],
            output,
            f"wheel-{name}",
        )
        sdist_dir = output / f"sdist-{name}"
        run(
            [
                *command,
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(sdist_dir),
                str(source),
            ],
            output,
            f"sdist-{name}",
        )
        stem = f"{package}-{expected['version']}"
        archive = sdist_dir / f"{stem}.tar.gz"
        unpacked = unpack(archive, output / f"unpack-{name}", stem)
        run(
            [*command, str(unpacked / "tools/check_native_layout.py")],
            output,
            f"sdist-materials-{name}",
        )
        run(
            [
                *command,
                str(unpacked / "p1_dev.py"),
                "build",
                "--output",
                str(output / f"rebuild-{name}"),
            ],
            output,
            f"rebuild-{name}",
        )
        paths = [archive]
        for directory in (f"wheel-{name}", f"rebuild-{name}"):
            paths.append(Path(read(output / directory / "artifact.json")["wheel"]))
        artifacts[name] = [
            {"path": str(path.relative_to(output)), "sha256": digest(path)}
            for path in paths
        ]
    after = foundation()
    save(output / "foundation-after.json", after)
    if before != after:
        raise ValueError("Protected foundation distributions changed during the build")
    run(
        [*command, pip_check, "--output", str(output / "pip-after")],
        output,
        "pip-after",
    )
    final_source = source_check(workspace, pair_path)
    if not final_source["passed"]:
        raise ValueError("Sources changed during the build")
    save(
        output / "artifacts.json",
        {
            "scope": "build_and_sdist_rebuild_not_ABI_or_runtime",
            "pair_sha256": digest(pair_path),
            "repositories": artifacts,
            "passed": True,
        },
    )


def main() -> int:
    """Run one explicit action; failures retain evidence and exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("source", "init", "review", "build"))
    parser.add_argument("--pair", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    try:
        if args.action in ("source", "build") and args.workspace is None:
            raise ValueError("--workspace required")
        if args.action == "build":
            build(args.workspace.resolve(), args.pair.resolve(), args.output.resolve())
            return 0
        if args.action == "source":
            report = source_check(args.workspace.resolve(), args.pair)
        elif args.action == "init":
            report = new_campaign(args.pair)
        else:
            if args.campaign is None or args.evidence_root is None:
                raise ValueError("--campaign and --evidence-root required")
            report = review(read(args.campaign), args.pair, args.evidence_root)
        save(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if args.action == "init" or report["passed"] else 1
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
