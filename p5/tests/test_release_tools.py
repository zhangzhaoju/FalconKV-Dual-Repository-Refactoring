# SPDX-License-Identifier: Apache-2.0
"""Host-only synthetic fixtures; these tests are not intranet qualification."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "p5/tools"))
import compare_perf  # noqa: E402
import prepare_image  # noqa: E402
import qualify  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "backup_sources", ROOT / "p6/tools/backup_sources.py"
)
backup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backup)


@pytest.fixture
def pair(tmp_path: Path) -> Path:
    path = tmp_path / "pair.json"
    qualify.save(
        path,
        {
            "candidate": "fixture",
            "repositories": {
                "vllm": {"version": "1+fixture"},
                "LMCache": {"version": "2+fixture"},
            },
        },
    )
    return path


def test_new_matrix_never_claims_pass(pair: Path) -> None:
    campaign = qualify.new_campaign(pair)
    assert len(campaign["cases"]) == 45
    assert all(row["status"] == "pending" for row in campaign["cases"].values())
    assert not qualify.review(campaign, pair, pair.parent)["passed"]


def test_review_requires_exact_pair_and_coverage_and_hashes(pair: Path) -> None:
    evidence = pair.parent / "fixture-evidence.txt"
    evidence.write_text("synthetic test fixture, not a real test result")
    campaign = qualify.new_campaign(pair)
    campaign["reviewer"] = "fixture reviewer"
    for row in campaign["cases"].values():
        row.update(
            status="pass",
            summary="synthetic",
            evidence=[{"path": evidence.name, "sha256": qualify.digest(evidence)}],
        )
    assert qualify.review(campaign, pair, pair.parent)["passed"]
    campaign["cases"].pop("B11")
    assert not qualify.review(campaign, pair, pair.parent)["passed"]
    campaign["pair_sha256"] = "0" * 64
    assert not qualify.review(campaign, pair, pair.parent)["passed"]


@pytest.mark.parametrize("relative", ["../other", "/tmp/other"])
def test_evidence_cannot_escape(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError):
        qualify.evidence_path(tmp_path, {"path": relative, "sha256": "0" * 64})


def test_report_never_overwrites(pair: Path) -> None:
    with pytest.raises(FileExistsError):
        qualify.save(pair, {})


@pytest.mark.parametrize(
    "name,kind",
    [("../escape", "file"), ("pkg/escape", "symlink"), ("elsewhere/a", "file")],
)
def test_unpack_rejects_unsafe_members(tmp_path: Path, name: str, kind: str) -> None:
    path = tmp_path / "input.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        if kind == "symlink":
            info.type = tarfile.SYMTYPE
            info.linkname = "../../escape"
            tar.addfile(info)
        else:
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        qualify.unpack(path, tmp_path / "unpack", "pkg")


def campaign() -> dict:
    row = {
        "completed": True,
        "interrupted": False,
        "expected_requests": 1,
        "window_start_monotonic_ns": 0,
        "window_end_monotonic_ns": 10_000_000_000,
        "requests": [
            {
                "id": "one",
                "status": "ok",
                "input_tokens": 3,
                "input_sha256": "1" * 64,
                "output_tokens": 2,
                "ttft_ms": 500,
                "e2e_ms": 1000,
                "itl_ms": [500],
            }
        ],
    }
    return {
        "schema_version": 1,
        "profile_sha256": "2" * 64,
        "workload_sha256": "3" * 64,
        "pair_sha256": "4" * 64,
        "topology": "TP8/DP2",
        "cache_phase": "cold",
        "runs": [dict(copy.deepcopy(row), round=str(i)) for i in range(3)],
    }


def test_perf_uses_wall_time_and_surfaces_tail_metrics() -> None:
    before, after = campaign(), campaign()
    report = compare_perf.compare(before, after, 5)
    assert report["passed"]
    assert report["metrics"]["request_throughput"]["candidate_median"] == 0.1
    assert report["metrics"]["output_token_throughput"]["candidate_median"] == 0.2
    assert "ttft_p99_ms" in report["metrics"]
    for row in after["runs"]:
        row["window_end_monotonic_ns"] = 20_000_000_000
    assert not compare_perf.compare(before, after, 5)["passed"]


@pytest.mark.parametrize(
    "change",
    [
        "interrupted",
        "zero",
        "failed",
        "duplicate",
        "missing",
        "nan",
        "wrong_input",
        "chunk_itl",
        "two_rounds",
        "wrong_phase",
    ],
)
def test_perf_rejects_unqualified_evidence(change: str) -> None:
    before, after = campaign(), campaign()
    run, row = after["runs"][0], after["runs"][0]["requests"][0]
    if change == "interrupted":
        run["interrupted"] = True
    elif change == "zero":
        row["output_tokens"] = 0
    elif change == "failed":
        row["status"] = "http400"
    elif change == "duplicate":
        run["requests"].append(copy.deepcopy(row))
        run["expected_requests"] = 2
    elif change == "missing":
        run["expected_requests"] = 2
    elif change == "nan":
        row["ttft_ms"] = float("nan")
    elif change == "wrong_input":
        row["input_sha256"] = "0" * 64
    elif change == "chunk_itl":
        row["output_tokens"] = 4
    elif change == "two_rounds":
        after["runs"].pop()
    else:
        after["cache_phase"] = "warm"
    with pytest.raises(ValueError):
        compare_perf.compare(before, after, 5)


def make_repo(path: Path) -> None:
    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    (path / "source.txt").write_text("baseline\n")
    subprocess.run(["git", "-C", str(path), "add", "source.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )


def test_backup_restore_is_read_only_for_source(tmp_path: Path) -> None:
    root, archive = tmp_path / "source", tmp_path / "archive"
    make_repo(root)
    head = backup.git(root, "rev-parse", "HEAD")
    assert backup.create({"fixture": root}, archive)["passed"]
    report = backup.verify(archive, tmp_path / "restore")
    assert report["passed"]
    assert report["production_recovery_verified"] is False
    assert report["old_checkouts_may_be_deleted"] is False
    assert backup.git(root, "rev-parse", "HEAD") == head
    assert not backup.git(root, "status", "--porcelain")
    with (archive / "000.bundle").open("ab") as stream:
        stream.write(b"changed")
    assert not backup.verify(archive, tmp_path / "restore-corrupt")["passed"]


def test_backup_refuses_dirty_sources_and_nested_destination(tmp_path: Path) -> None:
    root = tmp_path / "source"
    make_repo(root)
    with pytest.raises(ValueError):
        backup.create({"fixture": root}, root / "archive")
    (root / "untracked.txt").write_text("must preserve")
    report = backup.create({"fixture": root}, tmp_path / "archive")
    assert not report["passed"]
    assert (root / "untracked.txt").read_text() == "must preserve"


def test_backup_recovers_a_complete_submodule_without_fetching(tmp_path: Path) -> None:
    parent, child = tmp_path / "parent", tmp_path / "child"
    make_repo(parent)
    make_repo(child)
    subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(child),
            "deps/child",
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-am",
            "pin child",
        ],
        check=True,
        capture_output=True,
    )
    archive = tmp_path / "archive"
    report = backup.create({"parent": parent}, archive)
    assert report["passed"] and len(report["entries"]) == 2
    restored = backup.verify(archive, tmp_path / "restored")
    assert restored["passed"]
    assert (
        tmp_path / "restored/repo-000/deps/child/source.txt"
    ).read_text() == "baseline\n"


def test_backup_fails_closed_on_missing_submodule_payload(tmp_path: Path) -> None:
    parent, child = tmp_path / "parent", tmp_path / "child"
    make_repo(parent)
    make_repo(child)
    pin = backup.git(child, "rev-parse", "HEAD")
    subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{pin},deps/missing",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(parent),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "missing child",
        ],
        check=True,
        capture_output=True,
    )
    archive = tmp_path / "archive"
    assert not backup.create({"parent": parent}, archive)["passed"]
    assert not backup.verify(archive, tmp_path / "restored")["passed"]


def test_source_pair_gate_rejects_branch_drift_and_untracked_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repos"
    workspace.mkdir()
    pair = {"candidate": "fixture", "repositories": {}}
    for name in ("vllm", "LMCache"):
        root = workspace / name
        make_repo(root)
        package = name.lower()
        (root / "pyproject.toml").write_text(
            f'[project]\nname="{package}"\nversion="1"\n'
        )
        (root / "release-profile.json").write_text(
            json.dumps({"candidate": "fixture", "versions": {package: "1"}})
        )
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-m",
                "profile",
            ],
            check=True,
            capture_output=True,
        )
        pair["repositories"][name] = {
            "commit": backup.git(root, "rev-parse", "HEAD"),
            "tree": backup.git(root, "rev-parse", "HEAD^{tree}"),
            "version": "1",
        }
    pair_path = tmp_path / "pair.json"
    qualify.save(pair_path, pair)
    assert qualify.source_check(workspace, pair_path)["passed"]
    (workspace / "vllm/submodule-materials.json").write_text("{}")
    assert qualify.source_check(workspace, pair_path)["passed"]
    (workspace / "vllm/extra.py").write_text("# unreviewed")
    assert not qualify.source_check(workspace, pair_path)["passed"]
    pair["repositories"]["LMCache"]["commit"] = "0" * 40
    drift = tmp_path / "pair-drift.json"
    qualify.save(drift, pair)
    assert not qualify.source_check(workspace, drift)["passed"]


def test_image_requires_pinned_base_and_exact_artifact_pair(pair: Path) -> None:
    with pytest.raises(ValueError):
        prepare_image.prepare(pair, pair, "base:latest", pair.parent / "image")
    artifacts = pair.parent / "artifacts.json"
    qualify.save(artifacts, {"passed": True, "pair_sha256": "wrong"})
    with pytest.raises(ValueError):
        prepare_image.prepare(
            pair, artifacts, "base@sha256:" + "1" * 64, pair.parent / "image"
        )


def test_image_context_contains_only_exact_wheels(pair: Path) -> None:
    rows = {}
    for repo, package, version in (
        ("vllm", "vllm", "1+fixture"),
        ("LMCache", "lmcache", "2+fixture"),
    ):
        directory = pair.parent / f"wheel-{repo}"
        directory.mkdir()
        path = directory / f"{package}-{version}-cp311-cp311-linux_aarch64.whl"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(
                f"{package}-{version}.dist-info/METADATA",
                f"Name: {package}\nVersion: {version}\n",
            )
        rows[repo] = [
            {"path": str(path.relative_to(pair.parent)), "sha256": qualify.digest(path)}
        ]
    artifacts = pair.parent / "artifacts.json"
    qualify.save(
        artifacts,
        {"passed": True, "pair_sha256": qualify.digest(pair), "repositories": rows},
    )
    report = prepare_image.prepare(
        pair, artifacts, "base@sha256:" + "1" * 64, pair.parent / "image"
    )
    assert report["image_built"] is False
    assert set(report["wheels"]) == {"vllm", "lmcache"}
    assert (pair.parent / "image/install_image.py").is_file()
