# SPDX-License-Identifier: Apache-2.0
"""Collect a reproducible, non-importing P0 source inventory (stdlib only).

No source checkout is changed. No dependencies, tags, LFS or submodules are
downloaded. Output directories must be new. Bundles contain only the current
branch's reachable history, not all branches. Static findings are candidates,
not proof of runtime reachability, compatibility or safe deletion.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPOS = ("vllm", "vllm-ascend", "LMCache", "LMCache-Ascend")
PACKAGES = dict(zip(REPOS, ("vllm", "vllm_ascend", "lmcache", "lmcache_ascend")))
MODEL_ALLOWLIST = (
    "DeepseekForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "GlmMoeDsaForCausalLM",
    "ChatGLMModel",
    "ChatGLMForConditionalGeneration",
    "GlmForCausalLM",
    "Glm4ForCausalLM",
    "Glm4MoeForCausalLM",
    "Glm4MoeLiteForCausalLM",
    "DeepSeekMTPModel",
    "Glm4MoeMTPModel",
    "Glm4MoeLiteMTPModel",
)
CONDITIONAL_DRAFTS = (
    "Eagle3DeepseekV2ForCausalLM",
    "Eagle3DeepseekV3ForCausalLM",
    "EagleDeepSeekMTPModel",
)
DOWNLOAD_RE = re.compile(
    r"FetchContent|ExternalProject|GIT_REPOSITORY|https?://|urlretrieve|"
    r"urlopen|requests\.(get|post)|git (clone|fetch)|pip install|apt-get|yum install"
)


def git(repo: Path, *args: str) -> bytes:
    """Run local Git operations with lazy remote fetch and hooks disabled."""
    env = dict(
        os.environ,
        GIT_NO_LAZY_FETCH="1",
        GIT_TERMINAL_PROMPT="0",
        GIT_LFS_SKIP_SMUDGE="1",
    )
    return subprocess.check_output(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(repo),
            *args,
        ],
        env=env,
        stderr=subprocess.PIPE,
    )


def git_text(repo: Path, *args: str) -> str:
    """Return stripped UTF-8 Git output."""
    return git(repo, *args).decode("utf-8").strip()


def digest_file(path: Path) -> str:
    """Hash a regular file without loading a large bundle into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write a newly generated artifact; never overwrite an existing one."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def expression(node: ast.AST | None) -> str:
    """Render a static expression without evaluating source code."""
    return ast.unparse(node) if node is not None else ""


class PythonInventory(ast.NodeVisitor):
    """Index imports, class contracts and patch candidates with lexical guards."""

    def __init__(self, is_patch_file: bool = False) -> None:
        self.imports: list[dict] = []
        self.classes: list[dict] = []
        self.dynamic_imports: list[dict] = []
        self.patch_candidates: list[dict] = []
        self.patch_calls: list[dict] = []
        self.scope: list[str] = []
        self.guards: list[str] = []
        self.is_patch_file = is_patch_file

    def location(self, node: ast.AST) -> dict:
        """Describe source location and lexical, not runtime, activation guards."""
        return {
            "line": node.lineno,
            "end_line": node.end_lineno,
            "scope": ".".join(self.scope) or "<module>",
            "lexical_guards": list(self.guards),
        }

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        test = expression(node.test)
        for block, guard in ((node.body, test), (node.orelse, f"not ({test})")):
            self.guards.append(guard)
            for child in block:
                self.visit(child)
            self.guards.pop()

    def visit_Import(self, node: ast.Import) -> None:
        self.imports.append(
            {
                **self.location(node),
                "module": None,
                "level": 0,
                "names": [{"name": x.name, "asname": x.asname} for x in node.names],
            }
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.imports.append(
            {
                **self.location(node),
                "module": node.module,
                "level": node.level,
                "names": [{"name": x.name, "asname": x.asname} for x in node.names],
            }
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods = [
            x
            for x in node.body
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        self.classes.append(
            {
                **self.location(node),
                "name": node.name,
                "bases": [expression(x) for x in node.bases],
                "methods": [
                    {
                        "name": x.name,
                        "line": x.lineno,
                        "decorators": [expression(d) for d in x.decorator_list],
                    }
                    for x in methods
                ],
                "super_calls": [
                    {"line": x.lineno, "expression": expression(x)}
                    for x in ast.walk(node)
                    if isinstance(x, ast.Call)
                    and isinstance(x.func, ast.Attribute)
                    and isinstance(x.func.value, ast.Call)
                    and expression(x.func.value.func) == "super"
                ],
            }
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.is_patch_file:
            for target in node.targets:
                if isinstance(target, (ast.Attribute, ast.Subscript)):
                    self.patch_candidates.append(
                        {
                            **self.location(node),
                            "kind": "assignment_candidate",
                            "target": expression(target),
                            "replacement": expression(node.value)[:600],
                            "review_status": "needs_semantic_review",
                        }
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.is_patch_file and isinstance(
            node.target, (ast.Attribute, ast.Subscript)
        ):
            self.patch_candidates.append(
                {
                    **self.location(node),
                    "kind": "annotated_assignment_candidate",
                    "target": expression(node.target),
                    "replacement": expression(node.value)[:600],
                    "review_status": "needs_semantic_review",
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = expression(node.func)
        if name.endswith("import_module") or name == "__import__":
            self.dynamic_imports.append(
                {
                    **self.location(node),
                    "call": expression(node),
                    "literal": bool(
                        node.args and isinstance(node.args[0], ast.Constant)
                    ),
                }
            )
        if (
            name.startswith("_patch_")
            or name.startswith("patch_")
            or name == "run_patches"
        ):
            self.patch_calls.append({**self.location(node), "call": name})
        if self.is_patch_file and (
            name == "setattr"
            or "patch" in name.lower()
            or name.endswith(".register_oot")
        ):
            self.patch_candidates.append(
                {
                    **self.location(node),
                    "kind": "call_candidate",
                    "target": name,
                    "replacement": expression(node)[:600],
                    "review_status": "needs_semantic_review",
                }
            )
        self.generic_visit(node)


def patch_path(repo: str, path: str) -> bool:
    """Include patch directories, injection roots and runner CUDA wrappers."""
    return path.startswith(PACKAGES[repo] + "/") and (
        "/patch/" in path
        or "/patches/" in path
        or path == "lmcache_ascend/__init__.py"
        or path
        in (
            "vllm_ascend/worker/model_runner_v1.py",
            "vllm_ascend/worker/v2/model_runner.py",
        )
    )


def model_registry(tree: ast.AST) -> list[dict]:
    """Extract literal registry entries, including aliases and non-text models."""
    entries = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            try:
                pair = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            if not (
                isinstance(pair, tuple)
                and len(pair) == 2
                and all(isinstance(x, str) for x in pair)
            ):
                continue
            arch = key.value
            disposition = (
                "retain_text_or_mtp"
                if arch in MODEL_ALLOWLIST
                else "target_draft_pending_profile"
                if arch in CONDITIONAL_DRAFTS
                else "remove_after_dependency_extraction"
            )
            entries.append(
                {
                    "architecture": arch,
                    "module": pair[0],
                    "class": pair[1],
                    "line": key.lineno,
                    "disposition": disposition,
                    "runtime_verified": False,
                }
            )
    return entries


def restore_bundle(
    repo: Path, branch: str, head: str, tree: str, destination: Path
) -> dict:
    """Create and verify a current-branch bundle by restoring its Git objects."""
    ref = f"refs/heads/{branch}" if branch else "HEAD"
    # Reachable tags are needed to reproduce setuptools-scm version discovery.
    # They do not add features from another branch.
    tag_names = git_text(repo, "tag", "--merged", head).splitlines()
    git(
        repo,
        "bundle",
        "create",
        str(destination),
        ref,
        *(f"refs/tags/{tag}" for tag in tag_names),
    )
    verification = git_text(repo, "bundle", "verify", str(destination))
    with tempfile.TemporaryDirectory(prefix="falconkv-p0-restore-") as tmp:
        recovered = Path(tmp) / "repository.git"
        args = ["clone", "--bare", "--single-branch"]
        if branch:
            args += ["--branch", branch]
        git(repo, *args, str(destination), str(recovered))
        restored_head = git_text(recovered, "rev-parse", "HEAD")
        restored_tree = git_text(recovered, "rev-parse", "HEAD^{tree}")
        git(recovered, "fsck", "--full")
        if (restored_head, restored_tree) != (head, tree):
            raise RuntimeError(f"Bundle restoration mismatch: {repo.name}")
    return {
        "file": destination.name,
        "sha256": digest_file(destination),
        "size_bytes": destination.stat().st_size,
        "verification": verification,
        "restored_head": restored_head,
        "restored_tree": restored_tree,
        "fsck": "passed",
        "method": "temporary_bare_clone_and_tree_comparison",
        "reachable_tags_included": tag_names,
        "excludes": [
            "other_branch_refs",
            "untracked_files",
            "ignored_files",
            "LFS_payloads",
            "submodule_payloads",
        ],
    }


def collect(workspace: Path, output: Path, bundle_dir: Path | None) -> dict:
    """Freeze clean repository identities and emit static source evidence."""
    # Fail before generating any artifact if source worktrees need preservation.
    for name in REPOS:
        repo = workspace / name
        if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
            raise RuntimeError(
                f"{name}: dirty/untracked worktree; preserve and review it first"
            )
        if git_text(repo, "rev-parse", "--is-shallow-repository") != "false":
            raise RuntimeError(
                f"{name}: shallow history; cannot claim a complete branch backup"
            )
    output.mkdir(parents=True, exist_ok=False)
    if bundle_dir is not None:
        bundle_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_level": "static_source_only",
        "repos": [],
    }
    python_index = []
    patches = []
    dependencies = []
    registry = []
    errors = []
    for name in REPOS:
        repo = workspace / name
        print(f"Scanning {name}", flush=True)
        head = git_text(repo, "rev-parse", "HEAD")
        tree_hash = git_text(repo, "rev-parse", "HEAD^{tree}")
        branch = git_text(repo, "branch", "--show-current")
        record = {
            "repository": name,
            "branch": branch,
            "head": head,
            "tree": tree_hash,
            "describe": git_text(repo, "describe", "--tags", "--always", "HEAD"),
            "commit_time": git_text(repo, "show", "-s", "--format=%cI", "HEAD"),
            "worktree_clean": True,
            "files": [],
            "gitlinks": [],
            "lfs_pointers": [],
            "ignored_paths_not_backed_up": git(
                repo, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
            )
            .decode()
            .split("\0")[:-1],
        }
        for item in git(repo, "ls-tree", "-r", "-z", "HEAD").split(b"\0"):
            if not item:
                continue
            metadata, raw_path = item.split(b"\t", 1)
            mode, kind, blob = metadata.decode().split()
            rel = raw_path.decode()
            path = repo / rel
            file_record = {"path": rel, "mode": mode, "git_object": blob, "type": kind}
            if kind != "blob":
                record["gitlinks"].append(file_record)
                continue
            data = os.readlink(path).encode() if mode == "120000" else path.read_bytes()
            file_record.update(
                size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
            record["files"].append(file_record)
            if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
                record["lfs_pointers"].append(rel)
            if mode == "120000":
                continue
            if rel.endswith(".py"):
                try:
                    parsed = ast.parse(data, filename=f"{name}/{rel}")
                    visitor = PythonInventory(patch_path(name, rel))
                    visitor.visit(parsed)
                except (SyntaxError, ValueError, UnicodeError) as exc:
                    errors.append({"repository": name, "path": rel, "error": str(exc)})
                    continue
                entry = {
                    "repository": name,
                    "path": rel,
                    "imports": visitor.imports,
                    "classes": visitor.classes,
                    "dynamic_imports": visitor.dynamic_imports,
                    "patch_calls": visitor.patch_calls,
                }
                python_index.append(entry)
                if visitor.is_patch_file or visitor.patch_calls:
                    patches.append(
                        {
                            "repository": name,
                            "path": rel,
                            "imports": visitor.imports,
                            "activation_calls": visitor.patch_calls,
                            "candidates": visitor.patch_candidates,
                            "review_status": "static_candidates_runtime_order_unverified",
                        }
                    )
                if rel == "vllm/model_executor/models/registry.py":
                    registry = model_registry(parsed)
            is_build = (
                rel in ("setup.py", "setup.cfg", "pyproject.toml", "requirements.txt")
                or rel.startswith(
                    ("requirements/", "cmake/", "docker/", ".github/", ".buildkite/")
                )
                or path.name == "CMakeLists.txt"
                or rel.endswith(".sh")
            )
            is_runtime = rel.startswith(PACKAGES[name] + "/") and rel.endswith(".py")
            if is_build or is_runtime:
                try:
                    lines = data.decode("utf-8").splitlines()
                except UnicodeError:
                    continue
                for number, line in enumerate(lines, 1):
                    tags = []
                    if is_build and DOWNLOAD_RE.search(line):
                        tags.append("download_or_network_candidate")
                    if is_build and (
                        "torch" in line.lower() or "triton" in line.lower()
                    ):
                        tags.append("framework_dependency")
                    if is_runtime and re.search(
                        r"2\.(?:10|11)|is_torch_equal|torch\._|torch\.library", line
                    ):
                        tags.append("torch_api_compatibility_review")
                    if tags:
                        dependencies.append(
                            {
                                "repository": name,
                                "path": rel,
                                "line": number,
                                "tags": tags,
                                "text": line.strip(),
                                "verified": False,
                            }
                        )
        if bundle_dir is not None:
            print(f"Bundling and restore-verifying {name}", flush=True)
            record["backup"] = restore_bundle(
                repo, branch, head, tree_hash, bundle_dir / f"{name}.bundle"
            )
        if git_text(repo, "rev-parse", "HEAD") != head or git(
            repo, "status", "--porcelain=v1", "--untracked-files=all"
        ):
            raise RuntimeError(
                f"{name}: source changed during collection; do not use this run"
            )
        manifest["repos"].append(record)
    summary = {
        "repositories": len(manifest["repos"]),
        "tracked_files": sum(len(r["files"]) for r in manifest["repos"]),
        "python_files_parsed": len(python_index),
        "parse_errors": errors,
        "patch_files_and_call_sites": len(patches),
        "patch_candidates": sum(len(p["candidates"]) for p in patches),
        "registry_entries": len(registry),
        "dependency_findings": len(dependencies),
        "limitations": [
            "AST does not prove runtime activation, complete dynamic imports or MRO",
            "Assignments in patch files may be ordinary instance updates",
            "No package imports, build, install, network or NPU tests performed",
        ],
    }
    write_json(output / "source-manifest.json", manifest)
    write_json(
        output / "python-dependencies.json",
        {"files": python_index, "parse_errors": errors},
    )
    write_json(
        output / "patch-inventory.json", {"files": patches, "runtime_verified": False}
    )
    write_json(output / "dependency-findings.json", {"findings": dependencies})
    write_json(
        output / "model-registry.json",
        {
            "entries": registry,
            "dispositions": dict(Counter(x["disposition"] for x in registry)),
        },
    )
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    """Parse explicit locations; collection never overwrites an old baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path)
    args = parser.parse_args()
    collect(
        args.workspace.resolve(),
        args.output.resolve(),
        args.bundle_dir.resolve() if args.bundle_dir else None,
    )


if __name__ == "__main__":
    main()
