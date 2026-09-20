# SPDX-License-Identifier: Apache-2.0
"""Project the immutable P0 inventory onto the CURRENT approved GLM-5.2 scope.

These maps are review inputs, never executable deletion/rename instructions.
No framework is imported or modified. Output must be a new directory.
"""

from __future__ import annotations

import argparse
import configparser
import json
from pathlib import Path

from source_inventory import write_json


def disposition(repo: str, path: str) -> dict:
    """Classify a source path conservatively; unresolved closure blocks deletion."""
    target_repo = "vllm" if repo.startswith("vllm") else "LMCache"
    result = {
        "source_repository": repo,
        "source_path": path,
        "target_repository": target_repo,
        "target_path": path,
        "action": "review_dependency_closure",
        "ready_to_execute": False,
        "reason": "Confirm relevance to GLM-5.2 and the approved inference/cache paths",
        "validation": ["A02", "A05"],
    }
    if path.endswith(("LICENSE", "NOTICE")) or "licenses/" in path.lower():
        result.update(
            action="preserve_provenance",
            reason="Retain license and copyright provenance",
        )
    elif repo == "vllm" and path.startswith("vllm/model_executor/models/"):
        name = Path(path).name
        if name in ("deepseek_v2.py", "deepseek_mtp.py"):
            result.update(
                action="retain_glm52_implementation_components",
                reason="GLM52 DSA and its internal MTP share these implementations; remove non-target public registrations later",
                validation=["B01", "B04", "B11", "C03", "C05"],
            )
        elif name in ("llama.py", "llama_eagle3.py", "eagle.py", "eagle3.py"):
            result.update(
                action="extract_required_components_then_remove_public_model",
                reason="Follow GLM52 MTP/proposer closure before removing foreign model implementations",
                validation=["B11", "A10"],
            )
        elif name in (
            "__init__.py",
            "registry.py",
            "utils.py",
            "interfaces.py",
            "interfaces_base.py",
        ):
            result.update(
                action="retain_and_narrow_common_infrastructure",
                validation=["A08", "A10"],
            )
        else:
            result.update(
                action="non_target_model_candidate_remove_after_closure",
                reason="GLM52-only public support; shared helpers must be extracted before deletion",
                validation=["A08", "A10", "B11"],
            )
    elif repo == "vllm-ascend" and path.startswith("vllm_ascend/"):
        suffix = path.removeprefix("vllm_ascend/")
        if suffix.startswith("patch/"):
            result.update(
                action="inline_effects_or_remove_non_target_patch",
                target_path=None,
                reason="Semantic ownership and activation order must be reviewed per patch candidate",
                validation=["A03", "B04", "B11", "C10"],
            )
        elif suffix.startswith("_310p/"):
            result.update(
                action="defer_other_ascend_hardware_decision",
                target_path=None,
                reason="910B3 first acceptance does not authorize deleting other Ascend variants",
            )
        else:
            replacements = {
                "worker/": "vllm/v1/worker/",
                "attention/": "vllm/v1/attention/backends/ascend/",
                "quantization/": "vllm/model_executor/layers/quantization/ascend/",
                "ops/": "vllm/model_executor/layers/ascend/",
                "spec_decode/": "vllm/v1/spec_decode/ascend/",
                "distributed/": "vllm/distributed/",
                "compilation/": "vllm/compilation/ascend/",
            }
            target = next(
                (
                    new + suffix[len(old) :]
                    for old, new in replacements.items()
                    if suffix.startswith(old)
                ),
                None,
            )
            result.update(
                action="merge_target_dependency_components",
                target_path=target,
                reason="Proposed ownership only; do not overwrite destination classes or include unrelated model paths",
                validation=["A03", "B03", "B10", "B11", "C13"],
            )
    elif repo == "LMCache-Ascend" and path.startswith("lmcache_ascend/"):
        suffix = path.removeprefix("lmcache_ascend/")
        if suffix == "__init__.py" or suffix.startswith("integration/patch/"):
            result.update(
                action="inline_injections_then_remove",
                target_path=None,
                validation=["A03", "C03", "C08"],
            )
        elif suffix.startswith(("mindspore/", "integration/sglang/")):
            result.update(action="non_target_integration_candidate", target_path=None)
        else:
            result.update(
                action="merge_with_base_preserving_contracts",
                target_path="lmcache/" + suffix,
                reason="Parent and Ascend subclass behavior must both survive; do not replace the parent wholesale",
                validation=["C02", "C03", "C06", "C07", "C08", "C09", "C13"],
            )
    elif path.startswith(("tests/", "examples/", "docs/")):
        result.update(
            action="retain_relevant_tests_docs_and_examples_after_review",
            reason="Keep target coverage and reject obsolete device/model entry points",
        )
    return result


def class_contracts(index: list[dict]) -> list[dict]:
    """List inherited/overridden direct methods; do not pretend to resolve full MRO."""
    classes = {}
    for file in index:
        for cls in file["classes"]:
            classes[(file["repository"], file["path"], cls["name"])] = cls
    pairs = [
        (
            ("vllm-ascend", "vllm_ascend/worker/model_runner_v1.py", "NPUModelRunner"),
            ("vllm", "vllm/v1/worker/gpu_model_runner.py", "GPUModelRunner"),
        ),
        (
            ("vllm-ascend", "vllm_ascend/worker/v2/model_runner.py", "NPUModelRunner"),
            ("vllm", "vllm/v1/worker/gpu/model_runner.py", "GPUModelRunner"),
        ),
        (
            (
                "LMCache-Ascend",
                "lmcache_ascend/v1/cache_engine.py",
                "AscendLMCacheEngine",
            ),
            ("LMCache", "lmcache/v1/cache_engine.py", "LMCacheEngine"),
        ),
        (
            (
                "LMCache-Ascend",
                "lmcache_ascend/integration/vllm/vllm_v1_adapter.py",
                "LMCacheAscendConnectorV1Impl",
            ),
            (
                "LMCache",
                "lmcache/integration/vllm/vllm_v1_adapter.py",
                "LMCacheConnectorV1Impl",
            ),
        ),
    ]
    records = []
    for child_id, parent_id in pairs:
        child, parent = classes.get(child_id), classes.get(parent_id)
        if child is None or parent is None:
            records.append(
                {
                    "child": child_id,
                    "parent": parent_id,
                    "status": "symbol_resolution_required",
                }
            )
            continue
        c_methods = {m["name"] for m in child["methods"]}
        p_methods = {m["name"] for m in parent["methods"]}
        records.append(
            {
                "child": child_id,
                "parent": parent_id,
                "bases": child["bases"],
                "overridden_direct_methods": sorted(c_methods & p_methods),
                "parent_methods_not_overridden_in_child": sorted(p_methods - c_methods),
                "child_only_methods": sorted(c_methods - p_methods),
                "super_calls": child["super_calls"],
                "status": "static_comparison_needs_mixin_MRO_and_runtime_review",
            }
        )
    return records


def main() -> None:
    """Generate source-to-target review maps and a submodule material checklist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.baseline / "source-manifest.json").read_text())
    index = json.loads((args.baseline / "python-dependencies.json").read_text())[
        "files"
    ]
    scope = json.loads(args.scope.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    mappings = [
        disposition(repo["repository"], file["path"])
        for repo in source["repos"]
        for file in repo["files"]
    ]
    write_json(
        args.output / "migration-map.json",
        {
            "scope_revision": scope["scope_revision"],
            "ready_for_bulk_operations": False,
            "files": mappings,
        },
    )
    write_json(
        args.output / "class-contracts.json", {"contracts": class_contracts(index)}
    )
    registry = json.loads((args.baseline / "model-registry.json").read_text())[
        "entries"
    ]
    projected = []
    for row in registry:
        architecture = row["architecture"]
        action = (
            "public_candidate_pending_actual_config"
            if architecture == "GlmMoeDsaForCausalLM"
            else "internal_glm52_mtp_only"
            if architecture == "DeepSeekMTPModel"
            else "remove_public_support_after_required_components_extracted"
        )
        projected.append(
            {
                "architecture": architecture,
                "module": row["module"],
                "class": row["class"],
                "line": row["line"],
                "current_scope_action": action,
            }
        )
    write_json(
        args.output / "model-scope-projection.json",
        {
            "scope_revision": scope["scope_revision"],
            "supersedes_baseline_registry_dispositions": True,
            "entries": projected,
        },
    )
    submodules = []
    for repo in source["repos"]:
        config = configparser.ConfigParser()
        config.read(args.workspace / repo["repository"] / ".gitmodules")
        for link in repo["gitlinks"]:
            section = next(
                (
                    s
                    for s in config.sections()
                    if config.get(s, "path", fallback="") == link["path"]
                ),
                None,
            )
            submodules.append(
                {
                    "repository": repo["repository"],
                    "path": link["path"],
                    "commit": link["git_object"],
                    "url": config.get(section, "url") if section else None,
                    "initialized": (
                        args.workspace / repo["repository"] / link["path"] / ".git"
                    ).exists(),
                    "payload_backed_up": False,
                    "required_action": "Obtain exact gitlink commit through allowed proxy/internal source, verify then archive payload before build",
                    "branch_head_is_not_acceptable_substitute": True,
                }
            )
    write_json(args.output / "submodule-materials.json", {"submodules": submodules})
    print(
        json.dumps(
            {
                "mapped_files": len(mappings),
                "registry_rows": len(projected),
                "submodules": len(submodules),
                "scope": scope["scope_revision"],
            }
        )
    )


if __name__ == "__main__":
    main()
