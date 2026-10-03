#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static intra-package import/export closure; no torch or package import.

Dynamic __getattr__/star exports are reported as unknown, not assumed verified.
This does not prove runtime/ABI correctness or detect dynamically built imports.
"""

import argparse
import ast
import json
from pathlib import Path

GENERATED = {"vllm._build_info", "vllm._version", "vllm._ascend_C",
             "lmcache._build_info", "lmcache._version", "lmcache.c_ops", "lmcache.lmcache_redis"}


def module_nodes(tree):
    """Traverse module control flow, never treat function locals as exports."""
    for node in ast.iter_child_nodes(tree):
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            yield from module_nodes(node)


def check(repo: Path, package: str) -> dict:
    files = list((repo / package).rglob("*.py"))
    trees, exports, dynamic, namespaces = {}, {}, set(), set()
    for file in files:
        rel = file.relative_to(repo)
        name = str(rel).removesuffix(".py").removesuffix("/__init__").replace("/", ".")
        tree = ast.parse(file.read_text(), filename=str(rel))
        trees[name] = (file, tree)
        names = set()
        for node in module_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add(alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name))
        exports[name] = names
        if any(isinstance(n, ast.FunctionDef) and n.name == "__getattr__" for n in tree.body) or "*" in names:
            dynamic.add(name)
        parts = name.split(".")
        namespaces.update(".".join(parts[:i]) for i in range(1, len(parts)))
    errors, unknown = [], []
    for name, (file, tree) in trees.items():
        parent = name if file.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = node.module or ""
            if node.level:
                parts = parent.split(".")
                target = ".".join(parts[:len(parts) - node.level + 1] + ([target] if target else []))
            if not (target == package or target.startswith(package + ".")):
                continue
            if target not in trees:
                # Native extensions and generated modules require intranet checks.
                if target not in namespaces:
                    item = {"file": str(file.relative_to(repo)), "line": node.lineno, "module": target}
                    (unknown if target in GENERATED else errors).append(item)
                continue
            for alias in node.names:
                member = target + "." + alias.name
                if alias.name == "*" or member in trees or member in namespaces or alias.name in exports[target]:
                    continue
                item = {"file": str(file.relative_to(repo)), "line": node.lineno, "import": member}
                (unknown if target in dynamic or member in GENERATED else errors).append(item)
    return {"scope": "static_named_exports_not_runtime", "errors": errors, "dynamic_or_native": unknown, "passed": not errors}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("package")
    args = parser.parse_args()
    result = check(args.repo, args.package)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
