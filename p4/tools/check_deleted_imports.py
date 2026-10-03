#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Find retained Python imports of modules deleted since the frozen P3 pair.

This is a static deletion-closure check, not a substitute for intranet imports,
native linking, wheel/sdist contents, or NPU inference. String references are
reported separately because some are historical documentation or messages.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess


def module_name(path: str) -> str:
    return path.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def check(repo: Path, package: str) -> dict:
    """Compare Python import targets against the frozen and remaining modules."""
    old_paths = subprocess.check_output(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "p3-frozen-20261003", "--", package], text=True).splitlines()
    old = {module_name(p) for p in old_paths if p.endswith(".py")}
    files = list((repo / package).rglob("*.py"))
    current = {module_name(str(p.relative_to(repo))) for p in files}
    deleted = old - current
    errors, strings = [], []
    for path in files:
        rel = str(path.relative_to(repo))
        this_module = module_name(rel)
        parent = this_module if path.name == "__init__.py" else this_module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = node.module or ""
                if node.level:
                    parts = parent.split(".")
                    prefix = ".".join(parts[:len(parts) - node.level + 1] + ([prefix] if prefix else []))
                targets = [prefix] + [prefix + "." + alias.name for alias in node.names]
            for target in targets:
                if target in deleted:
                    errors.append({"file": rel, "line": node.lineno, "target": target})
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value.replace(":", ".")
                if "\n" not in text and text.startswith(package + "."):
                    while text.startswith(package + "."):
                        if text in deleted:
                            strings.append({"file": rel, "line": node.lineno, "target": text})
                            break
                        text = text.rpartition(".")[0]
    return {"scope": "static_deleted_module_imports_not_runtime", "deleted_modules": len(deleted),
            "errors": sorted(errors, key=lambda e: (e['file'], e['line'])),
            "string_references": sorted(strings, key=lambda e: (e['file'], e['line'])),
            "passed": not errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("package")
    args = parser.parse_args()
    result = check(args.repo, args.package)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
