#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only AST-targeted source transformation helpers; emit apply_patch text."""

import ast
import difflib
from pathlib import Path
import textwrap


def replace_def(source: str, name: str, replacement: str) -> str:
    """Replace all declarations (including overloads) with one implementation."""
    lines = source.splitlines(True)
    nodes = [n for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name]
    if not nodes:
        raise ValueError(f"Missing declaration: {name}")
    for index, node in reversed(list(enumerate(nodes))):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        body = textwrap.indent(textwrap.dedent(replacement).strip() + "\n", " " * node.col_offset) if index == 0 and replacement.strip() else ""
        lines[start:node.end_lineno] = [body]
    result = "".join(lines)
    ast.parse(result)
    return result


def emit(path: Path, old: str, new: str) -> None:
    """Print a text patch, never write the target file."""
    if old == new:
        return
    if path.suffix == ".py":
        ast.parse(new)
    print(f"*** Update File: {path}")
    for line in list(difflib.unified_diff(old.splitlines(True), new.splitlines(True)))[2:]:
        print("@@" if line.startswith("@@") else line.rstrip("\n"))
