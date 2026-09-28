"""Source-preserving helpers for the reviewed P2 symbol migrations."""

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "p1-repos/vllm"


def read(path):
    return (ROOT / path).read_text()


def write(path, source):
    ast.parse(source)
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)


def find(source, name, owner=None):
    tree = ast.parse(source)
    nodes = tree.body
    if owner:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    return next(n for n in nodes if getattr(n, "name", None) == name)


def start(node):
    return min([node.lineno, *[n.lineno for n in getattr(node, "decorator_list", [])]]) - 1


def segment(source, node):
    return textwrap.dedent("".join(source.splitlines(True)[start(node):node.end_lineno]))


def definition(path, name, owner=None):
    source = read(path)
    return segment(source, find(source, name, owner))


def replace(path, name, replacement, owner=None):
    source = read(path)
    node = find(source, name, owner)
    lines = source.splitlines(True)
    replacement = textwrap.indent(textwrap.dedent(replacement).rstrip() + "\n", " " * node.col_offset)
    lines[start(node):node.end_lineno] = [replacement]
    write(path, "".join(lines))


def append_class(path, owner, text):
    source = read(path)
    node = find(source, owner)
    lines = source.splitlines(True)
    lines[node.end_lineno:node.end_lineno] = ["\n" + textwrap.indent(textwrap.dedent(text).rstrip() + "\n", "    ")]
    write(path, "".join(lines))


def prepend_body(path, name, text, owner=None):
    source = read(path)
    node = find(source, name, owner)
    index = node.body[0].lineno - 1
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        index = node.body[0].end_lineno
    lines = source.splitlines(True)
    lines[index:index] = [textwrap.indent(textwrap.dedent(text).rstrip() + "\n", " " * (node.col_offset + 4))]
    write(path, "".join(lines))


def add_imports(path, text):
    source = read(path)
    tree = ast.parse(source)
    index = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            index = node.end_lineno
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            index = node.end_lineno
        else:
            if index == 0:
                index = node.lineno - 1
            break
    lines = source.splitlines(True)
    lines[index:index] = ["\n" + textwrap.dedent(text).rstrip() + "\n\n"]
    write(path, "".join(lines))


def replace_text(path, old, new, count=1):
    source = read(path)
    assert source.count(old) == count, (path, old, source.count(old), count)
    write(path, source.replace(old, new))
