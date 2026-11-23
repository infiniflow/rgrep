#!/usr/bin/env python3
"""Build a retrieval eval set from a source tree, using the code's own prose.

Two query sets, both with an exact (path, name) ground truth, so the evaluator
does not have to guess which result "should" have won:

  nl      the first sentence of a function/class docstring or Go doc comment
  symbol  the identifier itself

Docstrings are the closest thing a repository has to real developer questions,
and identifiers are what developers actually type. Neither set is generated
from the index, so nothing leaks between index building and evaluation.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re


GO_FUNC = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*[\(\[]")


def first_sentence(text: str) -> str:
    """One clean line: the first sentence, stripped of markdown and quotes."""
    text = " ".join(text.split())
    text = text.strip("\"'`")
    for stop in (". ", "? ", "! ", ".\n"):
        if stop in text:
            text = text[: text.index(stop)]
            break
    return text.strip()


def python_queries(path: str, rel: str) -> list[dict]:
    try:
        with open(path, "rb") as handle:
            source = handle.read()
        tree = ast.parse(source)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return []

    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        sentence = first_sentence(doc)
        if len(sentence) < 25:
            continue
        out.append(
            {
                "path": rel,
                "name": node.name,
                "line": node.lineno,
                "nl": sentence,
                "symbol": node.name,
            }
        )
    return out


def go_queries(path: str, rel: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
    except (UnicodeDecodeError, OSError):
        return []

    out = []
    for index, line in enumerate(lines):
        match = GO_FUNC.match(line)
        if not match:
            continue
        name = match.group(1)
        # The doc comment is the contiguous // block directly above.
        block = []
        cursor = index - 1
        while cursor >= 0 and lines[cursor].startswith("//"):
            block.append(lines[cursor].lstrip("/ ").strip())
            cursor -= 1
        if not block:
            continue
        comment = " ".join(reversed(block))
        # Skip directives like //go:build and //nolint.
        if re.match(r"^[a-z0-9]+:", comment):
            continue
        sentence = first_sentence(comment)
        if len(sentence) < 25:
            continue
        out.append(
            {
                "path": rel,
                "name": name,
                "line": index + 1,
                "nl": sentence,
                "symbol": name,
            }
        )
    return out


def ts_queries(path: str, rel: str) -> list[dict]:
    """JSDoc immediately above an exported function, class or const arrow."""
    try:
        source = io.open(path, "r", encoding="utf-8").read()
    except (UnicodeDecodeError, OSError):
        return []

    out = []
    pattern = re.compile(
        r"/\*\*(?P<doc>.*?)\*/\s*(?:export\s+)?(?:async\s+)?(?:function|class)\s+(?P<name>[A-Za-z_$][\w$]*)",
        re.S,
    )
    for match in pattern.finditer(source):
        doc = re.sub(r"^\s*\*\s?", "", match.group("doc"), flags=re.M)
        sentence = first_sentence(doc)
        if len(sentence) < 25:
            continue
        line = source[: match.start()].count("\n") + 1
        out.append(
            {
                "path": rel,
                "name": match.group("name"),
                "line": line,
                "nl": sentence,
                "symbol": match.group("name"),
            }
        )
    return out


HANDLERS = {".py": python_queries, ".go": go_queries, ".ts": ts_queries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=200, help="queries per set")
    parser.add_argument("--min-name", type=int, default=6)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    rows: list[dict] = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [
            d for d in dirs if d not in {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build"}
        ]
        for name in files:
            handler = HANDLERS.get(os.path.splitext(name)[1])
            if handler is None:
                continue
            full = os.path.join(base, name)
            if os.path.getsize(full) > 512 * 1024:
                continue
            rel = os.path.relpath(full, root)
            rows.extend(handler(full, rel))

    # A stable, evenly spread sample: walking order is directory order, so an
    # every-k-th pick covers the tree without random seeds.
    def sample(items: list[dict], key: str) -> list[dict]:
        if len(items) <= args.limit:
            return items
        step = len(items) / args.limit
        return [items[int(i * step)] for i in range(args.limit)]

    nl = sample([r for r in rows if r["nl"]], args.limit)
    symbol = sample([r for r in rows if len(r["symbol"]) >= args.min_name], args.limit)

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in nl:
            handle.write(json.dumps({"set": "nl", "query": row["nl"], "path": row["path"], "name": row["name"]}) + "\n")
        for row in symbol:
            handle.write(
                json.dumps({"set": "symbol", "query": row["symbol"], "path": row["path"], "name": row["name"]}) + "\n"
            )
    print(f"{len(nl)} nl + {len(symbol)} symbol queries -> {args.out}")


if __name__ == "__main__":
    main()
