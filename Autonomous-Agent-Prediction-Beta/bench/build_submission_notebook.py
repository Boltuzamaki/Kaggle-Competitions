#!/usr/bin/env python3
"""Turn an agent package directory into a self-contained Kaggle notebook.

Every file in the package becomes a `%%writefile` cell, so the notebook is the
readable source of truth for the config and reproduces `submission.zip` byte for
byte on a fresh run. Nothing is downloaded and no dataset is attached.

Usage:
  python bench/build_submission_notebook.py --package bench/v8_pkg \
      --out kaggle_notebook_submit --slug freeroll-hardened-portfolio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ordered so a reader meets the orchestrator first, then prompts, then the ML core.
ORDER = ("agent.yaml", "agents/", "prompts/", "configs/", "skills/")


def sort_key(rel: str):
    for i, prefix in enumerate(ORDER):
        if rel == prefix or rel.startswith(prefix):
            return (i, rel)
    return (len(ORDER), rel)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


def build(package: Path, title: str, notes: str) -> dict:
    files = sorted(
        (p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts
         and p.suffix != ".pyc"),
        key=lambda p: sort_key(str(p.relative_to(package)).replace("\\", "/")),
    )

    cells = [md(notes)]
    dirs = sorted({str(p.relative_to(package).parent).replace("\\", "/")
                   for p in files} - {"."})
    mk = "import os\n\nfor d in [\n" + "".join(
        f'    "submission/{d}",\n' for d in dirs
    ) + "]:\n    os.makedirs(d, exist_ok=True)\nprint('submission/ tree created')\n"
    cells.append(code(mk))

    for path in files:
        rel = str(path.relative_to(package)).replace("\\", "/")
        body = path.read_text(encoding="utf-8")
        if not body.endswith("\n"):
            body += "\n"
        cells.append(code(f"%%writefile submission/{rel}\n{body}"))

    cells.append(md(
        "## Validate, then package\n\n"
        "The gate below is deliberately blocking: every `.py` must parse, `agent.yaml` must sit\n"
        "at the archive root, every `!include` and `config_path` must resolve, and no bytecode\n"
        "may be present. A config that fails any of these starts and then dies inside the\n"
        "container, which costs the whole session.\n"
    ))
    cells.append(code(VALIDATE))
    cells.append(code(PACKAGE))
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


VALIDATE = '''import ast, os
from pathlib import Path

pkg = Path("submission")
problems = []

if not (pkg / "agent.yaml").is_file():
    problems.append("agent.yaml missing at archive root")

for py in pkg.rglob("*.py"):
    try:
        ast.parse(py.read_text())
    except SyntaxError as exc:
        problems.append(f"{py}: {exc}")

if any(pkg.rglob("__pycache__")) or any(pkg.rglob("*.pyc")):
    problems.append("bytecode artifacts present")

for y in pkg.rglob("*.yaml"):
    for line in y.read_text().splitlines():
        if "!include" in line:
            target = (y.parent / line.split("!include")[1].strip()).resolve()
            if not target.is_file():
                problems.append(f"{y}: unresolved !include -> {target}")
        if "config_path:" in line:
            target = (pkg / line.split("config_path:")[1].strip()).resolve()
            if not target.is_file():
                problems.append(f"{y}: unresolved config_path -> {target}")

# The evaluator rejects transport options outside its pinned config schema.
for y in pkg.rglob("*.yaml"):
    text = y.read_text()
    for token in ("http_options:", "retry_options:"):
        if token in text:
            problems.append(f"{y}: evaluator-forbidden field {token}")

print("files:", sum(1 for _ in pkg.rglob("*") if _.is_file()))
if problems:
    for p in problems:
        print("  FAIL", p)
    raise SystemExit("validation failed")
print("validation passed")
'''

PACKAGE = '''import hashlib, zipfile
from pathlib import Path

pkg = Path("submission")
files = sorted(p for p in pkg.rglob("*") if p.is_file())

# Fixed timestamps + sorted entries => the same inputs always give the same sha256,
# which is what makes "resubmit the previous archive" a meaningful rollback.
with zipfile.ZipFile("submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for path in files:
        info = zipfile.ZipInfo(str(path.relative_to(pkg)), date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, path.read_bytes())

blob = Path("submission.zip").read_bytes()
print("submission.zip", len(blob), "bytes")
print("sha256", hashlib.sha256(blob).hexdigest())
for path in files:
    print("  ", path.relative_to(pkg))
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", default="Autonomous Agent — Hardened Freeroll Portfolio")
    ap.add_argument("--notes", default="")
    ap.add_argument("--user", default="boltuzamaki")
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    package = Path(args.package).resolve()
    out_dir = (ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    notes = args.notes
    if notes and Path(notes).is_file():
        notes = Path(notes).read_text()

    nb_name = f"{args.slug}.ipynb"
    (out_dir / nb_name).write_text(
        json.dumps(build(package, args.title, notes or f"# {args.title}\n"), indent=1),
        encoding="utf-8",
    )
    (out_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{args.slug}",
        "title": args.title,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(args.private),
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["autonomous-agent-prediction-beta"],
        "model_sources": [],
    }, indent=2), encoding="utf-8")

    print(f"wrote {out_dir / nb_name}")
    print(f"wrote {out_dir / 'kernel-metadata.json'}")
    print(f"push with:  kaggle kernels push -p {out_dir}")


if __name__ == "__main__":
    main()
