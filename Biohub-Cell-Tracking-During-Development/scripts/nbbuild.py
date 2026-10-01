"""Tiny helper for assembling .ipynb files from (kind, source) cell lists.

Notebooks are generated rather than hand-edited so that ``src/biohub_ct.py``
stays the single source of truth: the Kaggle notebooks inline it verbatim via
:func:`library_cell` and therefore never drift from the local copy.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

def md(text: str) -> tuple[str, str]:
    return ("markdown", text.strip("\n"))

def code(text: str) -> tuple[str, str]:
    return ("code", text.strip("\n"))

def library_cell(target: str = "biohub_ct.py", source: str | None = None) -> tuple[str, str]:
    """A code cell that writes one of the ``src/`` modules into the kernel.

    ``source`` defaults to ``target``, so ``library_cell("biohub_unet.py")``
    inlines ``src/biohub_unet.py``. (An earlier version hard-coded a single
    source path and silently wrote biohub_ct.py's contents under whatever
    filename it was given, which broke the training notebook at import time.)
    """
    src_path = SRC / (source or target)
    if not src_path.is_file():
        raise FileNotFoundError(f"library_cell: no such module {src_path}")
    return code(f"%%writefile {target}\n" + src_path.read_text().rstrip("\n"))

def _verify_inlined_modules(cells: list[tuple[str, str]]) -> None:
    """Assert every ``%%writefile`` cell carries the matching ``src/`` module.

    Cheap insurance: a mismatch here surfaces only as an AttributeError minutes
    into a Kaggle GPU run, which is an expensive way to find a typo.
    """
    for kind, source in cells:
        if kind != "code" or not source.startswith("%%writefile "):
            continue
        first, _, body = source.partition("\n")
        target = first.split(None, 1)[1].strip()
        expected = SRC / target
        if not expected.is_file():
            continue
        if body.rstrip("\n") != expected.read_text().rstrip("\n"):
            raise AssertionError(
                f"inlined cell for {target} does not match {expected}"
            )

def build(path: Path | str, cells: list[tuple[str, str]], *,
          kernelspec: str = "python3", accelerator_note: str | None = None) -> Path:
    path = Path(path)
    _verify_inlined_modules(cells)
    nb_cells = []
    for kind, source in cells:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": kernelspec,
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1))
    return path
