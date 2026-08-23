"""Load the two self-contained public agents embedded in the meta notebook.

The notebook stores each submission as JSON payload fields (``main_py`` and
``deck_csv``). This adapter parses those assignments without executing the
notebook's build cells, then loads each agent in an isolated module namespace.
It is for local benchmarking; our candidate source remains separate.
"""

from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import types
import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_DIR = str(_ROOT / "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)
_NOTEBOOK = (
    _ROOT
    / "references"
    / "top_rankers"
    / "romanrozen_v10"
    / "strong-start-baseline-agent-v10-lb-950.ipynb"
)


def _payloads():
    notebook = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if "AGENT_PAYLOADS" not in names:
                continue
            call = node.value
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "loads"
                and call.args
            ):
                continue
            return json.loads(ast.literal_eval(call.args[0]))
    raise RuntimeError(f"AGENT_PAYLOADS not found in {_NOTEBOOK}")


_TEMP_DIRS = []


def _load(key, payload):
    deck = [int(line) for line in payload["deck_csv"].splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"public payload {key} deck has {len(deck)} cards")

    temp = tempfile.TemporaryDirectory(prefix=f"ptcg_public_{key.lower()}_")
    _TEMP_DIRS.append(temp)  # keep files alive for the module's lifetime
    directory = Path(temp.name)
    (directory / "deck.csv").write_text(payload["deck_csv"], encoding="utf-8")
    main_path = directory / "main.py"
    main_path.write_text(payload["main_py"], encoding="utf-8")

    module = types.ModuleType(f"ptcg_public_{key.lower()}")
    module.__file__ = str(main_path)
    old_cwd = os.getcwd()
    try:
        os.chdir(directory)
        exec(compile(payload["main_py"], str(main_path), "exec"), module.__dict__)
    finally:
        os.chdir(old_cwd)
    if not callable(getattr(module, "agent", None)):
        raise TypeError(f"public payload {key} has no callable agent")
    return module.agent, deck, payload["label"]


_PAYLOADS = _payloads()
ROMAN_A_AGENT, ROMAN_A_DECK, ROMAN_A_LABEL = _load("A", _PAYLOADS["A"])
ROMAN_B_AGENT, ROMAN_B_DECK, ROMAN_B_LABEL = _load("B", _PAYLOADS["B"])


def load_source_agent(name, directory):
    """Load a source-only agent directory without changing its bytes."""
    directory = Path(directory).resolve()
    deck = [int(x) for x in (directory / "deck.csv").read_text().splitlines() if x.strip()]
    if len(deck) != 60:
        raise ValueError(f"{name} deck has {len(deck)} cards")
    path = directory / "main.py"
    spec = importlib.util.spec_from_file_location(f"ptcg_source_{name}", path)
    module = importlib.util.module_from_spec(spec)
    old_cwd = os.getcwd()
    try:
        os.chdir(directory)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    fn = getattr(module, "agent", None)
    if not callable(fn):
        # Historical packages sometimes leave a differently named final entrypoint.
        callables = [v for k, v in module.__dict__.items()
                     if callable(v) and not k.startswith("_")]
        if not callables:
            raise TypeError(f"{name} has no callable entrypoint")
        fn = callables[-1]
    return fn, deck


_ZOO = _ROOT / "references" / "public_sim_repo" / "agent_zoo" / "sources"
NAOTO_1027_AGENT, NAOTO_1027_DECK = load_source_agent(
    "naoto_1027", _ZOO / "635999b1a0f7-e07b796d823c")
ARCH_1030_AGENT, ARCH_1030_DECK = load_source_agent(
    "arch_1030", _ZOO / "a4c53101be30-fbe6ab599922")
GARCHOMP_V28_AGENT, GARCHOMP_V28_DECK = load_source_agent(
    "garchomp_v28", _ROOT / "references" / "competitor_refresh" /
    "garchomp_v28" / "extracted")


if __name__ == "__main__":
    print(f"A: {ROMAN_A_LABEL} | deck={len(ROMAN_A_DECK)} | callable={callable(ROMAN_A_AGENT)}")
    print(f"B: {ROMAN_B_LABEL} | deck={len(ROMAN_B_DECK)} | callable={callable(ROMAN_B_AGENT)}")
