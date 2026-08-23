"""
Package the agent into `submission.tar.gz` for the PTCG AI Battle (cabt) comp.

The cabt submission is an agent bundle whose entrypoint `main.py` exposes
`agent(obs) -> list[int]`. Kaggle loads main.py with exec() (NO __file__), but
the loader appends the agent's dir to sys.path, so we can BUNDLE the `cg/` engine
folder next to main.py and `from cg.api import ...` resolves at eval time (the
search agent needs it). cg/ ships both libcg.so (Linux eval) and cg.dll (local
Windows), so sim.py loads the right one per OS. The deck is EMBEDDED in main.py.
We smoke-test the agent before packaging so a broken agent never ships.

Usage
-----
    python tools/build_submission.py
    # -> creates ./submission.tar.gz

Then (after accepting the competition rules on Kaggle):
    kaggle competitions submit -c pokemon-tcg-ai-battle \
        -f submission.tar.gz -m "heuristic v1"
"""

from __future__ import annotations

import os
import sys
import tarfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.join(_ROOT, "agent")
_OUT = os.path.join(_ROOT, "submission.tar.gz")

# Files at the archive root: the layered agent + its modules, plus cg/ engine.
# cg/ ships libcg.so (Linux eval) + cg.dll (local Windows). model.pth is included
# when present (the trained RL net); without it the agent runs as v3 search.
_PY_FILES = ["main.py", "search_agent.py", "rlnet.py"]
_OPTIONAL = ["model.pth", "model.pth.arch.json", "rl_config.py"]
_MAIN = (os.path.join(_AGENT_DIR, "main.py"), "main.py")
_CG_DIR = os.path.join(_AGENT_DIR, "cg")


def _smoke_test() -> int:
    """Import the agent and exercise it; fail loudly if broken."""
    sys.path.insert(0, _AGENT_DIR)
    import main  # noqa: E402

    if len(main.DECK) != 60:
        raise SystemExit(f"DECK must be exactly 60 cards, found {len(main.DECK)}")

    deck_sel = main.agent({"select": None})
    if len(deck_sel) != 60:
        raise SystemExit(f"deck-select returned {len(deck_sel)} cards, expected 60")

    move = main.agent({"current": {}, "select": {"option": [{"type": 13}, {"type": 14}], "maxCount": 1}})
    if not move or move[0] not in (0, 1):
        raise SystemExit(f"in-game decision returned invalid selection: {move}")

    print(f"  smoke: RL net active={main._RL is not None}")
    return len(main.DECK)


def _cg_members():
    """(src, arcname) pairs for every file under cg/, skipping caches."""
    out = []
    for root, dirs, files in os.walk(_CG_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".pyc"):
                continue
            src = os.path.join(root, f)
            rel = os.path.relpath(src, _AGENT_DIR)  # -> cg/...
            out.append((src, rel.replace(os.sep, "/")))
    return out


def main() -> None:
    if not os.path.isdir(_CG_DIR):
        raise SystemExit(f"Missing cg/ engine dir: {_CG_DIR}")
    for f in _PY_FILES:
        if not os.path.exists(os.path.join(_AGENT_DIR, f)):
            raise SystemExit(f"Missing required file: {f}")

    n = _smoke_test()
    members = [(os.path.join(_AGENT_DIR, f), f) for f in _PY_FILES]
    for f in _OPTIONAL:
        p = os.path.join(_AGENT_DIR, f)
        if os.path.exists(p):
            members.append((p, f))
    members += _cg_members()

    with tarfile.open(_OUT, "w:gz") as tar:
        for src, arcname in members:
            tar.add(src, arcname=arcname)

    size_kb = os.path.getsize(_OUT) / 1024
    names = [m[1] for m in members]
    print(f"Built {_OUT}  ({size_kb:.1f} KB)")
    print(f"  {len(names)} files: {names}  |  embedded deck: {n} cards  |  search+smoke OK")
    print("\nSubmit with (after accepting competition rules on Kaggle):")
    print('  kaggle competitions submit -c pokemon-tcg-ai-battle \\')
    print('      -f submission.tar.gz -m "heuristic v1"')


if __name__ == "__main__":
    main()
