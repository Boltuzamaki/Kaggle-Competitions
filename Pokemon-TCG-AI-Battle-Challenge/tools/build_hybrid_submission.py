"""
Package the HYBRID CANDIDATE (agent/main_hybrid.py: our domain policy driving
determinized search, piloting the Mega Lucario ex deck) into
`submission_hybrid.tar.gz`. Kept SEPARATE from build_submission.py (which always
packages the safe, proven default agent/main.py) so building/testing this
candidate can never accidentally touch the safety submission.

Only submit this after it has beaten v3 with statistical significance in
tools/arena.py -- see PLAN.md Part I, rule 8: "Never replace the safety
submission until a new candidate is proven locally and on the leaderboard."

Usage
-----
    python tools/build_hybrid_submission.py
    # -> creates ./submission_hybrid.tar.gz

Then (only once proven):
    kaggle competitions submit -c pokemon-tcg-ai-battle \
        -f submission_hybrid.tar.gz -m "hybrid: domain policy + search, Mega Lucario ex"
"""

from __future__ import annotations

import os
import sys
import tarfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT_DIR = os.path.join(_ROOT, "agent")
_OUT = os.path.join(_ROOT, "submission_hybrid.tar.gz")

# main_hybrid.py is renamed to main.py at the archive root (the entrypoint
# Kaggle looks for). hybrid_agent.py + domain_policy.py are its dependencies.
_ENTRY = os.path.join(_AGENT_DIR, "main_hybrid.py")
_DEPS = ["hybrid_agent.py", "domain_policy.py"]
_CG_DIR = os.path.join(_AGENT_DIR, "cg")


def _smoke_test() -> int:
    """Import main_hybrid as `main` and exercise it; fail loudly if broken."""
    sys.path.insert(0, _AGENT_DIR)
    import importlib.util
    spec = importlib.util.spec_from_file_location("main", _ENTRY)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)

    if len(main.DECK) != 60:
        raise SystemExit(f"DECK must be exactly 60 cards, found {len(main.DECK)}")

    deck_sel = main.agent({"select": None})
    if len(deck_sel) != 60:
        raise SystemExit(f"deck-select returned {len(deck_sel)} cards, expected 60")

    move = main.agent({"current": {}, "select": {"option": [{"type": 13}, {"type": 14}], "maxCount": 1}})
    if not move or move[0] not in (0, 1):
        raise SystemExit(f"in-game decision returned invalid selection: {move}")

    return len(main.DECK)


def _cg_members():
    out = []
    for root, dirs, files in os.walk(_CG_DIR):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".pyc"):
                continue
            src = os.path.join(root, f)
            rel = os.path.relpath(src, _AGENT_DIR)
            out.append((src, rel.replace(os.sep, "/")))
    return out


def main() -> None:
    if not os.path.isdir(_CG_DIR):
        raise SystemExit(f"Missing cg/ engine dir: {_CG_DIR}")
    if not os.path.exists(_ENTRY):
        raise SystemExit(f"Missing entry file: {_ENTRY}")
    for f in _DEPS:
        if not os.path.exists(os.path.join(_AGENT_DIR, f)):
            raise SystemExit(f"Missing required dependency: {f}")

    n = _smoke_test()
    members = [(_ENTRY, "main.py")] + [
        (os.path.join(_AGENT_DIR, f), f) for f in _DEPS
    ] + _cg_members()

    with tarfile.open(_OUT, "w:gz") as tar:
        for src, arcname in members:
            tar.add(src, arcname=arcname)

    size_kb = os.path.getsize(_OUT) / 1024
    names = [m[1] for m in members]
    print(f"Built {_OUT}  ({size_kb:.1f} KB)")
    print(f"  {len(names)} files: {names}  |  embedded deck: {n} cards  |  smoke OK")
    print("\nOnly submit once proven vs v3 in tools/arena.py. Then:")
    print('  kaggle competitions submit -c pokemon-tcg-ai-battle \\')
    print('      -f submission_hybrid.tar.gz -m "hybrid: domain policy + search, Mega Lucario ex"')


if __name__ == "__main__":
    main()
