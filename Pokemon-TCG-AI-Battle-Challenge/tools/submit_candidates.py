"""
Submission pipeline: build + submit a set of candidate agents to Kaggle.

Each candidate = (name, kind, search). For RL candidates we write agent/rl_config.py
(sets inference MCTS depth) and ensure agent/model.pth is present; for the v3
candidate we remove model.pth so the bundle runs pure search. build_submission.py
smoke-tests + Kaggle-validates each bundle before we submit.

Usage:
    python tools/submit_candidates.py            # build + submit the default set
    python tools/submit_candidates.py --dry-run  # build + validate only, no submit
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AGENT = os.path.join(_ROOT, "agent")
_MODELS = os.path.join(_ROOT, "models")
_PY = sys.executable
_COMP = "pokemon-tcg-ai-battle"

# (label, kind, search_count). kind: "rl" uses the net; "v3" is pure search.
CANDIDATES = [
    ("rl-net-s48", "rl", 48),
    ("rl-net-s96", "rl", 96),
]


def _set_rl_config(search: int, budget: float = 8.0) -> None:
    with open(os.path.join(_AGENT, "rl_config.py"), "w") as f:
        f.write(f"# Auto-written by submit_candidates.py\nSEARCH = {search}\nBUDGET = {budget}\n")


def _ensure_model() -> None:
    dst = os.path.join(_AGENT, "model.pth")
    if not os.path.exists(dst):
        src = os.path.join(_MODELS, "model_rl_iter49.pth")
        if not os.path.exists(src):
            raise SystemExit(f"trained net not found: {src}")
        shutil.copy(src, dst)


def _clear_rl() -> None:
    for f in ("model.pth", "rl_config.py"):
        p = os.path.join(_AGENT, f)
        if os.path.exists(p):
            os.remove(p)


def _build() -> None:
    r = subprocess.run([_PY, os.path.join(_ROOT, "tools", "build_submission.py")],
                       capture_output=True, text=True)
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
    if r.returncode != 0:
        raise SystemExit(f"build failed:\n{r.stdout}\n{r.stderr}")


def _submit(label: str) -> None:
    env = dict(os.environ, KAGGLE_CONFIG_DIR="C:/Users/chand/.kaggle")
    r = subprocess.run(
        [_PY, "-m", "kaggle", "competitions", "submit", "-c", _COMP,
         "-f", os.path.join(_ROOT, "submission.tar.gz"), "-m", label],
        capture_output=True, text=True, env=env)
    ok = "Successfully submitted" in (r.stdout + r.stderr)
    print(f"  submit [{label}]: {'OK' if ok else 'FAILED'}")
    if not ok:
        print(r.stdout, r.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for label, kind, search in CANDIDATES:
        print(f"\n=== candidate: {label} ({kind}, search={search}) ===")
        if kind == "rl":
            _ensure_model()
            _set_rl_config(search)
        else:
            _clear_rl()
        _build()
        if args.dry_run:
            print("  (dry-run: not submitting)")
        else:
            _submit(label)

    print("\nDone. Check status: python -m kaggle competitions submissions " + _COMP)


if __name__ == "__main__":
    main()
