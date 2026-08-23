"""Build the Kaggle CPU kernel that tunes our Grimmsnarl policy weights.

Hill-climbing with random restarts over agent/grimmsnarl_policy.WEIGHTS, scored
by win-rate against a fixed opponent panel drawn from the live meta. Writes the
best genome to /kaggle/working/grimm_w.json for the local build to pick up.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_grimmsnarl_tuning"

SETUP = '''\
import glob, json, os, shutil, sys
from pathlib import Path

# The dataset may be served either already-extracted or as project.zip.
matches = glob.glob("/kaggle/input/**/tools/arena.py", recursive=True)
if not matches:
    archives = glob.glob("/kaggle/input/**/project.zip", recursive=True)
    if not archives:
        raise RuntimeError("ptcg-grimmsnarl-assets is not attached")
    root = Path("/kaggle/working/ptcg_asset")
    root.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(archives[0], root)
    matches = glob.glob(str(root / "**/tools/arena.py"), recursive=True)
    if not matches:
        raise RuntimeError("project.zip has no tools/arena.py")
project = Path(matches[0]).parents[1]
work = Path("/kaggle/working/ptcg")
shutil.copytree(project, work, dirs_exist_ok=True)
for p in glob.glob("/kaggle/input/**/cg-lib", recursive=True):
    sys.path.insert(0, p)
for p in (work, work / "agent", work / "tools", work / "references" / "top_rankers"):
    sys.path.insert(0, str(p))
os.chdir(work)
print("project", work)
'''

RUN = '''\
import json, os, random, time
from concurrent.futures import ProcessPoolExecutor

from kaggle_environments import make
import arena, grimmsnarl_policy, domain_policy, meta_decks

BASE = dict(grimmsnarl_policy.WEIGHTS)
# Opponent panel: the mirror dominates the real field, so it carries the most
# weight, with two off-archetype checks to stop the genome overfitting to it.
PANEL = [("grimmsnarl", meta_decks.GRIMMSNARL, 3),
         ("dragapult", meta_decks.DRAGAPULT, 1),
         ("alakazam", meta_decks.ALAKAZAM, 1)]
GAMES_PER = int(os.environ.get("GAMES_PER", "8"))
BUDGET_S = float(os.environ.get("BUDGET_S", "28000"))


def evaluate(genome, games_per=GAMES_PER, seed=0):
    """Win-rate of `genome` against the panel, seats balanced."""
    grimmsnarl_policy.W.clear(); grimmsnarl_policy.W.update(genome)
    ours = lambda obs: grimmsnarl_policy.grimmsnarl_agent(obs, meta_decks.GRIMMSNARL)
    wins = total = 0
    for name, deck, weight in PANEL:
        opp = (lambda d: (lambda obs: domain_policy.domain_agent(obs, d)))(deck)
        for i in range(games_per * weight):
            a, b = (ours, opp) if i % 2 == 0 else (opp, ours)
            out = arena.play_game(make, a, b)
            total += 1
            if out.winner is not None:
                me = 0 if i % 2 == 0 else 1
                wins += int(out.winner == me)
    return wins / max(total, 1), total


def mutate(genome, rng, rate=0.30, scale=0.35):
    child = dict(genome)
    for k in child:
        if rng.random() < rate:
            child[k] = max(0.0, child[k] * (1.0 + rng.gauss(0, scale)))
    return child


def climb(seed):
    rng = random.Random(seed)
    cur = dict(BASE) if seed == 0 else mutate(BASE, rng, rate=0.5, scale=0.5)
    cur_score, _ = evaluate(cur, seed=seed)
    history = [("init", cur_score)]
    t0 = time.time()
    while time.time() - t0 < BUDGET_S / 4:
        cand = mutate(cur, rng)
        score, _ = evaluate(cand, seed=seed)
        if score > cur_score:
            cur, cur_score = cand, score
            history.append((f"t{int(time.time()-t0)}", score))
    return {"seed": seed, "score": cur_score, "genome": cur, "history": history}


if __name__ == "__main__":
    t0 = time.time()
    base_score, n = evaluate(BASE)
    print(f"baseline genome: {base_score:.3f} over {n} games", flush=True)

    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(climb, [0, 1, 2, 3]))

    results.sort(key=lambda r: -r["score"])
    best = results[0]
    print(json.dumps({"baseline": base_score,
                      "best": best["score"],
                      "per_seed": [(r["seed"], r["score"]) for r in results]}, indent=1))

    # Confirm the winner on a fresh, larger sample before promoting it.
    confirm, cn = evaluate(best["genome"], games_per=GAMES_PER * 4, seed=99)
    base_confirm, _ = evaluate(BASE, games_per=GAMES_PER * 4, seed=99)
    print(f"confirmation over {cn} games: best={confirm:.3f} baseline={base_confirm:.3f}")

    out = {"baseline_score": base_score, "search_score": best["score"],
           "confirm_best": confirm, "confirm_baseline": base_confirm,
           "promote": bool(confirm > base_confirm),
           "genome": best["genome"], "elapsed_s": time.time() - t0}
    Path("/kaggle/working/grimm_tuning_result.json").write_text(json.dumps(out, indent=1))
    Path("/kaggle/working/grimm_w.json").write_text(json.dumps(best["genome"], indent=1))
    print("wrote grimm_w.json; promote =", out["promote"])
'''


def notebook(cells):
    return {
        "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": s.splitlines(keepends=True)} for s in cells],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "kernel-metadata.json").write_text(json.dumps({
        "id": "boltuzamaki/ptcg-cpu-grimmsnarl-tuning",
        "title": "PTCG CPU Grimmsnarl tuning",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["research"],
        "dataset_sources": ["boltuzamaki/ptcg-grimmsnarl-assets", "kiyotah/cg-lib"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }, indent=1), encoding="utf-8")
    (TARGET / "notebook.ipynb").write_text(
        json.dumps(notebook([SETUP, RUN]), indent=1), encoding="utf-8")
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
