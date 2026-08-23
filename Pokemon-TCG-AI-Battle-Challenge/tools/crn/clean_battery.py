"""Overnight A/B battery on a verified-DETERMINISTIC opponent panel.

Why this exists: `td-td_08` and `hybs-other` -- two of our own search-based
competitors -- return different results for the same seed on repeated runs. They
appeared in most of today's panels, so a large share of the "discordant" seeds in
those tests was the OPPONENT disagreeing with itself, not the change under test.
That inflates discordance, destroys the variance reduction CRN is supposed to
buy, and is the most likely reason tech_basics read 47/31 on one panel and 52/60
on another.

Verified deterministic (same seed, repeated runs, identical outcomes):
    public-archaludon, meta-grimmsnarl, public-alakazam

The `control` arm is stock-vs-stock. On a clean panel it MUST come back 0/0
discordant. If it does not, the harness is still leaking randomness and every
other number in this run is void -- check it first.

Arms cover both of tonight's ideas plus the candidates that were rejected under
noisy conditions and deserve a fair re-test:

    control      stock              sanity check, must be 0/0
    depth2/4     multi-turn lookahead past the fork's 2-turn horizon
    tech_basics  deck change that led at p=0.089 then reversed
    templates    archetype belief model, previously p=0.52
"""
from __future__ import annotations

import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build, mcnemar  # noqa: E402
import deep_search  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
STOCK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]

NIGHT_STRETCHER, FEZANDIPITI, SHAYMIN = 1097, 140, 343
TECH = Counter(STOCK)
TECH[NIGHT_STRETCHER] -= 2
TECH[FEZANDIPITI] += 1
TECH[SHAYMIN] += 1
TECH_DECK = []
for _c in sorted(TECH):
    TECH_DECK += [_c] * TECH[_c]

PANEL = ["public-archaludon", "meta-grimmsnarl", "public-alakazam"]

# name -> (depth or None, deck, keep_templates)
ARMS = {
    "control":     (None, STOCK, False),
    "depth2":      (2, STOCK, False),
    "depth4":      (4, STOCK, False),
    "tech_basics": (None, TECH_DECK, False),
    "templates":   (None, STOCK, True),
}


def _mk(tag, depth, deck, templates):
    s = ilu.spec_from_file_location("fm_" + tag, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    # The 0.80 s deadline binds occasionally under load, so a different number of
    # determinizations completes and the agent stops being reproducible -- that is
    # the residual ~1.5% discordance seen in the stock-vs-stock control. Setting
    # it unbounded removes the last known source of harness noise.
    _b = os.environ.get("CB_BUDGET")
    if _b:
        m.TIME_BUDGET_S = float(_b)
    if not templates:
        m._TEMPLATES = []
        m._TEMPLATE_SIG = []
    if depth is not None:
        deep_search.install(m, extra_turns=depth)

    def w(o, _m=m, _d=deck):
        return list(_d) if o.get("select") is None else _m.agent(o)
    return w


def chunk(args):
    name, seeds = args
    import arena  # noqa: F401
    depth, deck, tpl = ARMS[name]
    cand = _mk(name, depth, deck, tpl)
    base = _mk("base_" + name, None, STOCK, False)
    aw = bw = a = b = 0
    for opp in PANEL:
        fo, do = build(opp)
        for s in seeds:
            x = y = 0
            for seat in (0, 1):
                random.seed(s)
                x += int(play(s, cand, fo, deck, do) == 0) if seat == 0 else \
                     int(play(s, fo, cand, do, deck) == 1)
            for seat in (0, 1):
                random.seed(s)
                y += int(play(s, base, fo, STOCK, do) == 0) if seat == 0 else \
                     int(play(s, fo, base, do, STOCK) == 1)
            aw += x
            bw += y
            if x > y:
                a += 1
            elif y > x:
                b += 1
    return name, aw, bw, a, b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    # CB_SEED0 lets a remote arm cover a DISJOINT seed range, so local and Kaggle
    # runs pool into one larger paired sample instead of repeating each other.
    seed0 = int(os.environ.get("CB_SEED0", "605000"))
    seeds = [seed0 + i for i in range(n)]
    jobs = []
    # Split every arm into several seed chunks so all workers stay busy: one job
    # per arm would leave 8 cores running only len(ARMS) tasks, and the depth4
    # arm is ~3x slower than the rest so it would straggle alone at the end.
    per = int(os.environ.get("CB_CHUNKS", 4))
    for name in ARMS:
        parts = [seeds[i::per] for i in range(per)]
        for p in parts:
            jobs.append((name, p))
    print(f"{len(ARMS)} arms x {n} seeds x {len(PANEL)} deterministic opponents; "
          f"{len(jobs)} jobs on {workers} workers", flush=True)

    agg = {k: [0, 0, 0, 0] for k in ARMS}
    with mp.get_context("fork").Pool(workers) as pool:
        for name, aw, bw, a, b in pool.imap_unordered(chunk, jobs):
            g = agg[name]
            g[0] += aw
            g[1] += bw
            g[2] += a
            g[3] += b
            print(f"    [partial] {name}: {g[0]} vs {g[1]}  disc {g[2]}/{g[3]}", flush=True)

    print("\n===== RESULTS (clean panel) =====", flush=True)
    for name in ARMS:
        aw, bw, a, b = agg[name]
        p = mcnemar(a, b)
        if name == "control":
            ok = "HARNESS OK" if (a == 0 and b == 0) else "!! LEAKING RANDOMNESS !!"
            print(f"  {name:12s} {aw:4d} vs {bw:4d}  disc {a:3d}/{b:3d}  {ok}")
            continue
        tag = "BETTER" if (a > b and p < 0.05) else ("worse" if b > a else "flat")
        print(f"  {name:12s} {aw:4d} vs {bw:4d}  disc {a:3d}/{b:3d}  p={p:.4f}  {tag}")


if __name__ == "__main__":
    main()
