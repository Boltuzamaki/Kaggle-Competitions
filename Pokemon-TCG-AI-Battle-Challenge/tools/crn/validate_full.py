"""Two-part gate a tuned weight file must clear before it earns a submission slot.

  MIRROR   candidate vs stock head-to-head, on seeds disjoint from tuning.
           This is what beat_stock optimises, so passing it only shows the tuner
           did not overfit its own seeds.

  FIELD    candidate vs stock, both played against a panel of DIFFERENT
           deterministic opponents, paired on shared seeds.

Both matter, and the second is the one that has been missing. Ratings in this
game are strongly non-transitive -- our own agents measured 68.8% against
Alakazam and 25% against Archaludon -- so a genome that beats the mirror can
easily be worse against the actual field. beat_stock tunes purely on the mirror,
which makes a field check mandatory rather than optional.

Panel is restricted to opponents verified reproducible run-to-run; td-td_08 and
hybs-other are excluded because they return different results for the same seed
and silently destroy the pairing.
"""
from __future__ import annotations

import importlib.util as ilu
import json
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build, mcnemar  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]
PANEL = ["public-archaludon", "meta-grimmsnarl", "public-alakazam"]


def _load(tag, weights=None):
    s = ilu.spec_from_file_location("fm_" + tag, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    m.TIME_BUDGET_S = 1e9          # deadline never binds -> reproducible
    if weights:
        m.WEIGHTS.clear()
        m.WEIGHTS.update(weights)
    return m


def mirror(args):
    wfile, seeds = args
    w = json.load(open(wfile))
    c, s_ = _load("mc", w), _load("ms")
    a = b = cw = sw = 0
    for sd in seeds:
        x = y = 0
        for seat in (0, 1):
            random.seed(sd)
            r = play(sd, c.agent, s_.agent, DECK, DECK) if seat == 0 else \
                play(sd, s_.agent, c.agent, DECK, DECK)
            x += int(r == (0 if seat == 0 else 1))
            y += int(r == (1 if seat == 0 else 0))
        cw += x; sw += y
        if x > y: a += 1
        elif y > x: b += 1
    return ("mirror", cw, sw, a, b)


def field(args):
    wfile, seeds = args
    import arena  # noqa: F401
    w = json.load(open(wfile))
    c, s_ = _load("fc", w), _load("fs")
    a = b = cw = sw = 0
    for opp in PANEL:
        fo, do = build(opp)
        for sd in seeds:
            x = y = 0
            for seat in (0, 1):
                random.seed(sd)
                x += int(play(sd, c.agent, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(sd, fo, c.agent, do, DECK) == 1)
            for seat in (0, 1):
                random.seed(sd)
                y += int(play(sd, s_.agent, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(sd, fo, s_.agent, do, DECK) == 1)
            cw += x; sw += y
            if x > y: a += 1
            elif y > x: b += 1
    return ("field", cw, sw, a, b)


def main():
    wfile = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    seeds = [314000 + i for i in range(n)]
    k = max(1, workers // 2)
    jobs = [(mirror, (wfile, seeds[i::k])) for i in range(k)]
    jobs += [(field, (wfile, seeds[i::k])) for i in range(k)]
    print(f"validating {wfile}: {n} disjoint seeds, mirror + {len(PANEL)}-opponent field",
          flush=True)

    agg = {"mirror": [0, 0, 0, 0], "field": [0, 0, 0, 0]}
    with mp.get_context("fork").Pool(workers) as pool:
        for tag, cw, sw, a, b in pool.imap_unordered(_run, jobs):
            g = agg[tag]
            g[0] += cw; g[1] += sw; g[2] += a; g[3] += b

    ok = True
    for tag in ("mirror", "field"):
        cw, sw, a, b = agg[tag]
        p = mcnemar(a, b)
        good = a > b and p < 0.05
        neutral = not (b > a and p < 0.05)
        print(f"  {tag:6s} {cw:4d} vs stock {sw:4d}  disc {a:3d}/{b:3d}  p={p:.4f}  "
              f"{'BETTER' if good else ('worse' if b > a else 'flat')}")
        # mirror must be a real win; field must at minimum not be a real loss
        ok = ok and (good if tag == "mirror" else neutral)
    print("\n  SHIPPABLE" if ok else "\n  DO NOT SHIP")


def _run(job):
    fn, args = job
    return fn(args)


if __name__ == "__main__":
    main()
