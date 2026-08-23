"""Which of the 47 field-representative opponents are reproducible?

A league is only worth building from opponents that return the same result for the
same seed. td-td_08 and hybs-other are known to fail that test, and because they
sat in most panels a large share of every "discordant seed" count was the opponent
disagreeing with itself rather than the change under test.

Each opponent plays a fixed, RNG-free reference agent twice over the same seeds.
Any disagreement between the two runs is nondeterminism in the opponent, since the
engine itself is verified reproducible when both sides are fixed policies.

Writes tools/crn/clean_opponents.json for the league to consume.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools"),
          os.path.join(ROOT, "references", "top_rankers")):
    sys.path.insert(0, p)

from paired_eval import play, build  # noqa: E402

DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]


def _fixed(o):
    """Reference agent: first legal option, no RNG, no clock."""
    if o.get("select") is None:
        return list(DECK)
    sel = o["select"]
    n = len(sel.get("option") or [])
    mc = sel.get("maxCount") or 1
    return list(range(min(mc, n)))


def check(name):
    import arena  # noqa: F401
    seeds = [994000 + i for i in range(int(os.environ.get("DS_SEEDS", "6")))]
    runs = []
    try:
        for _r in range(2):
            fo, do = build(name)
            out = []
            for s in seeds:
                random.seed(s)
                out.append(play(s, _fixed, fo, DECK, do))
            runs.append(out)
    except SystemExit:
        return name, None, "unknown competitor"
    except Exception as e:
        return name, None, f"{type(e).__name__}: {e}"[:60]
    return name, runs[0] == runs[1], ""


def main():
    import arena
    names = [c[0] for c in arena.COMPETITORS
             if c[0].startswith(("public-", "meta-", "td-", "hybs-"))]
    workers = int(os.environ.get("DS_WORKERS", "6"))
    print(f"checking {len(names)} opponents on {workers} workers", flush=True)

    ok, bad, err = [], [], []
    with mp.get_context("fork").Pool(workers) as pool:
        for name, det, msg in pool.imap_unordered(check, names):
            if det is None:
                err.append((name, msg))
                print(f"  {name:22s} ERROR {msg}", flush=True)
            elif det:
                ok.append(name)
                print(f"  {name:22s} deterministic", flush=True)
            else:
                bad.append(name)
                print(f"  {name:22s} NONDETERMINISTIC -- excluded", flush=True)

    out = os.path.join(HERE, "clean_opponents.json")
    json.dump({"deterministic": sorted(ok), "nondeterministic": sorted(bad),
               "errors": err}, open(out, "w"), indent=1)
    print(f"\n  usable: {len(ok)} / {len(names)}   excluded: {len(bad)}   errors: {len(err)}")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
