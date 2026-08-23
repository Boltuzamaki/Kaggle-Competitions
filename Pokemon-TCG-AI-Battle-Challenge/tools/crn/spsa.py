"""SPSA tuning of the policy priority weights against the LEAGUE.

Method from the Approvers chess write-up ("SPSA for tuning various constants").
SPSA perturbs every parameter at once with a random +-1 vector and needs only two
objective evaluations per iteration regardless of dimension -- the right tool when
each evaluation costs hundreds of games and the objective is very noisy.

Three earlier tuning failures are addressed by construction:
  * weak opponent  -> objective is the weighted LEAGUE, not two fixed agents;
  * pure noise     -> every evaluation is PAIRED on a fixed seed set, so the
                      comparison shares shuffles and flips;
  * overfitting    -> the final genome is confirmed on a DISJOINT seed set before
                      it is written out, the gate that correctly rejected our
                      earlier Kaggle sweep.

Runs indefinitely until the iteration budget is exhausted; safe to leave overnight.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from paired_eval import play, build  # noqa: E402
import hybrid_agent  # noqa: E402
import meta_decks  # noqa: E402

# Which weight table to tune, and which deck to pilot.
_MOD = os.environ.get("SPSA_MODULE", "tuned_policy")
_DECKNAME = os.environ.get("SPSA_DECK", "GARCHOMP")
_WFILE = os.environ.get("SPSA_WFILE", "policy_w.json")
tuned_policy = __import__(_MOD)
DECK = getattr(meta_decks, _DECKNAME)
POOL = [("public-archaludon", 2.0), ("public-alakazam", 2.0),
        ("hybs-grimmsnarl", 1.0), ("td-td_08", 1.0)]


def objective(genome, opps, seeds):
    tuned_policy.W.clear()
    tuned_policy.W.update(genome)

    def cand(obs):
        return hybrid_agent.hybrid_agent(obs, DECK, policy_module=tuned_policy)

    num = den = 0.0
    for fo, do, w in opps:
        for s in seeds:
            got = 0
            for seat in (0, 1):
                random.seed(s)
                if seat == 0:
                    got += int(play(s, cand, fo, DECK, do) == 0)
                else:
                    got += int(play(s, fo, cand, do, DECK) == 1)
            num += w * got
            den += w * 2
    return num / max(den, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--confirm-seeds", type=int, default=24)
    ap.add_argument("--a", type=float, default=0.12, help="step size")
    ap.add_argument("--c", type=float, default=0.15, help="perturbation size")
    a = ap.parse_args()

    import arena  # noqa: F401
    opps = []
    for name, w in POOL:
        try:
            fo, do = build(name)
            opps.append((fo, do, w))
        except SystemExit:
            print(f"  (skip {name})")

    base = dict(tuned_policy.W)
    keys = sorted(base)
    rng = random.Random(4242)
    seeds = [200000 + i for i in range(a.seeds)]

    cur = dict(base)
    best = dict(base)
    base_score = objective(base, opps, seeds)
    best_score = base_score
    print(f"baseline league score {base_score:.4f}  ({len(keys)} params)", flush=True)

    hist = []
    t0 = time.time()
    for k in range(1, a.iters + 1):
        ak = a.a / (k ** 0.602)
        ck = a.c / (k ** 0.101)
        delta = {p: (1 if rng.random() < 0.5 else -1) for p in keys}
        plus = {p: max(1e-6, cur[p] * (1 + ck * delta[p])) for p in keys}
        minus = {p: max(1e-6, cur[p] * (1 - ck * delta[p])) for p in keys}
        yp = objective(plus, opps, seeds)
        ym = objective(minus, opps, seeds)
        g = (yp - ym) / (2 * ck)
        cur = {p: max(1e-6, cur[p] * (1 + ak * g * delta[p])) for p in keys}
        sc = objective(cur, opps, seeds)
        if sc > best_score:
            best_score, best = sc, dict(cur)
        hist.append({"iter": k, "plus": yp, "minus": ym, "cur": sc})
        print(f"  iter {k:3d}  +{yp:.3f} -{ym:.3f}  cur {sc:.3f}  "
              f"best {best_score:.3f}  ({(time.time()-t0)/60:.0f}m)", flush=True)

    print("\nconfirming on disjoint seeds...")
    cs = [500000 + i for i in range(a.confirm_seeds)]
    cb = objective(base, opps, cs)
    cc = objective(best, opps, cs)
    promote = cc > cb
    print(f"  baseline  {cb:.4f}")
    print(f"  tuned     {cc:.4f}")
    print(f"  promote:  {promote}")

    out = os.path.join(ROOT, "scratchpad", "scrape_20260804", os.environ.get("SPSA_OUT", "spsa_result.json"))
    json.dump({"baseline": base, "best": best, "search_score": best_score,
               "confirm_baseline": cb, "confirm_tuned": cc,
               "promote": bool(promote), "history": hist}, open(out, "w"), indent=1)
    print("wrote", out)
    if promote:
        wp = os.path.join(ROOT, "agent", _WFILE)
        json.dump(best, open(wp, "w"), indent=1)
        print("wrote", wp, "(NOT submitted -- needs approval)")


if __name__ == "__main__":
    main()
