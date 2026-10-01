"""Local league harness for Kaggriculture.

The market is shared, so a matchup is NOT symmetric: always play both seats.
The ladder scores win/loss/tie only, so win-rate is the objective and bank is
only a diagnostic.

Usage:
  python eval.py A B [--seeds N] [--steps N] [--procs N]
  python eval.py main.py starter --seeds 8
  python eval.py main.py main_v2.py --seeds 8
"""
import argparse
import os
import sys
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("PYTHONHASHSEED", "0")


def _run_one(args):
    a, b, seed, steps = args
    from kaggle_environments import make
    try:
        env = make("kaggriculture",
                   configuration={"episodeSteps": steps, "seed": seed},
                   debug=False)
        env.run([a, b])
        last = env.steps[-1]
        r0 = last[0].reward if last[0].reward is not None else 0.0
        r1 = last[1].reward if last[1].reward is not None else 0.0
        s0, s1 = last[0].status, last[1].status
        return dict(seed=seed, r0=r0, r1=r1, s0=s0, s1=s1, err=None)
    except Exception as e:
        return dict(seed=seed, r0=0.0, r1=0.0, s0="ERROR", s1="ERROR",
                    err=f"{e}\n{traceback.format_exc()[-500:]}")


def league(a, b, seeds, steps, procs):
    jobs = []
    for s in range(seeds):
        jobs.append((a, b, s, steps))     # A in seat 0
        jobs.append((b, a, s, steps))     # A in seat 1
    t0 = time.time()
    if procs > 1:
        with ProcessPoolExecutor(max_workers=procs) as ex:
            res = list(ex.map(_run_one, jobs))
    else:
        res = [_run_one(j) for j in jobs]

    wins = losses = ties = 0
    a_banks, b_banks = [], []
    errors = []
    for i, r in enumerate(res):
        a_seat0 = (i % 2 == 0)
        ra = r["r0"] if a_seat0 else r["r1"]
        rb = r["r1"] if a_seat0 else r["r0"]
        sa = r["s0"] if a_seat0 else r["s1"]
        sb = r["s1"] if a_seat0 else r["s0"]
        if r["err"] or sa != "DONE" or sb != "DONE":
            errors.append((r["seed"], a_seat0, sa, sb, r["err"]))
        a_banks.append(ra)
        b_banks.append(rb)
        if ra > rb:
            wins += 1
        elif ra < rb:
            losses += 1
        else:
            ties += 1
    n = len(res)
    dt = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  {a}   vs   {b}")
    print(f"{'='*70}")
    print(f"  games      : {n}  ({seeds} seeds x 2 seats)   [{dt:.0f}s]")
    print(f"  record     : {wins}W - {losses}L - {ties}T"
          f"   win% = {100.0*wins/max(1,n):.1f}")
    print(f"  mean bank  : A={sum(a_banks)/n:>10,.0f}   B={sum(b_banks)/n:>10,.0f}"
          f"   margin={sum(a_banks)/n - sum(b_banks)/n:>+10,.0f}")
    print(f"  A bank rng : {min(a_banks):,.0f} .. {max(a_banks):,.0f}")
    if errors:
        print(f"  !! ERRORS  : {len(errors)}")
        for e in errors[:3]:
            print(f"     seed={e[0]} a_seat0={e[1]} statuses={e[2]}/{e[3]}")
            if e[4]:
                print("     " + e[4].replace("\n", "\n     ")[:600])
    else:
        print("  errors     : none  (all DONE)")
    return dict(wins=wins, losses=losses, ties=ties, n=n,
                a_mean=sum(a_banks)/n, b_mean=sum(b_banks)/n, errors=len(errors))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    r = league(args.a, args.b, args.seeds, args.steps, args.procs)
    if args.json:
        json.dump(r, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()
