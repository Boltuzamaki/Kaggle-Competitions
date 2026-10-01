"""Parameter sweep for main.py.

Each config is evaluated as mean final bank over N seeds x both seats against a
fixed opponent. Bank is the tuning signal; win-rate against real agents is the
final arbiter (see eval.py) -- bank is just far less noisy per game.

  python tune.py --opp starter --seeds 4 --spec sweep.json
"""
import argparse
import itertools
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor


def _run(task):
    name, override, seed, seat, steps, opp = task
    os.environ["KAGGRICULTURE_P"] = json.dumps(override)
    from kaggle_environments import make
    agents = ["main.py", opp] if seat == 0 else [opp, "main.py"]
    try:
        env = make("kaggriculture",
                   configuration={"episodeSteps": steps, "seed": seed},
                   debug=False)
        env.run(agents)
        last = env.steps[-1]
        me = last[seat].reward or 0.0
        them = last[1 - seat].reward or 0.0
        ok = last[seat].status == "DONE"
        return name, me, them, ok
    except Exception as e:
        return name, 0.0, 0.0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="starter",
                    help="comma-separated opponents; use real agents, not starter")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--spec", required=True,
                    help="JSON: {'base': {...}, 'grid': {'key': [v1, v2]}}")
    args = ap.parse_args()

    spec = json.load(open(args.spec))
    base = spec.get("base", {})
    grid = spec.get("grid", {})
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys))) or [()]

    configs = {}
    for combo in combos:
        override = dict(base)
        override.update(dict(zip(keys, combo)))
        name = ",".join(f"{k}={v}" for k, v in zip(keys, combo)) or "base"
        configs[name] = override

    opponents = [o.strip() for o in args.opp.split(",") if o.strip()]
    tasks = []
    for name, ov in configs.items():
        for opp in opponents:
            for s in range(args.seeds):
                for seat in (0, 1):
                    tasks.append((name, ov, s, seat, args.steps, opp))

    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        results = list(ex.map(_run, tasks))

    agg = {}
    for name, me, them, ok in results:
        d = agg.setdefault(name, {"me": [], "score": 0.0, "n": 0, "err": 0})
        d["me"].append(me)
        d["n"] += 1
        # ladder pays win/tie/loss only
        d["score"] += 1.0 if me > them else (0.5 if me == them else 0.0)
        if not ok:
            d["err"] += 1

    rows = []
    for name, d in agg.items():
        rows.append((d["score"] / max(1, d["n"]), statistics.mean(d["me"]),
                     name, d["n"], d["err"]))
    rows.sort(reverse=True)
    print(f"\n{'score':>7}  {'mean bank':>12}  {'games':>6}  {'err':>4}  config")
    print("-" * 92)
    for score, mean, name, n, err in rows:
        print(f"{score:>7.3f}  {mean:>12,.0f}  {n:>6}  {err:>4}  {name}")


if __name__ == "__main__":
    main()
