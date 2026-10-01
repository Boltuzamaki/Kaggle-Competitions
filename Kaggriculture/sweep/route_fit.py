"""Re-fit the shop-pair -> route table of the open-loop base agent.

The base ships 9 real route tapes (0-8) but its router table names routes 9-12
for five of its fifteen shop pairs; the chassis silently falls back to route 0
for those, so a third of the openings are unrouted.  More importantly the table
was fit against its author's opponent pool, not ours.

Method: force each route, play it from both seats on paired seeds against the
opponent pool, bucket results by the day-6 shop pair the router actually keys
on, and pick the argmax match score per bucket.  Ladder pays win/loss/tie only,
so bank is a diagnostic, never the objective.

  python sweep/route_fit.py --seeds 40 --procs 15 --out research/route_fit.csv
"""
import argparse, csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import variants  # noqa: E402

DAY6_STEP = 144


def build_route_variants(base_path, out_dir, routes):
    os.makedirs(out_dir, exist_ok=True)
    src = open(base_path).read()
    paths = {}
    for r in routes:
        p = os.path.join(out_dir, f"route{r}.py")
        with open(p, "w") as f:
            f.write(variants.build(src, table={}, default=r))
        paths[r] = p
    return paths


def play(job):
    route, path, opp, seed, seat = job
    from kaggle_environments import make
    agents = [path, opp] if seat == 0 else [opp, path]
    t0 = time.time()
    try:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.run(agents)
        last = env.steps[-1]
        clean = all(s["status"] == "DONE" for s in last)
        banks = [f["money"] for f in last[0]["observation"]["farms"]]
        mine, theirs = banks[seat], banks[1 - seat]
        shops = env.steps[DAY6_STEP][0]["observation"]["town"]["unlocked_shops"][:2]
    except Exception as exc:  # a crash is a ladder loss, record it as one
        return dict(route=route, opp=os.path.basename(opp), seed=seed, seat=seat,
                    pair="ERROR", score=0.0, mine=0, theirs=0, clean=False,
                    secs=round(time.time() - t0, 1), err=repr(exc)[:90])
    score = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return dict(route=route, opp=os.path.basename(opp), seed=seed, seat=seat,
                pair="|".join(shops), score=score, mine=mine, theirs=theirs,
                clean=clean, secs=round(time.time() - t0, 1), err="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "gauntlet2", "aurax7.py"))
    ap.add_argument("--opponents", nargs="*", default=None)
    ap.add_argument("--routes", type=int, nargs="*", default=list(range(9)))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "route_fit.csv"))
    ap.add_argument("--work", default=os.path.join(ROOT, "sweep", "_routes"))
    args = ap.parse_args()

    opps = args.opponents or [os.path.join(ROOT, "gauntlet2", f) for f in
                              ("ahmedberatozer.py", "tetsutani.py", "raykkretzschmar.py",
                               "boatlee.py", "prvsiyan.py")]
    paths = build_route_variants(args.base, args.work, args.routes)
    jobs = [(r, paths[r], o, s, seat)
            for r in args.routes
            for o in opps
            for s in range(args.seed_offset, args.seed_offset + args.seeds)
            for seat in (0, 1)]
    print(f"{len(jobs)} games | routes={args.routes} | {len(opps)} opponents "
          f"| seeds {args.seed_offset}..{args.seed_offset + args.seeds - 1} | {args.procs} procs",
          flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()
    with open(args.out, "w", newline="") as fh:
        w = None
        with ProcessPoolExecutor(args.procs) as ex:
            for i, rec in enumerate(ex.map(play, jobs, chunksize=1), 1):
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(rec))
                    w.writeheader()
                w.writerow(rec)
                if i % 50 == 0:
                    fh.flush()
                    print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
