"""Fit the router's shop-pair -> route table, one pair at a time.

The base routes only 15 of the 64 possible day-6 shop pairs, and five of those
15 name route tapes that were never shipped, so the chassis quietly falls back
to route 0.  Everything else -- about three quarters of openings -- is unrouted
by construction.

A router entry for one pair only changes play on the seeds that actually draw
it, so each (pair, route) cell is measured on that pair's seeds only, against
the unmodified base.  On every other seed the variant is byte-identical to the
base and the game would be a guaranteed tie, which is why they are not played.

  python sweep/pair_sweep.py --shard 0 --shards 7 --seeds-per-pair 6
"""
import argparse, csv, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import variants  # noqa: E402


def cells(index, routes, seeds_per_pair, shard, shards):
    by_pair = index["by_pair"]
    out = []
    for pair in sorted(by_pair):
        seeds = sorted(by_pair[pair])[:seeds_per_pair]
        if not seeds:
            continue
        for r in routes:
            out.append((pair, r, seeds))
    return [c for i, c in enumerate(out) if i % shards == shard]


def play(job):
    pair, route, path, base, seed, seat = job
    from kaggle_environments import make
    agents = [path, base] if seat == 0 else [base, path]
    t0 = time.time()
    try:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.run(agents)
        last = env.steps[-1]
        clean = all(s["status"] == "DONE" for s in last)
        banks = [f["money"] for f in last[0]["observation"]["farms"]]
        mine, theirs = banks[seat], banks[1 - seat]
    except Exception as exc:
        return dict(pair=pair, route=route, seed=seed, seat=seat, score=0.0,
                    mine=0, theirs=0, clean=False, secs=round(time.time() - t0, 1),
                    err=repr(exc)[:90])
    score = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return dict(pair=pair, route=route, seed=seed, seat=seat, score=score,
                mine=mine, theirs=theirs, clean=clean,
                secs=round(time.time() - t0, 1), err="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "gauntlet2", "aurax7.py"))
    ap.add_argument("--index", default=os.path.join(ROOT, "research", "seed_pairs.json"))
    ap.add_argument("--routes", type=int, nargs="*", default=list(range(9)))
    ap.add_argument("--seeds-per-pair", type=int, default=6)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "pair_sweep.csv"))
    ap.add_argument("--work", default=os.path.join(ROOT, "sweep", "_pairs"))
    args = ap.parse_args()

    index = json.load(open(args.index))
    todo = cells(index, args.routes, args.seeds_per_pair, args.shard, args.shards)
    os.makedirs(args.work, exist_ok=True)
    src = open(args.base).read()

    jobs = []
    for pair, route, seeds in todo:
        key = pair.replace("|", "_") + f"_r{route}"
        path = os.path.join(args.work, key + ".py")
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write(variants.build_spec(src, {"router_fix": {tuple(pair.split("|")): route}}))
        for s in seeds:
            for seat in (0, 1):
                jobs.append((pair, route, path, args.base, s, seat))

    print(f"shard {args.shard}/{args.shards}: {len(todo)} cells, {len(jobs)} games, "
          f"{args.procs} procs", flush=True)
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
                fh.flush()
                if i % 25 == 0:
                    print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
