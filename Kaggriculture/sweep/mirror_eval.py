"""Evaluate variants of the base agent against the base itself.

Why mirror and not the gauntlet: every one of the nine public agents we can
extract loses 100% of games to this base, so the gauntlet has no resolving
power above it -- a pilot over all nine route tapes scored a flat 1.000 against
two of them.  The base is the only opponent locally that can still tell two
candidates apart.  Bank margin in the mirror is the fine-grained signal; win
rate is what the ladder actually pays, so both are reported.

  python sweep/mirror_eval.py --specs specs/flags.json --seeds 10
"""
import argparse, csv, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import variants  # noqa: E402


def materialise(base_path, specs, work):
    os.makedirs(work, exist_ok=True)
    src = open(base_path).read()
    paths = {}
    for name, spec in specs.items():
        p = os.path.join(work, f"{name}.py")
        with open(p, "w") as f:
            f.write(variants.build_spec(src, spec))
        paths[name] = p
    return paths


def play(job):
    name, path, base, seed, seat = job
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
        shops = env.steps[144][0]["observation"]["town"]["unlocked_shops"][:2]
    except Exception as exc:
        return dict(variant=name, seed=seed, seat=seat, pair="ERROR", score=0.0,
                    mine=0, theirs=0, clean=False, secs=round(time.time() - t0, 1),
                    err=repr(exc)[:90])
    score = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
    return dict(variant=name, seed=seed, seat=seat, pair="|".join(shops), score=score,
                mine=mine, theirs=theirs, clean=clean,
                secs=round(time.time() - t0, 1), err="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "gauntlet2", "aurax7.py"))
    ap.add_argument("--specs", required=True, help="JSON {name: spec}")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default=os.path.join(ROOT, "research", "mirror_eval.csv"))
    ap.add_argument("--work", default=os.path.join(ROOT, "sweep", "_variants"))
    args = ap.parse_args()

    raw = json.load(open(args.specs))
    specs = {}
    for name, spec in raw.items():
        fix = spec.get("router_fix")
        if fix:  # JSON cannot key on tuples; accept "A|B"
            spec = dict(spec, router_fix={tuple(k.split("|")): v for k, v in fix.items()})
        specs[name] = spec

    paths = materialise(args.base, specs, args.work)
    jobs = [(n, paths[n], args.base, s, seat)
            for n in specs
            for s in range(args.seed_offset, args.seed_offset + args.seeds)
            for seat in (0, 1)]
    print(f"{len(jobs)} mirror games | {len(specs)} variants | "
          f"seeds {args.seed_offset}..{args.seed_offset + args.seeds - 1} | "
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
                    print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
