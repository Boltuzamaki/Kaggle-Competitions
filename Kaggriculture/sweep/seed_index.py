"""Index seeds by the day-6 shop pair the router keys on.

A router entry for (YARN_STORE, PET_CAFE) only changes anything on the ~1 seed
in 64 that actually draws that pair, so measuring it on unstratified seeds
spends 63/64 of the compute on games where the variant is byte-identical to the
base.  Shops are drawn by the environment's own RNG and are fixed by the seed,
so the pair can be read off a 145-step run with two PASS agents -- about 200 ms
instead of the ~25 s a full mirror game costs.

  python sweep/seed_index.py --seeds 4000 --out research/seed_pairs.json
"""
import argparse, collections, json, os, time
from concurrent.futures import ProcessPoolExecutor

DAY6_STEP = 144


def pair_for(seed):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": DAY6_STEP + 2, "seed": seed})
    env.run(["pass", "pass"])
    shops = env.steps[DAY6_STEP][0]["observation"]["town"]["unlocked_shops"][:2]
    return seed, tuple(shops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2000)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", default="research/seed_pairs.json")
    args = ap.parse_args()
    t0 = time.time()
    idx = {}
    with ProcessPoolExecutor(args.procs) as ex:
        for seed, pair in ex.map(pair_for, range(args.seeds), chunksize=16):
            idx[seed] = list(pair)
    by = collections.defaultdict(list)
    for s, p in idx.items():
        by["|".join(p)].append(s)
    json.dump({"by_seed": idx, "by_pair": by}, open(args.out, "w"))
    print(f"{args.seeds} seeds in {time.time()-t0:.0f}s | {len(by)} distinct pairs")
    for p, ss in sorted(by.items(), key=lambda kv: -len(kv[1]))[:5]:
        print(f"  {p:<34} {len(ss)}")
    print(f"  ... rarest: {min(len(v) for v in by.values())} seeds")


if __name__ == "__main__":
    main()
