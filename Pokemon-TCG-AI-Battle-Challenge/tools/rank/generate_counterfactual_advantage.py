"""Generate paired router deviations with win/loss advantage labels."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle

os.environ.setdefault("TRAJ_BASE", "router-v12")
from generate_winner_trajectories import OPPONENTS, init_worker, play_job


def paired_group(group):
    seed, seat, opponent, epsilons = group
    baseline_win, _ = play_job((seed, seat, opponent, 0.0, True))
    kept = []
    for epsilon in epsilons:
        explored_win, rows = play_job((seed, seat, opponent, epsilon, True))
        advantage = int(explored_win) - int(baseline_win)
        for row in rows:
            if not row.get("explored") or row.get("y") == row.get("baseline_y"):
                continue
            row["baseline_win"] = bool(baseline_win)
            row["explored_win"] = bool(explored_win)
            row["advantage"] = advantage
            kept.append(row)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=2300000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epsilons", default="0.01,0.02,0.05")
    ap.add_argument("--opponents", default=",".join(OPPONENTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    epsilons = tuple(float(x) for x in args.epsilons.split(","))
    opponents = tuple(x.strip() for x in args.opponents.split(",") if x.strip())
    groups = [(args.seed0 + i, seat, opponent, epsilons)
              for i in range(args.seeds) for seat in (0, 1)
              for opponent in opponents]
    rows = []
    with mp.get_context("fork").Pool(args.workers, initializer=init_worker) as pool:
        for index, part in enumerate(pool.imap_unordered(paired_group, groups, chunksize=2), 1):
            rows.extend(part)
            if index % 200 == 0:
                print(index, len(groups), len(rows), flush=True)
    with open(args.out, "wb") as handle:
        pickle.dump(rows, handle, protocol=4)
    counts = {value: sum(row["advantage"] == value for row in rows)
              for value in (-1, 0, 1)}
    print({"rows": len(rows), "advantage_counts": counts}, flush=True)


if __name__ == "__main__":
    main()
