"""Mine stable, interpretable action-substitution rules from paired outcomes."""
from __future__ import annotations

import argparse
import pickle
from collections import defaultdict


def option_key(row, index):
    nums = row["nums"][index]
    # Card/type plus area, player, attack/ability and in-play target encode the
    # public semantic identity without depending on option ordering.
    return (int(row["cids"][index]), int(row["types"][index]),
            int(nums[0]), int(nums[1]), int(nums[3]), int(nums[4]), int(nums[5]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--min-support", type=int, default=8)
    args = parser.parse_args()
    rows = pickle.load(open(args.data, "rb"))
    stats = defaultdict(lambda: [set(), set(), set(), set()])
    for row in rows:
        baseline, chosen = int(row["baseline_y"]), int(row["y"])
        if not (0 <= baseline < len(row["cids"]) and 0 <= chosen < len(row["cids"])):
            continue
        key = (str(row["opponent"]), int(row["ctxid"]),
               option_key(row, baseline), option_key(row, chosen))
        episode = str(row["episode"])
        positive = int(row["advantage"]) > 0
        heldout = int(row["seed"]) % 5 == 0
        stats[key][2 * heldout + (0 if positive else 1)].add(episode)

    candidates = []
    for key, groups in stats.items():
        train_pos, train_neg, valid_pos, valid_neg = map(len, groups)
        train_n, valid_n = train_pos + train_neg, valid_pos + valid_neg
        if train_n < args.min_support or valid_n < 2:
            continue
        train_precision = train_pos / train_n
        valid_precision = valid_pos / valid_n
        if train_precision >= 0.70 and valid_precision >= 0.65:
            candidates.append((valid_precision, valid_n, train_precision, train_n, key,
                               train_pos, train_neg, valid_pos, valid_neg))
    candidates.sort(reverse=True)
    for item in candidates[:100]:
        vp, vn, tp, tn, key, a, b, c, d = item
        print({"opponent": key[0], "ctx": key[1], "baseline": key[2], "choose": key[3],
               "train": f"{a}/{tn}={tp:.3f}", "heldout": f"{c}/{vn}={vp:.3f}"})
    print("stable_rules", len(candidates))


if __name__ == "__main__":
    main()
