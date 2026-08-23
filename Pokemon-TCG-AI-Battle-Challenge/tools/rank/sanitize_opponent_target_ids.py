"""Remove card identities corrupted by the historical opponent-zone resolver bug.

Old datasets resolved every CARD option through the acting player's zones even
when ``playerIndex`` identified the opponent.  Until a corpus is regenerated,
zeroing those identities is conservative: the model retains option type/zone
and public state, but cannot learn a false card association.
"""
from __future__ import annotations

import argparse
import pickle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = pickle.load(open(args.data, "rb"))
    cleared = 0
    for row in rows:
        seat = int(row.get("seat", 0))
        cids = list(row["cids"])
        for index, nums in enumerate(row["nums"]):
            player_index = int(nums[3])
            if player_index >= 0 and player_index != seat and cids[index] != 0:
                cids[index] = 0
                cleared += 1
        row["cids"] = cids
    with open(args.out, "wb") as handle:
        pickle.dump(rows, handle, protocol=4)
    print({"rows": len(rows), "cleared_option_ids": cleared})


if __name__ == "__main__":
    main()
