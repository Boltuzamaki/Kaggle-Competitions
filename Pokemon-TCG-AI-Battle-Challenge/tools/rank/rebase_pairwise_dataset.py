"""Move the recorded baseline action to option index zero for pairwise models."""
from __future__ import annotations

import argparse
import pickle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = pickle.load(open(args.data, "rb"))
    output = []
    for source in rows:
        row = dict(source)
        baseline = int(row.get("baseline_y", 0))
        chosen = int(row["y"])
        option_count = len(row["cids"])
        if not (0 <= baseline < option_count and 0 <= chosen < option_count):
            continue
        if baseline:
            for key in ("cids", "types", "nums"):
                values = list(row[key])
                values[0], values[baseline] = values[baseline], values[0]
                row[key] = values
            if chosen == 0:
                chosen = baseline
            elif chosen == baseline:
                chosen = 0
        row["y"] = chosen
        row["baseline_y"] = 0
        # Exploration rows must represent a genuine deviation after rebasing.
        if chosen == 0:
            continue
        output.append(row)
    with open(args.out, "wb") as handle:
        pickle.dump(output, handle, protocol=4)
    print({"input": len(rows), "output": len(output), "baseline_index": 0})


if __name__ == "__main__":
    main()
