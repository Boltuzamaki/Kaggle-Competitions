"""Split a counterfactual pairwise dataset into opponent-specific files."""
from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.data, "rb") as handle:
        rows = pickle.load(handle)
    grouped = defaultdict(list)
    for row in rows:
        if int(row.get("advantage", 0)) != 0:
            grouped[str(row.get("opponent", "unknown"))].append(row)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for opponent, part in sorted(grouped.items()):
        path = output / f"{opponent}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(part, handle, protocol=4)
        counts = Counter(int(row["advantage"]) for row in part)
        print(opponent, len(part), dict(sorted(counts.items())), path)


if __name__ == "__main__":
    main()
