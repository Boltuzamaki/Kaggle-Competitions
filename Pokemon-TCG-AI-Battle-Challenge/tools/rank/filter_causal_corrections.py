"""Keep only actions actually changed in matched loss-to-win trajectories."""
from __future__ import annotations
import argparse, pickle
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--single-deviation", action="store_true",
                help="keep only episodes containing exactly one changed action")
a = ap.parse_args()
rows = pickle.load(open(a.input, "rb"))
kept = [row for row in rows if row.get("explored") and row.get("y") != row.get("baseline_y")]
if a.single_deviation:
    counts = Counter(str(row["episode"]) for row in kept)
    kept = [row for row in kept if counts[str(row["episode"])] == 1]
with open(a.out, "wb") as fh:
    pickle.dump(kept, fh, protocol=4)
print("input", len(rows), "causal changed actions", len(kept),
      "single_deviation", a.single_deviation, flush=True)
