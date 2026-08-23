"""Build a reproducible elite-majority DAgger mixture."""
from __future__ import annotations
import argparse, pickle, random

ap = argparse.ArgumentParser()
ap.add_argument("--elite", required=True)
ap.add_argument("--selfplay", action="append", default=[])
ap.add_argument("--per-selfplay", type=int, default=30000)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20260809)
a = ap.parse_args()
rng = random.Random(a.seed)
rows = list(pickle.load(open(a.elite, "rb")))
print("elite", len(rows), flush=True)
for path in a.selfplay:
    part = list(pickle.load(open(path, "rb")))
    rng.shuffle(part)
    kept = part[:min(a.per_selfplay, len(part))]
    rows.extend(kept)
    print(path, len(part), "kept", len(kept), flush=True)
rng.shuffle(rows)
with open(a.out, "wb") as fh:
    pickle.dump(rows, fh, protocol=4)
print("wrote", len(rows), a.out, flush=True)
