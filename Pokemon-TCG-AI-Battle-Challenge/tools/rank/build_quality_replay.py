"""Deduplicate and balance elite anchors with causal changed-action rows."""
from __future__ import annotations
import argparse, hashlib, pickle, random
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--elite", required=True)
ap.add_argument("--correction", action="append", default=[])
ap.add_argument("--correction-repeat", type=int, default=12)
ap.add_argument("--cap-per-context", type=int, default=8000)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20260809)
a = ap.parse_args(); rng = random.Random(a.seed)

def key(row):
    payload = (row.get("ctxid"), tuple(row.get("cids", [])),
               tuple(row.get("types", [])), row.get("y"),
               tuple(round(float(x), 2) for x in row.get("ctx", [])))
    return hashlib.blake2b(pickle.dumps(payload, protocol=4), digest_size=16).digest()

elite = list(pickle.load(open(a.elite, "rb"))); rng.shuffle(elite)
seen=set(); counts=Counter(); out=[]
for row in elite:
    bucket=(row.get("ctxid",0), row.get("y",0), len(row.get("cids",[])))
    k=key(row)
    if k in seen or counts[bucket] >= a.cap_per_context: continue
    seen.add(k); counts[bucket]+=1; out.append(row)
elite_kept=len(out); corrections=[]; seen_corrections=set()
for path in a.correction:
    for row in pickle.load(open(path,"rb")):
        if not row.get("explored", True) or row.get("y") == row.get("baseline_y"): continue
        k=key(row)
        # Correction files are normally filtered from the anchored trajectories,
        # so their keys legitimately overlap the elite/anchor set.  Deduplicate
        # corrections against each other, not against the anchor, then repeat
        # them to implement the requested correction dose.
        if k in seen_corrections: continue
        seen_corrections.add(k); corrections.append(row)
for _ in range(a.correction_repeat): out.extend(corrections)
rng.shuffle(out)
with open(a.out,"wb") as fh: pickle.dump(out,fh,protocol=4)
print({"elite_input":len(elite),"elite_kept":elite_kept,
       "unique_corrections":len(corrections),"repeat":a.correction_repeat,
       "output":len(out)},flush=True)
