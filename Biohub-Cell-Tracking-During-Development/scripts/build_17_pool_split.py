"""Build notebooks/17_pool_split.ipynb - is a division missed by the POOL or by the BUDGET?

`16_div_rank_offline` tried to answer this on the public dumps and could not:
468,454 candidates containing 2 positives, because those 8 films hold ~4 labelled
divisions between them. For a rare-event metric the sample size that matters is
the number of *divisions*, not the number of candidates.

This runs the same analysis against our own held-out validator predictions from
`12_div_probe`, which covers 24 films selected division-first, and reports the
one split that decides how much the ranking fix can be worth:

* **budget-starved** - a proposal that would complete the division exists in the
  pool, but the geometric key never brings it inside the cap. `15_div_rank`
  fixes exactly these.
* **pool-absent** - no such proposal exists at any budget. Blocked earlier, by
  the `len(kids) != 1` precondition, the mutual-nearest-neighbour requirement, or
  the distance gates. A ranker cannot help; the generator has to change.
* **already-forked** - the pipeline already predicted the fork.

The ratio between the first two is the expected value of the whole division plan.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Pool-absent vs budget-starved - CPU

**No GPU.** Reads held-out validator predictions from `12_div_probe` plus the competition
ground-truth graphs.

### Why this notebook exists

The fork budget is ~5 slots per frame against a pool of ~13,000 proposals per film, and the
key that spends it (`parent_dist + 0.15 * sister_dist`, ascending) has AUC **0.349** against
real division geometry - it is anti-correlated with the target. Fixing that is only worth
something if the proposals it should be choosing are actually *in* the pool.

A first attempt at this measurement on public prediction dumps returned **2 positives in
468,454 candidates** and settled nothing. The dumps cover 8 films holding ~4 divisions. This
run uses 24 films chosen division-first.

### The three outcomes, per ground-truth division

| outcome | meaning | does the ranking fix help? |
|---|---|---|
| **already-forked** | the prediction already forks at the right place | n/a - already a TP |
| **budget-starved** | a completing proposal is in the pool, outside the cap | **yes** |
| **pool-absent** | no completing proposal at any budget | **no** - needs a wider generator | If `pool-absent` dominates, `15_div_rank` will disappoint and the effort belongs on the
candidate generator (the mutual-NN requirement is the first suspect). If `budget-starved`
dominates, the sign fix is the whole game.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_div_metric.py"),

    code(r"""
import sys, warnings, json, math
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import biohub_ct as B
import biohub_metric as M
import biohub_div_metric as D

COMP = next(p for p in [Path("/kaggle/input/biohub-cell-tracking-during-development"),
                        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")]
            if p.exists())
TRAIN = COMP / "train"
SCALE = B.SCALE

# the probe kernel's output; locate the predicted .geff graphs wherever they landed
ROOTS = [p for p in Path("/kaggle/input").iterdir() if p.is_dir()]
PRED_GEFFS = []
for r in ROOTS:
    PRED_GEFFS += [p for p in r.rglob("*.geff") if "predictions" in str(p)]
print(f"input roots: {[r.name for r in ROOTS]}")
print(f"predicted .geff graphs found: {len(PRED_GEFFS)}")
if not PRED_GEFFS:
    for r in ROOTS:
        print(" ", r, "->", sorted(x.name for x in r.iterdir())[:12])
    raise SystemExit("no predicted graphs found - check the kernel_sources attachment")

# the shipped candidate-generation settings
SAFE_DIV_MAX_UM             = 9.0
SAFE_DIV_SISTER_MAX_UM      = 14.0
SAFE_DIV_EXISTING_CHILD_MAX = 10.0
"""),

    md(r"""
## 1 · Resolve each ground-truth division into predicted node ids

Same windowed matching the official scorer uses: the parent side is the divider plus its
immediate predecessor, each daughter lineage is a child plus its successors.
"""),

    code(r"""
def division_targets(pred, gt):
    gt_succ, gt_pred = D._adjacency(gt)
    out = []
    for divider in sorted(D._forks(gt_succ)):
        children = gt_succ[divider]
        if len(children) < 2:
            continue
        window = {divider, *gt_pred.get(divider, []), *children,
                  *[gc for c in children for gc in gt_succ.get(c, [])]}
        sub = D._sub_track_graph(gt, window)
        p2g = M.match_nodes(pred, sub, 7.0)
        parents = {p for p, g in p2g.items() if g in {divider, *gt_pred.get(divider, [])}}
        lineages = [{p for p, g in p2g.items() if g in {c, *gt_succ.get(c, [])}}
                    for c in children[:2]]
        out.append(dict(divider=divider, parents=parents, lineages=lineages,
                        resolvable=bool(parents) and all(lineages)))
    return out

def classify(pred, targets, require_mutual_nn=True):
    # For each GT division, decide whether the prediction already forks there,
    # whether a completing proposal exists in the pool, and if so at what rank
    # the stock key would place it.
    succ, pred_adj = D._adjacency(pred)
    um = {int(i): p for i, p in zip(pred.ids, pred.coords() * SCALE)}
    tt = {int(i): int(t) for i, t in zip(pred.ids, pred.t)}
    by_t = {}
    for n, t in tt.items():
        by_t.setdefault(t, []).append(n)

    rows = []
    for tg in targets:
        if not tg["resolvable"]:
            rows.append(dict(divider=tg["divider"], outcome="unresolvable",
                             stock_rank=np.nan, like_rank=np.nan, pool=0))
            continue
        lin0, lin1 = tg["lineages"]

        # already a fork on the parent side?
        forked = any(len(succ.get(p, [])) >= 2 for p in tg["parents"])
        if forked:
            rows.append(dict(divider=tg["divider"], outcome="already_forked",
                             stock_rank=np.nan, like_rank=np.nan, pool=0))
            continue

        # every proposal this parent side could generate, and which complete it
        completing, pool_n = [], 0
        for source in tg["parents"]:
            kids = succ.get(source, [])
            if len(kids) != 1:
                continue
            child = kids[0]
            child_dist = float(np.linalg.norm(um[child] - um[source]))
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX:
                continue
            nxt = [n for n in by_t.get(tt[source] + 1, []) if n != child]
            if not nxt:
                continue
            d = np.array([np.linalg.norm(um[n] - um[source]) for n in nxt])
            mutual_nn = nxt[int(np.argmin(d))]
            for cand, parent_dist in zip(nxt, d):
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister = float(np.linalg.norm(um[cand] - um[child]))
                if sister > SAFE_DIV_SISTER_MAX_UM:
                    continue
                if require_mutual_nn and cand != mutual_nn:
                    continue
                pool_n += 1
                good = ((cand in lin0 and child in lin1) or
                        (cand in lin1 and child in lin0))
                if good:
                    completing.append((float(parent_dist), sister, child_dist))
        rows.append(dict(divider=tg["divider"],
                         outcome="in_pool" if completing else "pool_absent",
                         pool=pool_n,
                         best=completing[0] if completing else None))
    return rows
"""),

    md(r"""
## 2 · Run it over the held-out films

The rank a completing proposal receives under each key is computed against that film's whole
pool, because the cap is applied per frame on a sorted list - so "is it in the pool" and
"where in the queue" are different questions and only the second is a ranking problem.
"""),

    code(r"""
PRIOR = {"parent_dist": (True, 1.94503, 0.31430),
         "sister_dist": (True, 2.30300, 0.33479),
         "symmetry":    (False, 0.55760, 0.42911)}

def nll(name, v):
    is_log, mu, sd = PRIOR[name]
    v = float(v)
    if not np.isfinite(v):
        return 4.0
    if is_log:
        v = math.log(max(v, 0.1))
    return float(min(0.5 * ((v - mu) / sd) ** 2, 12.0))

def keys(parent_dist, sister, child_dist):
    stock = parent_dist + 0.15 * sister
    sym = abs(child_dist - parent_dist) / max((child_dist + parent_dist) / 2.0, 1e-6)
    like = nll("parent_dist", parent_dist) + nll("sister_dist", sister) + nll("symmetry", sym)
    return stock, like

def full_pool(pred, require_mutual_nn=True):
    # every proposal in the film, for rank computation
    succ, _ = D._adjacency(pred)
    um = {int(i): p for i, p in zip(pred.ids, pred.coords() * SCALE)}
    tt = {int(i): int(t) for i, t in zip(pred.ids, pred.t)}
    by_t = {}
    for n, t in tt.items():
        by_t.setdefault(t, []).append(n)
    out = []
    for source, kids in succ.items():
        if len(kids) != 1:
            continue
        child = kids[0]
        cd = float(np.linalg.norm(um[child] - um[source]))
        if cd > SAFE_DIV_EXISTING_CHILD_MAX:
            continue
        nxt = [n for n in by_t.get(tt[source] + 1, []) if n != child]
        if not nxt:
            continue
        d = np.array([np.linalg.norm(um[n] - um[source]) for n in nxt])
        mnn = nxt[int(np.argmin(d))]
        for cand, pdist in zip(nxt, d):
            if pdist > SAFE_DIV_MAX_UM:
                continue
            s = float(np.linalg.norm(um[cand] - um[child]))
            if s > SAFE_DIV_SISTER_MAX_UM:
                continue
            if require_mutual_nn and cand != mnn:
                continue
            out.append(keys(float(pdist), s, cd))
    return np.array(out) if out else np.zeros((0, 2))

records, failed = [], []
for geff in sorted(PRED_GEFFS):
    stem = geff.name[:-5]
    gt_path = TRAIN / f"{stem}.geff"
    if not gt_path.exists():
        continue
    try:
        pred = B.read_geff(geff)
        gt = B.read_geff(gt_path)
    except Exception as e:
        failed.append((stem, repr(e)[:90])); continue

    targets = division_targets(pred, gt)
    if not targets:
        continue
    rows = classify(pred, targets)
    pool = full_pool(pred)
    for r in rows:
        rec = dict(film=stem, embryo=stem.split("_")[0], divider=r["divider"],
                   outcome=r["outcome"], pool=r.get("pool", 0),
                   film_pool=len(pool))
        if r["outcome"] == "in_pool" and len(pool):
            st, lk = keys(*r["best"])
            rec["stock_rank"] = int((pool[:, 0] < st).sum())
            rec["like_rank"] = int((pool[:, 1] < lk).sum())
        records.append(rec)

print(f"films scored {len(set(r['film'] for r in records))}   failed {len(failed)}")
for s, e in failed[:5]:
    print("   FAILED", s, e)
R = pd.DataFrame(records)
print(f"\nGT divisions examined: {len(R)}")
print(R.outcome.value_counts().to_string())
"""),

    md(r"""
## 3 · The number that sizes the prize
"""),

    code(r"""
n = len(R)
counts = R.outcome.value_counts()
print(f"{'outcome':<18} {'n':>5} {'share':>8}   what it means")
for k, label in [("already_forked", "already a TP"),
                 ("in_pool", "RANKING can fix"),
                 ("pool_absent", "generator must change"),
                 ("unresolvable", "detection missed it")]:
    c = int(counts.get(k, 0))
    print(f"{k:<18} {c:>5} {c/max(n,1):>7.1%}   {label}")

inp = R[R.outcome == "in_pool"]
if len(inp):
    print(f"\nof the {len(inp)} that ranking could fix, where the completing proposal sits:")
    print(f"{'':<12} {'stock key':>12} {'likelihood':>12}")
    for cap in (5, 10, 25, 50, 100, 500):
        s = int((inp.stock_rank < cap).sum())
        l = int((inp.like_rank < cap).sum())
        print(f"  within top-{cap:<4} {s:>10} {l:>12}")
    print(f"\nmedian rank   stock {inp.stock_rank.median():.0f}   "
          f"likelihood {inp.like_rank.median():.0f}   "
          f"(pool median {R.film_pool.median():.0f})")

print("\nper embryo:")
print(R.groupby(["embryo", "outcome"]).size().unstack(fill_value=0).to_string())

R.to_csv("/kaggle/working/pool_split.csv", index=False)
summary = dict(divisions=int(n), outcomes={k: int(v) for k, v in counts.items()})
if len(inp):
    summary["rank_within_top25"] = dict(stock=int((inp.stock_rank < 25).sum()),
                                        likelihood=int((inp.like_rank < 25).sum()))
json.dump(summary, open("/kaggle/working/pool_split.json", "w"), indent=2)
print("\n" + json.dumps(summary, indent=2))
"""),

    md(r"""
## Reading it

* **`in_pool` large and the likelihood key pulls proposals inside the top-25 where the stock
  key does not**  the sign fix is the whole game and `15_div_rank` should move the score.
* **`pool_absent` dominates**  the candidate generator is the constraint, not the ranker.
  First suspect is the mutual-nearest-neighbour requirement, which forces the proposed
  daughter to be the closest node to the parent - and a real daughter sits ~7 µm away with
  duplicate detections nearer. Re-run with `require_mutual_nn=False` before concluding.
* **`unresolvable` large**  the detector is not finding the daughters and neither fix applies.

Report per embryo and reject anything that appears in only one: the hidden test is a third
embryo, and `6bba` carries 125 of the 151 divisions.
"""),
]

if __name__ == "__main__":
    build(REPO / "notebooks" / "17_pool_split.ipynb", CELLS)
