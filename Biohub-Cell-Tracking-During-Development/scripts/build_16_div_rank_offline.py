"""Build notebooks/16_div_rank_offline.ipynb - the fork ranker on REAL candidates.

`tests/test_div_ranker.py` shows the likelihood key beats the stock key on
*simulated* duplicate detections, and simulated ones at that - drawn small on
purpose. That confirms the mechanism and proves nothing about the size of the
win. The real candidate pool contains harder negatives: genuinely separated
neighbouring cells, gap-filled synthetic nodes, and detections on cells that
simply are not dividing.

This notebook replaces the simulation with the real thing, on CPU. It attaches
`zhincez/biohub-diagnostic-dumps` - predicted node/edge graphs from a fork of
the public stack - regenerates the safe-division candidate pool from those
predictions exactly as `add_safe_divisions_postlink` does, labels each candidate
against ground truth using the **patched** division rule, and scores the two
ranking keys on the pool that is actually ranked in production.

If the likelihood key does not win here, it will not win on Kaggle either, and
this costs minutes rather than a GPU run.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# The fork ranker, on real candidates - CPU

**No GPU.** Predicted graphs come from the public `zhincez/biohub-diagnostic-dumps` dataset;
ground truth from the competition `train/` graphs.

### What this is correcting

An earlier check scored the two ranking keys against **simulated** duplicate detections and
reported AUC 0.349 (stock) against 0.999 (likelihood). The mechanism it demonstrates is real - the stock key is used *ascending* and no real division has `parent_dist` under 2 µm - but
the 0.999 is an artefact of negatives that were generated to be tight. It must not be quoted.

Here the negatives are whatever the real pipeline actually proposes.

### The candidate pool, regenerated faithfully

`add_safe_divisions_postlink` walks every predicted node that currently has exactly one
child and looks for a second daughter among the nodes at `t+1`, subject to distance gates
and (optionally) a mutual-nearest-neighbour requirement. That loop is reproduced below
against the dumped predictions, so the pool being ranked is the production pool.

### Labelling a candidate

A proposal `(source  candidate)` is **positive** when adding that edge would satisfy the
patched rule for some real GT division: the source must match the GT divider or its
immediate predecessor, and the candidate and the existing child must match **different** GT
daughter lineages. Matching uses the same per-division windowed assignment the official
scorer uses, via `biohub_div_metric`.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_div_metric.py"),

    code(r"""
import sys, warnings, json, math, itertools
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
# Kaggle mounts attached datasets under either /kaggle/input/<slug> or
# /kaggle/input/datasets/<owner>/<slug> depending on the notebook's vintage, so
# resolve rather than assume - and list what is actually there if neither hits.
_DUMP_CANDIDATES = [Path("/kaggle/input/biohub-diagnostic-dumps"),
                    Path("/kaggle/input/datasets/zhincez/biohub-diagnostic-dumps")]
DUMPS = next((p for p in _DUMP_CANDIDATES if p.exists()), None)
if DUMPS is None:
    found = sorted(str(p) for p in Path("/kaggle/input").rglob("*") if p.is_dir())[:40]
    raise SystemExit("diagnostic dumps not found. /kaggle/input contains:\n  "
                     + "\n  ".join(found))
print("dumps at:", DUMPS)
SCALE = B.SCALE

# the shipped gates
SAFE_DIV_MAX_UM             = 9.0
SAFE_DIV_SISTER_MAX_UM      = 14.0
SAFE_DIV_EXISTING_CHILD_MAX = 10.0
REQUIRE_MUTUAL_NN           = True

print("dump configurations:", sorted(p.name for p in DUMPS.iterdir() if p.is_dir()))
"""),

    md(r"""
## 1 · Load the predicted graphs and their ground truth
"""),

    code(r"""
def load_dump(cfg_dir, stem):
    nodes = pd.read_csv(cfg_dir / f"{stem}_nodes.csv")
    edges = pd.read_csv(cfg_dir / f"{stem}_edges.csv")
    return B.TrackGraph(
        t=nodes.t.to_numpy(np.int64), z=nodes.z.to_numpy(float),
        y=nodes.y.to_numpy(float), x=nodes.x.to_numpy(float),
        ids=nodes.node_id.to_numpy(np.int64),
        edges=edges[["source_id", "target_id"]].to_numpy(np.int64),
        meta={})

pairs = []
for cfg in sorted(p for p in DUMPS.iterdir() if p.is_dir()):
    for nf in sorted(cfg.glob("*_nodes.csv")):
        stem = nf.name[:-len("_nodes.csv")]
        if (cfg / f"{stem}_edges.csv").is_file() and (TRAIN / f"{stem}.geff").exists():
            pairs.append((cfg.name, cfg, stem))
print(f"{len(pairs)} (config, film) pairs available")
print(pd.Series([c for c, _, _ in pairs]).value_counts().to_string())
"""),

    md(r"""
## 2 · Regenerate the candidate pool and label it

For each GT division the parent side and the two daughter lineages are resolved with the
same windowed matching the official scorer uses. A proposal is positive only if it would
complete that division under the **patched** local-topology rule - sharing a weakly
connected component, which the pre-patch rule accepted, is not enough.
"""),

    code(r"""
def division_targets(pred, gt):
    # For each GT division, resolve the parent side and the two daughter
    # lineages into predicted node ids, using the same per-division windowed
    # matching the official scorer uses.
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
        parent_gt = {divider, *gt_pred.get(divider, [])}
        parents = {p for p, g in p2g.items() if g in parent_gt}
        lineages = [{p for p, g in p2g.items() if g in {c, *gt_succ.get(c, [])}}
                    for c in children[:2]]
        if parents and all(lineages):
            out.append((divider, parents, lineages))
    return out

def candidate_pool(pred, targets):
    # Reproduce add_safe_divisions_postlink's proposal loop over a predicted graph.
    succ, _ = D._adjacency(pred)
    um = {int(i): p for i, p in zip(pred.ids, pred.coords() * SCALE)}
    tt = {int(i): int(t) for i, t in zip(pred.ids, pred.t)}
    by_t = {}
    for n, t in tt.items():
        by_t.setdefault(t, []).append(n)

    rows = []
    for source, kids in succ.items():
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
            if REQUIRE_MUTUAL_NN and cand != mutual_nn:
                continue
            label = 0
            for _div, parents, lineages in targets:
                if source not in parents:
                    continue
                # candidate and existing child must sit on DIFFERENT GT lineages
                if ((cand in lineages[0] and child in lineages[1]) or
                        (cand in lineages[1] and child in lineages[0])):
                    label = 1
                    break
            rows.append((source, cand, float(parent_dist), sister, child_dist, label))
    return pd.DataFrame(rows, columns=["source", "candidate", "parent_dist_um",
                                       "sister_dist_um", "child_dist_um", "label"])

all_rows, failed = [], []
for cfg_name, cfg_dir, stem in pairs:
    try:
        pred = load_dump(cfg_dir, stem)
        gt = B.read_geff(TRAIN / f"{stem}.geff")
    except Exception as e:
        failed.append((cfg_name, stem, repr(e)[:90])); continue
    targets = division_targets(pred, gt)
    pool = candidate_pool(pred, targets)
    pool["config"] = cfg_name
    pool["film"] = stem
    pool["embryo"] = stem.split("_")[0]
    pool["gt_divisions"] = len(D._forks(D._adjacency(gt)[0]))
    pool["reachable_divisions"] = len(targets)
    all_rows.append(pool)

print(f"films processed {len(all_rows)}   failed {len(failed)}")
for c, s, e in failed:
    print("   FAILED", c, s, e)
P = pd.concat(all_rows, ignore_index=True)
print(f"\ncandidates: {len(P):,}   positive: {int(P.label.sum())}   "
      f"base rate: {P.label.mean():.4%}")
print("\nby film:")
print(P.groupby(["config", "film"]).agg(
    candidates=("label", "size"), positives=("label", "sum"),
    gt_div=("gt_divisions", "first"), reachable=("reachable_divisions", "first")
).to_string())
"""),

    md(r"""
## 3 · The honest head-to-head

The stock key is consumed **ascending**, so it is negated before the AUC so that "higher =
more likely a real division" holds for both. An AUC below 0.5 for the stock key means it
systematically prefers the negatives.
"""),

    code(r"""
PRIOR = {"parent_dist": (True, 1.94503, 0.31430),
         "sister_dist": (True, 2.30300, 0.33479),
         "symmetry":    (False, 0.55760, 0.42911)}

def nll_term(name, v):
    is_log, mu, sd = PRIOR[name]
    v = float(v)
    if not np.isfinite(v):
        return 4.0
    if is_log:
        v = math.log(max(v, 0.1))
    return float(min(0.5 * ((v - mu) / sd) ** 2, 12.0))

def likelihood_key(row):
    sym = abs(row.child_dist_um - row.parent_dist_um) / max(
        (row.child_dist_um + row.parent_dist_um) / 2.0, 1e-6)
    return (nll_term("parent_dist", row.parent_dist_um)
            + nll_term("sister_dist", row.sister_dist_um)
            + nll_term("symmetry", sym))

P["stock_key"] = P.parent_dist_um + 0.15 * P.sister_dist_um
P["likelihood_key"] = [likelihood_key(r) for r in P.itertuples(index=False)]

def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, int)
    if label.sum() in (0, len(label)):
        return np.nan
    r = pd.Series(score).rank(method="average").to_numpy()
    p, n = label.sum(), len(label) - label.sum()
    return (r[label == 1].sum() - p * (p + 1) / 2) / (p * n)

print(f"{'scope':<26} {'n':>8} {'pos':>5} {'stock AUC':>10} {'likelihood AUC':>15}")
def report(name, sub):
    if len(sub) == 0 or sub.label.sum() == 0:
        print(f"{name:<26} {len(sub):>8} {int(sub.label.sum()):>5}"
              f"{'  (no positives)':>26}")
        return
    print(f"{name:<26} {len(sub):>8} {int(sub.label.sum()):>5} "
          f"{auc(-sub.stock_key, sub.label):>10.4f} {auc(-sub.likelihood_key, sub.label):>15.4f}")

report("ALL", P)
for emb, sub in P.groupby("embryo"):
    report(f"  embryo {emb}", sub)
for cfg, sub in P.groupby("config"):
    report(f"  config {cfg}", sub)
print("\n0.5 = chance. Stock below 0.5 means it prefers the negatives.")
"""),

    md(r"""
### Precision at the budget

AUC is the wrong summary for this problem: only the first few slots per frame are ever
spent. What matters is how many real divisions sit inside the budget. `SAFE_DIV_FRAME_FRAC_CAP
= 0.0076` against a few hundred nodes per frame is ~2-5 slots.
"""),

    code(r"""
print(f"{'budget (top-N per film)':<26} {'stock TP':>9} {'likelihood TP':>14} {'available':>10}")
for cap in (5, 10, 25, 50, 100, 250):
    stock_tp = lik_tp = 0
    for (_cfg, _film), sub in P.groupby(["config", "film"]):
        stock_tp += int(sub.nsmallest(cap, "stock_key").label.sum())
        lik_tp += int(sub.nsmallest(cap, "likelihood_key").label.sum())
    print(f"top-{cap:<22} {stock_tp:>9} {lik_tp:>14} {int(P.label.sum()):>10}")

print("\nThe 'available' column is every positive in the pool - the ceiling a perfect")
print("ranker would reach at an unlimited budget. The gap between it and the stock")
print("column is what the wrong sign is costing.")

P.to_csv("/kaggle/working/div_candidates.csv", index=False)
summary = dict(
    candidates=int(len(P)), positives=int(P.label.sum()),
    base_rate=float(P.label.mean()),
    auc_stock=float(auc(-P.stock_key, P.label)),
    auc_likelihood=float(auc(-P.likelihood_key, P.label)),
    by_embryo={e: dict(n=int(len(s)), pos=int(s.label.sum()),
                       auc_stock=float(auc(-s.stock_key, s.label)),
                       auc_likelihood=float(auc(-s.likelihood_key, s.label)))
               for e, s in P.groupby("embryo")},
)
json.dump(summary, open("/kaggle/working/div_rank_offline.json", "w"), indent=2, default=float)
print("\n" + json.dumps(summary, indent=2, default=float))
"""),

    md(r"""
## Reading it

* **Stock AUC below 0.5, likelihood clearly above**  the diagnosis holds on real candidates
  and `15_div_rank` should move the score. The size of the AUC gap is the honest number to
  quote from now on, replacing the simulated 0.999.
* **Both near 0.5**  geometry alone cannot rank this pool, and the ranker needs image
  evidence (`14_div_biology`) or a learned model rather than a fitted prior.
* **Split by embryo**  if the win appears in only one, reject it. The hidden test is a third
  embryo, and a one-embryo gain is an imaging artefact - the rule that correctly killed the
  ITEC repair in session 4.

One limitation to state: these dumps come from *a* fork of the public stack, not from our own
`12_div_probe` run, so the candidate pool is close to but not identical to the one our
submission will rank. The direction should transfer; the exact numbers need not.
"""),
]

if __name__ == "__main__":
    build(REPO / "notebooks" / "16_div_rank_offline.ipynb", CELLS)
