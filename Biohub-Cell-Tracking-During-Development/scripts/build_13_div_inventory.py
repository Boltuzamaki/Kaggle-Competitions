"""Build notebooks/13_div_inventory.ipynb - what the 151 real divisions look like.

CPU only, graph reads only, a few minutes. It answers the questions that set
every parameter of a division ranker, and that no public notebook has published:

1. Where are the divisions? (per embryo, per film, how concentrated)
2. What does a real division's *geometry* look like - parent step, sister
   separation, divergence, symmetry - and where do the shipped gates sit
   relative to those distributions?
3. What is the **ceiling**? Under the union-FP rule, what division Jaccard is
   reachable at a given precision, and how many forks can we afford to propose?

(3) is the one that decides the plan. The score is
``adj_edge_jaccard + 0.1 * division_jaccard``; gold needs division Jaccard ~0.20
against the ~0.125 the public stack gets. Whether that is a tuning problem or a
modelling problem depends entirely on the shape of the precision/recall curve
available to a *perfect* geometric ranker, which is what this measures.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Division inventory - all 199 training films, CPU, no images

**No GPU. No model. Graph arrays only (a few hundred bytes per film).**

### Why this notebook exists

The competition score is

$$\text{score} = \text{adj\_edge\_jaccard} + 0.1 \times \text{division\_jaccard}$$

and the public frontier is collecting almost none of the second term. Decomposing a
held-out run of the widely-forked 0.947 stack gives $0.9383 + 0.1 \times 0.125$ - so
**0.0125 of a possible 0.100**. The medal arithmetic, holding edge Jaccard fixed:

| target | division Jaccard needed |
|---|---|
| 0.958 (gold) | 0.197 |
| 0.970 (current LB leader) | 0.317 | Two things have to be true before chasing that is sensible, and both are measurable here
without a GPU:

* the divisions have to be **reachable** - enough of them, with geometry a ranker can see;
* the **false-positive budget** has to allow it, because under the official rule FPs are
  *unioned* across several categories, not summed.

### The metric detail that shapes everything

The organizers patched the division metric on **2026-07-17** (`aa65e90`). A predicted fork
is a false positive only if it is `considered` (local to some GT division), `evaluable`
(matched onto an *annotated* GT node that has a child), or `invalid`. **A spurious fork on an
unannotated cell is invisible.** Only ~3.6% of cells carry a label, so the effective FP rate
of a wide candidate pool is far lower than the shipped gates assume - which is the first
hint that gating is the wrong lever.

This notebook uses `biohub_div_metric`, our port of the **patched** scorer, verified against
`tracking_cellmot.division_metrics` on ten synthetic graphs (exact TP/FN/FP agreement).
The public stack's own validator still uses the **pre-patch** rule and reads about 2× high.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_div_metric.py"),

    code(r"""
import sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import biohub_ct as B
import biohub_div_metric as D

COMP = next(p for p in [Path("/kaggle/input/biohub-cell-tracking-during-development"),
                        Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")]
            if p.exists())
TRAIN = COMP / "train"
SCALE = B.SCALE
print("train films:", len(list(TRAIN.glob("*.geff"))))
"""),

    md(r"""
## 1 · Read every label graph

`zarr` is not installed on Kaggle, so `.geff` arrays are read straight from their
zstd chunks - the same reader the rest of this repo uses. Any film that fails to read is
**counted and reported**, never silently skipped: an earlier session lost a whole training
run to a bare `except: continue` over this exact directory.
"""),

    code(r"""
rows, div_rows, failed = [], [], []

for geff in sorted(TRAIN.glob("*.geff")):
    stem = geff.name[:-5]
    try:
        g = B.read_geff(geff)
    except Exception as e:
        failed.append((stem, repr(e)[:120]))
        continue

    succ, pred = D._adjacency(g)
    pos = {int(i): p for i, p in zip(g.ids, g.coords() * SCALE)}
    tt = {int(i): int(t) for i, t in zip(g.ids, g.t)}
    forks = D._forks(succ)

    rows.append(dict(
        film=stem, embryo=stem.split("_")[0], n_nodes=g.n_nodes, n_edges=g.n_edges,
        n_div=len(forks),
        est_nodes=float(g.meta.get("estimated_number_of_nodes", np.nan)),
    ))

    for f in forks:
        kids = succ[f][:2]
        parent_steps = [float(np.linalg.norm(pos[k] - pos[f])) for k in kids]
        sister = float(np.linalg.norm(pos[kids[0]] - pos[kids[1]]))
        # divergence: how much further apart the grandchildren are, one frame on
        gk = [succ.get(k, []) for k in kids]
        diverge = np.nan
        if all(len(x) >= 1 for x in gk):
            diverge = float(np.linalg.norm(pos[gk[0][0]] - pos[gk[1][0]])) - sister
        sym = abs(parent_steps[0] - parent_steps[1]) / max(np.mean(parent_steps), 1e-6)
        div_rows.append(dict(
            film=stem, embryo=stem.split("_")[0], divider=int(f), t=tt[f],
            parent_dist_um=max(parent_steps), parent_dist_min_um=min(parent_steps),
            sister_dist_um=sister, diverge_um=diverge, symmetry=sym,
            has_grandchildren=all(len(x) >= 1 for x in gk),
            has_parent=len(pred.get(f, [])) > 0,
        ))

films = pd.DataFrame(rows)
divs = pd.DataFrame(div_rows)
print(f"films read     : {len(films)}   failed: {len(failed)}")
for s, e in failed:
    print("   FAILED", s, e)
assert len(films) > 0, "no training graphs read - check the reader, do not proceed"
print(f"labelled nodes : {films.n_nodes.sum():,}")
print(f"divisions      : {len(divs)}")
print(divs.groupby('embryo').size().to_string())
"""),

    md(r"""
## 2 · How concentrated are they?

This decides how a held-out split must be built. If divisions sit in a handful of films, a
random split measures division Jaccard on almost nothing and every threshold tuned against it
is noise - which is the same trap that made three of session 3's n=2 "improvements" evaporate.
"""),

    code(r"""
per_film = films.set_index("film").n_div
print("films with >=1 division:", int((per_film > 0).sum()), "of", len(per_film))
print("\ndivisions per film:")
print(per_film.value_counts().sort_index().to_string())

print("\nlabel density by embryo (labelled nodes / estimated true nodes):")
films["label_frac"] = films.n_nodes / films.est_nodes
print(films.groupby("embryo").label_frac.describe()[["count", "50%", "min", "max"]].to_string())

# how many films are needed to cover most divisions, taking the richest first
order = per_film.sort_values(ascending=False)
cum = order.cumsum() / max(order.sum(), 1)
for frac in (0.5, 0.8, 0.9):
    print(f"  {frac:.0%} of divisions live in the richest {int((cum < frac).sum()) + 1} films")
"""),

    md(r"""
## 3 · The geometry of a real division, against the shipped gates

The public stack proposes a fork when an existing parentchild link can take a *second*
child, subject to four geometric gates. Their production values:

| gate | shipped value |
|---|---|
| `SAFE_DIV_MAX_UM` (parent  new daughter) | 9.0 |
| `SAFE_DIV_SISTER_MAX_UM` (daughter  daughter) | 14.0 |
| `SAFE_DIV_DIVERGE_UM` (grandchildren must separate by this much more) | 2.25 |
| `SAFE_DIV_SISTER_SYMMETRY_TAU` (relative asymmetry of the two parent steps) | 0.6 | Here is where each of those sits in the distribution of **real** divisions. A gate at the
median of the true distribution is a coin flip, not a precision filter.
"""),

    code(r"""
GATES = {
    "parent_dist_um":  ("SAFE_DIV_MAX_UM",                 9.0,  "le"),
    "sister_dist_um":  ("SAFE_DIV_SISTER_MAX_UM",          14.0, "le"),
    "diverge_um":      ("SAFE_DIV_DIVERGE_UM",             2.25, "ge"),
    "symmetry":        ("SAFE_DIV_SISTER_SYMMETRY_TAU",    0.6,  "le"),
}

print(f"{'quantity':<18} {'gate':<30} {'value':>7} {'pass':>7} {'pctile of real':>15}")
for col, (name, val, sense) in GATES.items():
    s = divs[col].dropna()
    passes = (s <= val) if sense == "le" else (s >= val)
    pct = (s < val).mean() * 100
    print(f"{col:<18} {name:<30} {val:>7.2f} {passes.mean():>6.1%} {pct:>14.1f}%")

print("\ndistribution of real divisions (um):")
print(divs[["parent_dist_um", "sister_dist_um", "diverge_um", "symmetry"]]
      .describe(percentiles=[.1, .25, .5, .75, .9]).round(3).to_string())
"""),

    md(r"""
### Cumulative gate survival

Gates are applied in sequence, so what matters is how many real divisions are still alive at
the end - the size of the pool a ranker is allowed to rank. If that number is small, no
ranker can help and the gates must move first.
"""),

    code(r"""
alive = pd.Series(True, index=divs.index)
print(f"{'after gate':<34} {'divisions still reachable':>26}")
print(f"{'(all labelled divisions)':<34} {len(divs):>26}")
for col, (name, val, sense) in GATES.items():
    ok = (divs[col] <= val) if sense == "le" else (divs[col] >= val)
    alive &= ok.fillna(False)
    print(f"{name:<34} {int(alive.sum()):>26}")

print(f"\nreachable fraction: {alive.mean():.1%}")
print("\nreachable by embryo:")
print(divs.assign(alive=alive).groupby("embryo").alive.agg(["sum", "size", "mean"]).to_string())
"""),

    md(r"""
## 4 · The ceiling - what division Jaccard is buyable, and at what FP cost

Division Jaccard is $TP/(TP+FP+FN)$ over *all* films pooled. With $D$ labelled divisions,
proposing forks that recover a fraction $r$ of them at precision $p$ gives

$$J = \frac{rD}{rD + rD(1-p)/p + (1-r)D} = \frac{r}{1 + r(1-p)/p}$$

so the FP count is what binds. The table below reads directly as a specification for the
ranker: to hit a target $J$, this is the recall/precision pair required.
"""),

    code(r"""
D_total = len(divs)
targets = [0.125, 0.20, 0.30, 0.40]
recalls = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

print(f"division Jaccard as a function of (recall, precision), D = {D_total}\n")
print(f"{'recall':>7} " + "".join(f"{p:>9.0%}" for p in [0.2, 0.3, 0.5, 0.7, 0.9]))
for r in recalls:
    cells = []
    for p in [0.2, 0.3, 0.5, 0.7, 0.9]:
        j = r / (1 + r * (1 - p) / p)
        cells.append(f"{j:>9.3f}")
    print(f"{r:>7.0%} " + "".join(cells))

print("\nwhat each medal target demands:")
print(f"{'target J':>9} {'score at edge 0.9383':>21}   precision needed at each recall")
for j in targets:
    pairs = []
    for r in recalls:
        # J = r / (1 + r(1-p)/p)  =>  p = rJ / (rJ + r - J), valid only for r >= J
        if r < j:
            continue
        p = (r * j) / (r * j + r - j)
        pairs.append(f"r={r:.0%}->p={p:.0%}")
    print(f"{j:>9.3f} {0.9383 + 0.1 * j:>21.4f}   " + "  ".join(pairs[:5]))

print("\nSanity: recall can never be below the target Jaccard - with zero false")
print("positives J == recall, so r < J is unachievable at any precision.")
"""),

    md(r"""
## 5 · The FP budget the metric actually charges

The union rule means the FP count is **not** the number of forks proposed. A fork is charged
only when it is `considered` (local to a GT division window), `evaluable` (sits on an
annotated GT node with a child), or `invalid`. Everything else is free.

So the practical question is: of the nodes a fork could be proposed on, what fraction are
even visible to the metric? That is the label density, and it differs by a factor of ~12
between the two embryos - which means the FP cost of an identical policy is ~12× higher on
one embryo than the other, and the hidden test is a **third, unseen** embryo.
"""),

    code(r"""
print("labelled fraction of all cells, by embryo:")
print(films.groupby("embryo").label_frac.median().round(4).to_string())

visible = films.label_frac.median()
print(f"\nmedian labelled fraction overall: {visible:.3f}")
print(f"=> roughly {1/visible:.0f} forks can be proposed per one that the metric can charge,")
print("   assuming proposals land on cells at random. Real proposals are not random - they")
print("   cluster on the parent side of tracks, which is where annotations also are - so")
print("   treat this as a loose upper bound on how cheap a wide pool is.")

summary = dict(
    films=len(films), divisions=int(D_total),
    divisions_per_embryo=divs.groupby("embryo").size().to_dict(),
    reachable_after_shipped_gates=int(alive.sum()),
    reachable_fraction=float(alive.mean()),
    label_fraction_by_embryo=films.groupby("embryo").label_frac.median().round(4).to_dict(),
)
Path("/kaggle/working/division_inventory.json").write_text(json.dumps(summary, indent=2, default=float))
divs.to_csv("/kaggle/working/divisions.csv", index=False)
films.to_csv("/kaggle/working/films.csv", index=False)
print("\n" + json.dumps(summary, indent=2, default=float))
"""),

    md(r"""
## What to read off this

* **Reachability after the shipped gates** is the headline. If it is well under half, the
  gates have to open before any ranker matters - and opening them is only safe because the
  metric cannot see forks on unannotated cells.
* **Where each gate sits in the real distribution** says which one to move first. A gate at
  the median costs half the divisions for no precision.
* **The recall/precision table** is the specification. It says whether gold is a tuning
  problem (achievable at low precision) or a modelling problem (needs a real classifier).
* **The embryo asymmetry in label density** is the transfer risk: an FP policy calibrated on
  the densely-labelled embryo will be charged differently on the hidden one.
"""),
]

if __name__ == "__main__":
    build(REPO / "notebooks" / "13_div_inventory.ipynb", CELLS)
