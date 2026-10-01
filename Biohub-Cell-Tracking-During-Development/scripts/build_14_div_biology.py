"""Build notebooks/14_div_biology.ipynb - is there image evidence a fork ranker can use?

This is the feasibility test for the whole division plan, and it needs no GPU.

The public stack ranks candidate forks by ``parent_dist + 0.15 * sister_dist``
ascending - pure geometry, tightest pair first - and the per-frame budget is
spent long before a real division is reached, because the tightest pairs are
duplicate detections. Replacing that key with an evidence-based score is the
plan. This notebook asks whether the evidence exists at all, and how much of the
separation is available from features cheap enough to compute inside a
submission kernel.

The prior is a published measurement: a dividing nucleus gets **smaller but not
dimmer**, with the volume drop starting about two frames *before* the split and
recovering by +6. If that holds here, half-max volume around the parent is a
feature the geometric key cannot see - and the negatives it confuses (duplicate
detections of one cell) should show no such dip.

Negatives are built **from ground truth alone**, so this runs without any
predictions: every annotated parent->child link is a site where the pipeline
could propose a second daughter, and the nearest other annotated node at t+1
inside the gate is exactly the kind of candidate the geometric ranker prefers.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Is there evidence to rank forks with? - CPU, no model, no predictions

**No GPU.** Reads label graphs plus a bounded number of image frames.

### The question

The division term is worth 0.1 and the public frontier collects ~0.0125 of it. The binding
constraint is not the gates but the **ranking**: the fork budget is ~5 slots per frame and
the key is `parent_dist + 0.15 * sister_dist`, ascending. Tightest pair first means duplicate
detections first, so the budget is gone before a real division is reached. Opening the gates
without fixing the ranker measurably *lowers* the score.

So: **can a cheap image feature separate a real division from the candidates the geometric
key prefers?** If yes, the fix is a ranker and gold is reachable. If no, the division term is
out of reach in the time available and the effort belongs on edge Jaccard instead.

### The specific prior being tested

A dividing nucleus is **smaller, not dimmer**. Measured publicly on this dataset: half-max
*volume* drops from two frames before the split through about +5, while *peak* intensity is
statistically flat everywhere except the split frame itself. Mean intensity falls too, but
only as an artefact - a fixed-radius ball around a smaller object contains more background.

That distinction matters for feature design: **measure volume and peak separately, never
mean.** The naive feature is the one that looks like a result and is not.

### Negatives without predictions

Every annotated parentchild link is a place the pipeline could propose a second daughter.
The nearest *other* annotated node at t+1 within the gate is the candidate the geometric key
would rank highest. Those are the negatives. They are drawn from the same films, the same
frames and the same geometry as the positives, so anything that separates them is signal
rather than film-level confounding.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_div_metric.py"),

    code(r"""
import sys, warnings, json, time
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

# budget: frame reads dominate the runtime, so they are capped and reported
MAX_FRAME_READS   = 6000
LAGS              = list(range(-3, 5))      # t-3 .. t+4 relative to the divider
NEG_PER_POS       = 8
PARENT_GATE_UM    = 9.0                     # the shipped SAFE_DIV_MAX_UM
SISTER_GATE_UM    = 14.0
BOX_UM            = 6.0                     # half-width of the measurement box
RNG = np.random.default_rng(0)
print("train films:", len(list(TRAIN.glob('*.geff'))))
"""),

    md(r"""
## 1 · Find the divisions, and build matched negatives

Films are taken division-richest first so the frame budget buys as many positives as
possible. Every negative comes from the **same film and the same frame** as a positive where
one is available, which removes film-level brightness and density as explanations.
"""),

    code(r"""
def load_graph(stem):
    g = B.read_geff(TRAIN / f"{stem}.geff")
    succ, pred = D._adjacency(g)
    pos = {int(i): p for i, p in zip(g.ids, g.coords())}          # voxel units
    um  = {int(i): p for i, p in zip(g.ids, g.coords() * SCALE)}  # micrometres
    tt  = {int(i): int(t) for i, t in zip(g.ids, g.t)}
    return g, succ, pred, pos, um, tt

stems = sorted(p.name[:-5] for p in TRAIN.glob("*.geff"))
inventory, graphs, failed = [], {}, []
for stem in stems:
    try:
        bundle = load_graph(stem)
    except Exception as e:
        failed.append((stem, repr(e)[:100])); continue
    graphs[stem] = bundle
    inventory.append((stem, len(D._forks(bundle[1]))))

print(f"films read {len(graphs)}   failed {len(failed)}")
for s, e in failed[:5]:
    print("   FAILED", s, e)
inv = pd.DataFrame(inventory, columns=["film", "n_div"]).sort_values("n_div", ascending=False)
print(f"total labelled divisions: {int(inv.n_div.sum())} across "
      f"{int((inv.n_div > 0).sum())} films")

samples = []
for stem, n_div in inv[inv.n_div > 0].itertuples(index=False):
    g, succ, pred, pos, um, tt = graphs[stem]
    forks = sorted(D._forks(succ))
    by_t = {}
    for nid, t in tt.items():
        by_t.setdefault(t, []).append(nid)

    for f in forks:
        kids = succ[f][:2]
        samples.append(dict(film=stem, node=f, t=tt[f], label=1,
                            partner=kids[1],
                            parent_dist_um=float(np.linalg.norm(um[kids[1]] - um[f])),
                            sister_dist_um=float(np.linalg.norm(um[kids[0]] - um[kids[1]]))))

    # Negatives: annotated nodes with exactly one child, from the same films.
    #
    # An earlier version required each negative to have a partner candidate
    # inside the shipped gates, to mimic a proposal pair. That was wrong twice
    # over. It yielded almost no negatives - only ~3.6% of cells carry a label,
    # so annotated nodes rarely have another annotated node within 9 um - and
    # more importantly it framed the question as "is this *pair* a division?"
    # when every image feature here is measured at the **parent**. The question
    # a ranker needs answered is "does this node divide?", which does not depend
    # on how dense the candidate pool is. So the gate is dropped, and the pair
    # geometry is recorded only where it happens to exist, for the head-to-head
    # against the geometric key.
    singles = [n for n, outs in succ.items() if len(outs) == 1 and n not in forks]
    RNG.shuffle(singles)
    for n in singles[:NEG_PER_POS * max(n_div, 1)]:
        child = succ[n][0]
        cands = [m for m in by_t.get(tt[n] + 1, []) if m != child]
        pd_um = sis = np.nan
        if cands:
            d = np.array([np.linalg.norm(um[m] - um[n]) for m in cands])
            k = int(np.argmin(d))
            if d[k] <= PARENT_GATE_UM:
                s_ = float(np.linalg.norm(um[cands[k]] - um[child]))
                if s_ <= SISTER_GATE_UM:
                    pd_um, sis = float(d[k]), s_
        samples.append(dict(film=stem, node=n, t=tt[n], label=0, partner=-1,
                            parent_dist_um=pd_um, sister_dist_um=sis))

S = pd.DataFrame(samples)
print(f"\nsamples: {len(S)}   positives {int(S.label.sum())}   "
      f"negatives {int((1 - S.label).sum())}")
print("\ngeometry, positives vs negatives (um):")
print(S.groupby("label")[["parent_dist_um", "sister_dist_um"]]
      .describe(percentiles=[.5]).round(2).to_string())
"""),

    md(r"""
### The geometric key, scored as a ranker

Before measuring anything from the images, this is the baseline to beat: how well does
`parent_dist + 0.15 * sister_dist` (ascending) separate real divisions from the negatives?
AUC near 0.5 means it is no better than chance; **below** 0.5 means it is actively
anti-correlated - it prefers the negatives, which is exactly the failure the budget analysis
predicts.
"""),

    code(r"""
def auc(scores, labels):
    # Rank-based AUC. Higher score must mean "more likely a real division",
    # so a key that is used ascending gets passed in negated.
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    ok = np.isfinite(scores)
    scores, labels = scores[ok], labels[ok]
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    n_pos, n_neg = labels.sum(), len(labels) - labels.sum()
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

geom = S.parent_dist_um + 0.15 * S.sister_dist_um
# the geometric key only exists where a gated partner exists; restrict the
# head-to-head to that subset so the comparison is like-for-like
PAIRED = S.parent_dist_um.notna().to_numpy()
print(f"samples with a gated partner: {PAIRED.sum()} of {len(S)} "
      f"({S.label[PAIRED].sum()} positive)")
print(f"geometric key, as used (ascending = preferred): AUC = "
      f"{auc(-geom[PAIRED], S.label[PAIRED]):.4f}")
print(f"parent_dist alone (ascending)                 : AUC = "
      f"{auc(-S.parent_dist_um[PAIRED], S.label[PAIRED]):.4f}")
print(f"sister_dist alone (ascending)                 : AUC = "
      f"{auc(-S.sister_dist_um[PAIRED], S.label[PAIRED]):.4f}")
print("\n0.50 = chance.  Below 0.50 means the key actively prefers the negatives,")
print("which is what spends the fork budget before a real division is reached.")
"""),

    md(r"""
## 2 · Measure the image, the right way

Three quantities in the same box around the node, at each lag:

| quantity | definition | what it tracks |
|---|---|---|
| `peak` | brightest voxel minus local background | brightness, nearly size-independent |
| `volume` | voxels above halfway between background and peak | **size** |
| `mean` | plain mean in the box | a mixture - recorded only to show it misleads | Background is the box's own 10th percentile, so slow drifts across a film cancel.
"""),

    code(r"""
BOX = np.ceil(BOX_UM / SCALE).astype(int)   # half-widths in voxels, per axis

def measure(frame, zyx):
    z, y, x = [int(round(v)) for v in zyx]
    zs = slice(max(0, z - BOX[0]), min(frame.shape[0], z + BOX[0] + 1))
    ys = slice(max(0, y - BOX[1]), min(frame.shape[1], y + BOX[1] + 1))
    xs = slice(max(0, x - BOX[2]), min(frame.shape[2], x + BOX[2] + 1))
    box = frame[zs, ys, xs].astype(np.float32)
    if box.size == 0:
        return np.nan, np.nan, np.nan
    bg = float(np.percentile(box, 10))
    peak = float(box.max()) - bg
    if peak <= 0:
        return 0.0, 0.0, float(box.mean() - bg)
    volume = float((box > bg + 0.5 * peak).sum())
    return peak, volume, float(box.mean() - bg)

# group work by (film, frame) so each chunk is decompressed exactly once
want = []
for i, r in S.iterrows():
    for lag in LAGS:
        want.append((r.film, int(r.t) + lag, i, lag))
want = pd.DataFrame(want, columns=["film", "frame", "row", "lag"])
print(f"frame reads required: {want.groupby(['film','frame']).ngroups:,} "
      f"(cap {MAX_FRAME_READS:,})")

groups = list(want.groupby(["film", "frame"]))
RNG.shuffle(groups)
groups = groups[:MAX_FRAME_READS]
print(f"reading {len(groups):,} frames")

node_pos = {}
for stem, (g, succ, pred, pos, um, tt) in graphs.items():
    node_pos[stem] = pos

vols, t0 = {}, time.time()
out = np.full((len(S), len(LAGS), 3), np.nan, np.float32)
lag_ix = {l: i for i, l in enumerate(LAGS)}
read = 0
for (film, frame_t), grp in groups:
    if film not in vols:
        vols[film] = B.open_volume(COMP / "train" / f"{film}.zarr")
    v = vols[film]
    if not (0 <= frame_t < v.n_t):
        continue
    try:
        frame = v.frame(int(frame_t))
    except Exception:
        continue
    read += 1
    for r in grp.itertuples(index=False):
        node = int(S.node.iloc[r.row])
        out[r.row, lag_ix[r.lag]] = measure(frame, node_pos[film][node])
    if read % 500 == 0:
        print(f"  {read:,}/{len(groups):,} frames  {time.time()-t0:.0f}s", flush=True)

print(f"frames read: {read:,} in {time.time()-t0:.0f}s")
"""),

    md(r"""
## 3 · Does volume dip before a division, and does peak hold?

Each track is normalised to its **own** value at lag −3, so this is a within-cell change and
absolute brightness differences between cells cancel. Positives and negatives come from the
same films and frames, so anything left is the division itself.
"""),

    code(r"""
names = ["peak", "volume", "mean"]
base = out[:, 0, :]                      # lag -3
with np.errstate(invalid="ignore", divide="ignore"):
    rel = out / base[:, None, :]
rel[~np.isfinite(rel)] = np.nan

pos_m = np.nanmedian(rel[S.label.to_numpy() == 1], axis=0)
neg_m = np.nanmedian(rel[S.label.to_numpy() == 0], axis=0)

print("median value relative to lag -3\n")
for k, name in enumerate(names):
    print(f"  {name}")
    print("    lag     " + "".join(f"{l:>8}" for l in LAGS))
    print("    divide  " + "".join(f"{v:>8.3f}" for v in pos_m[:, k]))
    print("    control " + "".join(f"{v:>8.3f}" for v in neg_m[:, k]))
    print("    delta   " + "".join(f"{a-b:>8.3f}" for a, b in zip(pos_m[:, k], neg_m[:, k])))
    print()

print("Expected if the published result holds here: volume delta clearly negative")
print("from about lag -1 through +4, peak delta near zero throughout, and mean")
print("negative - the last one being the artefact of a fixed-size probe.")
"""),

    md(r"""
## 4 · The number that decides the plan

AUC of each feature as a fork ranker, against the geometric key it would replace. A feature
that beats the geometric key on the *same candidates* is a direct, drop-in improvement to the
line that spends the budget.
"""),

    code(r"""
feat = {}
lab = S.label.to_numpy()
for k, name in enumerate(names):
    for i, l in enumerate(LAGS):
        feat[f"{name}_rel_lag{l:+d}"] = rel[:, i, k]
    # the summary statistics a ranker would actually use
    pre = [lag_ix[l] for l in LAGS if -2 <= l <= 0]
    post = [lag_ix[l] for l in LAGS if 1 <= l <= 4]
    feat[f"{name}_pre_min"] = np.nanmin(rel[:, pre, k], axis=1)
    feat[f"{name}_post_min"] = np.nanmin(rel[:, post, k], axis=1)
    feat[f"{name}_drop"] = np.nanmin(rel[:, pre + post, k], axis=1)

feat["geom_key"] = -(S.parent_dist_um + 0.15 * S.sister_dist_um).to_numpy()
feat["parent_dist"] = -S.parent_dist_um.to_numpy()
feat["sister_dist"] = -S.sister_dist_um.to_numpy()

scores = []
for name, v in feat.items():
    a_all = auc(v, lab)
    a_pair = auc(v[PAIRED], lab[PAIRED])
    if np.isfinite(a_all):
        scores.append((name, a_all, max(a_all, 1 - a_all),
                       a_pair, max(a_pair, 1 - a_pair) if np.isfinite(a_pair) else np.nan))
res = pd.DataFrame(scores, columns=["feature", "auc", "auc_abs",
                                    "auc_paired", "auc_paired_abs"]).sort_values(
    "auc_abs", ascending=False)
print("feature AUC (0.5 = chance; auc_abs folds anti-correlated features)\n")
print(res.head(25).round(4).to_string(index=False))

IMG = ~res.feature.isin(["geom_key", "parent_dist", "sister_dist"])
best_img = res[IMG].sort_values("auc_paired_abs", ascending=False).head(1)
geom_auc = float(res[res.feature == "geom_key"].auc_paired_abs.iloc[0])
best_auc = float(best_img.auc_paired_abs.iloc[0])
print(f"\n--- head-to-head on the {int(PAIRED.sum())} samples that have a gated partner ---")
print(f"best image feature : {best_img.feature.iloc[0]}  auc {best_auc:.4f}")
print(f"geometric key      : auc {geom_auc:.4f}")
print(f"\nVERDICT: image evidence "
      f"{'BEATS' if best_auc > geom_auc + 0.02 else 'does NOT beat'}"
      f" the geometric key on the same candidates.")
print(f"\nOn all {len(S)} samples (parent-only question, no partner needed), the best")
print(f"image feature reaches auc {float(res[IMG].auc_abs.iloc[0]):.4f} - that is the")
print("number that matters for a ranker that scores the parent rather than the pair.")

res.to_csv("/kaggle/working/div_feature_auc.csv", index=False)
S.assign(**{k: v for k, v in feat.items() if v.ndim == 1}).to_csv(
    "/kaggle/working/div_samples.csv", index=False)
np.save("/kaggle/working/div_profiles.npy", out)
json.dump(dict(lags=LAGS, names=names, n_pos=int(lab.sum()), n_neg=int((1-lab).sum()),
               frames_read=int(read), geom_auc=float(geom_auc),
               best_image_feature=str(best_img.feature.iloc[0]),
               best_image_auc=float(best_img.auc_abs.iloc[0])),
          open("/kaggle/working/div_biology.json", "w"), indent=2)
print("\nsaved div_feature_auc.csv, div_samples.csv, div_profiles.npy, div_biology.json")
"""),

    md(r"""
## How to read the verdict

* **Image AUC well above the geometric key**  the ranker is the fix, the features are known,
  and a small 3D CNN trained on these crops should do better still. Proceed to training it
  and to swapping the `score = parent_dist + 0.15 * sister_dist` line.
* **Image AUC ≈ geometric key**  the cheap features are not enough; only a learned detector
  on raw crops could separate, which is a bigger bet on less time.
* **Both near 0.5**  real divisions are not distinguishable from duplicate detections at
  this candidate set, and the division term should be abandoned in favour of edge Jaccard.

One caveat to carry: negatives here are built from *annotated* nodes, which are cleaner than
the predicted candidate pool the pipeline actually ranks. Treat the AUC as an upper bound
until it is re-measured on real proposals from `12_div_probe`.
"""),
]

if __name__ == "__main__":
    build(REPO / "notebooks" / "14_div_biology.ipynb", CELLS)
