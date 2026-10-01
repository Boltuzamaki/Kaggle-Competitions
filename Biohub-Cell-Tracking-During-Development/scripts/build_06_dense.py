"""Build notebooks/06_dense_rescue.ipynb - attack the videos that actually lose the score.

The 60-video CV showed the aggregate is dominated by a minority of hard videos:
20% score below 0.6, they are the dense ones, and lifting the worst 10 to the
median would add ~0.055 - an order of magnitude more than any hyperparameter
found so far.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Dense-video rescue - where the score is actually lost

**CPU only.**

### The evidence

From `notebooks/05_full_cv.ipynb` over 60 training videos:

| density quartile | cells/frame | mean spacing | node recall | adj |
|---|---|---|---|---|
| sparse | 55 | 27.4 µm | 0.947 | **0.884** |
| med-low | 106 | 22.0 µm | 0.848 | 0.696 |
| med-high | 223 | 17.2 µm | 0.904 | 0.712 |
| dense | 521 | 12.9 µm | 0.844 | **0.656** | Per-video correlations: node recall **+0.900**, density **−0.392**, node-count error −0.051.
So the count penalty barely matters in practice, and the score is essentially "how well did
detection do", which in turn is "how dense is this video".

Global hyperparameter sweeps cannot fix this - every setting is a compromise across a 10×
density range. **This notebook asks a different question: do dense videos want different
parameters from sparse ones?** If yes, a per-video rule keyed on an *observable* quantity
(peaks per frame, available at test time) is worth far more than any global tweak.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),

    code(r"""
import sys, time, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import biohub_ct as B
import biohub_metric as M

warnings.filterwarnings("ignore")

COMP = B.find_comp_dir()
TRAIN = COMP / "train"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")

names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").is_dir())
print(f"{len(names)} training videos")
"""),

    md(r"""
## Pick the videos by true density

`estimated_number_of_nodes` is ground-truth metadata, so it is used **only to choose the
evaluation set** - never inside the pipeline. The per-video rule tested later keys on the
*observed* peak count, which exists at test time.
"""),

    code(r"""
rows = []
for n in names:
    try:
        g = B.read_geff(TRAIN / f"{n}.geff")
        est = g.meta.get("estimated_number_of_nodes")
        if est:
            rows.append({"dataset": n, "embryo": B.embryo_of(n), "density": float(est) / 100.0})
    except Exception:
        pass
meta = pd.DataFrame(rows).sort_values("density")
print(f"{len(meta)} videos with density metadata")
print(meta["density"].describe().round(1).to_string())

N_PER_GROUP = 12
dense  = meta.tail(N_PER_GROUP)["dataset"].tolist()
sparse = meta.head(N_PER_GROUP)["dataset"].tolist()
mid    = meta.iloc[len(meta)//2 - N_PER_GROUP//2 : len(meta)//2 + N_PER_GROUP//2]["dataset"].tolist()

print(f"\ndense  ({N_PER_GROUP}): median {meta.tail(N_PER_GROUP)['density'].median():.0f} cells/frame")
print(f"mid    ({len(mid)}): median {meta.iloc[len(meta)//2-N_PER_GROUP//2:len(meta)//2+N_PER_GROUP//2]['density'].median():.0f}")
print(f"sparse ({N_PER_GROUP}): median {meta.head(N_PER_GROUP)['density'].median():.0f}")
"""),

    code(r"""
def score_group(cfg, subset, label=""):
    rows = []
    t0 = time.time()
    for name in subset:
        try:
            vol = B.open_volume(TRAIN / f"{name}.zarr")
            gt = B.read_geff(TRAIN / f"{name}.geff")
        except Exception:
            continue
        frames = B.detect_frames(vol, cfg)
        g = B.build_graph(frames, cfg)
        rows.append(M.evaluate(g, gt, n_total=gt.meta.get("estimated_number_of_nodes")))
    s = M.summarise(rows)
    print(f"  {label:<28} adj={s['adj_edge_jaccard']:.4f}  J={s['edge_jaccard']:.4f}  "
          f"recall={s['node_recall']:.3f}  ({time.time()-t0:.0f}s)")
    return s
"""),

    md(r"""
## Do dense and sparse videos want different parameters?

The same grid is run on each group. If the argmax differs between groups, a density-conditioned
rule is justified; if the same setting wins everywhere, it is not, and this line of attack is
dead - which is worth knowing just as definitively.
"""),

    code(r"""
GRID = [
    ("baseline",            dict()),
    ("min_dist 2.5",        dict(min_distance_um=2.5)),
    ("min_dist 3.0",        dict(min_distance_um=3.0)),
    ("min_dist 4.5",        dict(min_distance_um=4.5)),
    ("min_dist 5.5",        dict(min_distance_um=5.5)),
    ("scales small",        dict(dog_scales=((1.0, 3.0), (1.75, 5.25)))),
    ("scales large",        dict(dog_scales=((2.5, 7.5), (4.0, 12.0)))),
    ("rel 0.005",           dict(rel_threshold=0.005)),
    ("rel 0.05",            dict(rel_threshold=0.05)),
    ("link 5 µm",           dict(max_link_um=5.0)),
    ("link 9 µm",           dict(max_link_um=9.0)),
]

results = {}
for gname, subset in [("DENSE", dense), ("MID", mid), ("SPARSE", sparse)]:
    print(f"\n=== {gname} ===")
    results[gname] = {}
    for label, over in GRID:
        s = score_group(B.Config(**over), subset, label)
        results[gname][label] = s["adj_edge_jaccard"]

df = pd.DataFrame(results)
df["dense-sparse"] = df["DENSE"] - df["SPARSE"]
df.to_csv(OUT / "dense_rescue_grid.csv")
display(df.round(4))
"""),

    code(r"""
print("best setting per density group:\n")
for g in ["DENSE", "MID", "SPARSE"]:
    col = df[g].sort_values(ascending=False)
    base = df.loc["baseline", g]
    print(f"{g:<7} best = {col.index[0]:<16} {col.iloc[0]:.4f}   "
          f"(baseline {base:.4f}, delta {col.iloc[0]-base:+.4f})")

same = len({df[g].idxmax() for g in ['DENSE','MID','SPARSE']}) == 1
print("\n" + (" Same setting wins everywhere: a density-conditioned rule is NOT justified."
              if same else
              " Different settings win per group: a density-conditioned rule IS justified."))
"""),

    md(r"""
## If a density rule is justified

The pipeline cannot read `estimated_number_of_nodes` at test time, but the **observed peak
count** at default settings is an excellent stand-in and costs one extra detection pass over a
handful of frames. The cell below measures how well it tracks true density; anything above
~0.9 makes it a usable switch.
"""),

    code(r"""
probe = dense[:6] + mid[:6] + sparse[:6]
obs = []
cfg = B.Config()
for name in probe:
    try:
        vol = B.open_volume(TRAIN / f"{name}.zarr")
        gt = B.read_geff(TRAIN / f"{name}.geff")
    except Exception:
        continue
    idx = np.linspace(0, vol.n_t - 1, 4).astype(int)
    counts = []
    for t in idx:
        c, _ = B.detect_dog(vol.frame(int(t)), vol.quantiles, cfg.xy_downsample,
                            cfg.dog_scales, cfg.min_distance_um, cfg.rel_threshold, cfg.max_peaks)
        counts.append(len(c))
    obs.append({"dataset": name, "observed": float(np.mean(counts)),
                "true": gt.meta.get("estimated_number_of_nodes") / 100.0})

o = pd.DataFrame(obs)
r = o["observed"].corr(o["true"])
print(f"correlation(observed peaks/frame, true cells/frame) = {r:.3f}  over {len(o)} videos")
print(f"median ratio observed/true = {(o['observed']/o['true']).median():.3f}")
display(o.round(1))

import matplotlib.pyplot as plt
plt.figure(figsize=(5, 4.5))
plt.scatter(o["true"], o["observed"], s=30)
lim = [0, max(o["true"].max(), o["observed"].max()) * 1.05]
plt.plot(lim, lim, "k--", lw=0.8)
plt.xlabel("true cells/frame"); plt.ylabel("observed peaks/frame")
plt.title(f"density proxy (r = {r:.3f})"); plt.tight_layout(); plt.show()
"""),

    md(r"""
## Reading the result

* **Different argmax per group**  implement the rule in `B.Config` keyed on observed peaks per
  frame, then verify on the full 60-video CV before submitting. Expected upside is large: the
  dense quartile sits at 0.656 against 0.884 for sparse.
* **Same argmax everywhere**  the detector is not the limiting factor for dense videos in a way
  parameters can reach, and the effort belongs in the learned detector (`03`) and a global
  linker instead. Record it in `docs/experiment_log.md` either way.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "06_dense_rescue.ipynb", CELLS))
