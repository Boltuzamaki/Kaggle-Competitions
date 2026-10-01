"""Build notebooks/05_full_cv.ipynb - large-scale cross-validation on Kaggle.

Every tuning decision so far rested on two videos with ~50 GT edges each, where
one edge moves the Jaccard by ~0.02. This notebook scores against as many of the
199 training videos as the time budget allows, using a metric verified to match
the organizers' scorer exactly.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Full cross-validation - score against the real training set

**CPU only.** Attach the competition data and run.

### Why this notebook exists

Every parameter choice in this project has been made on **two** placeholder videos with ~50
ground-truth edges each. At that size a single edge moves the Jaccard by ~0.02, so most
"improvements" were inside the noise. Downloading more videos locally is impractical - the
Kaggle API rate-limits hard on thousands of small files - but on Kaggle all **199** training
videos are already mounted.

The blocker was the metric: the organizers' scorer needs `tracksdata`, `geff` and `polars`,
none of which are installed here. So `biohub_metric.py` reimplements it with numpy + scipy,
and `tests/test_metric_parity.py` asserts **exact** agreement with the reference on TP/FP/FN
and agreement to 1e-9 on the adjusted Jaccard, across three configurations. That test passes.

Reproduced quirks that each change the number materially:

* an edge is a false positive **only if** an endpoint matches an annotated GT node
* several predicted edges collapsing onto one GT edge count once
* out-degree capped at 2
* `adj = J·(1 − 0.1·(N_pred − N_true)/N_true)` - which exceeds 1 below the expected node count
* run level: adjusted Jaccard weighted by `TP+FP+FN`, plain Jaccard micro-averaged
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
print(f"{len(names)} training videos with both image and ground truth")

from collections import Counter
print("per embryo:", dict(Counter(B.embryo_of(n) for n in names)))
"""),

    md(r"""
## Choose the evaluation set

Sampling is **stratified by embryo** and seeded, so every configuration is scored on exactly
the same videos and results stay comparable across runs of this notebook.

Detection costs ~0.17 s/frame × 100 frames ≈ 17 s per video, so 60 videos is ~17 min for one
configuration. Raise `N_VIDEOS` for a final check, lower it while iterating.
"""),

    code(r"""
N_VIDEOS = 60
SEED = 7

rng = random.Random(SEED)
by_emb = {}
for n in names:
    by_emb.setdefault(B.embryo_of(n), []).append(n)

eval_names = []
for emb, group in sorted(by_emb.items()):
    group = sorted(group); rng.shuffle(group)
    take = max(1, round(N_VIDEOS * len(group) / len(names)))
    eval_names += group[:take]
eval_names = sorted(eval_names)[:N_VIDEOS]

print(f"evaluating on {len(eval_names)} videos:", dict(Counter(B.embryo_of(n) for n in eval_names)))
"""),

    md("## Score one configuration"),

    code(r"""
def score_config(cfg, subset, label="", verbose=False):
    rows, per_video = [], []
    t0 = time.time()
    for i, name in enumerate(subset):
        try:
            vol = B.open_volume(TRAIN / f"{name}.zarr")
            gt = B.read_geff(TRAIN / f"{name}.geff")
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {e}")
            continue
        frames = B.detect_frames(vol, cfg)
        graph = B.build_graph(frames, cfg)
        r = M.evaluate(graph, gt, n_total=gt.meta.get("estimated_number_of_nodes"))
        rows.append(r)
        per_video.append({
            "dataset": name, "embryo": B.embryo_of(name),
            "adj": r.adj_edge_jaccard, "edge_j": r.edge_jaccard,
            "node_recall": r.node_recall, "n_pred": r.num_pred_nodes,
            "n_true": gt.meta.get("estimated_number_of_nodes"),
            "peaks_per_frame": float(np.mean([len(f) for f in frames])),
            "tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn,
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(subset)}  ({time.time()-t0:.0f}s)")
    s = M.summarise(rows)
    print(f"{label:<40} {M.format_summary(s)}  ({time.time()-t0:.0f}s)")
    return s, pd.DataFrame(per_video)
"""),

    code(r"""
baseline_cfg = B.Config()
print("baseline configuration:")
for k, v in vars(baseline_cfg).items():
    print(f"   {k:<20} {v}")
print()

baseline, per_video = score_config(baseline_cfg, eval_names, "BASELINE", verbose=True)
per_video.to_csv(OUT / "cv_baseline_per_video.csv", index=False)
"""),

    md(r"""
## Where does the score actually vary?

The two locally-available videos scored 0.90 and 0.59. If that spread holds across 60 videos,
the aggregate is dominated by a subset of hard cases - and knowing *which* is worth more than
any single global parameter tweak.
"""),

    code(r"""
display(per_video.sort_values("adj").head(10)[
    ["dataset", "embryo", "adj", "edge_j", "node_recall", "peaks_per_frame", "n_pred", "n_true"]])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
ax[0].hist(per_video["adj"].dropna(), bins=25, color="#3b7dd8")
ax[0].set(title="adjusted edge Jaccard per video", xlabel="adj")
ax[1].scatter(per_video["node_recall"], per_video["adj"], s=18, alpha=0.7)
ax[1].set(title="node recall vs score", xlabel="node recall", ylabel="adj")
ratio = (per_video["n_pred"] - per_video["n_true"]) / per_video["n_true"]
ax[2].scatter(ratio, per_video["adj"], s=18, alpha=0.7, color="#e07b39")
ax[2].axvline(0, color="k", lw=0.8, ls="--")
ax[2].set(title="node-count error vs score", xlabel="(N_pred − N_true)/N_true", ylabel="adj")
plt.show()

print("per-embryo means:")
display(per_video.groupby("embryo")[["adj", "edge_j", "node_recall"]].agg(["mean", "count"]).round(4))
print(f"\ncorrelation(node_recall, adj) = {per_video['node_recall'].corr(per_video['adj']):.3f}")
print(f"correlation(|count error|, adj) = {ratio.abs().corr(per_video['adj']):.3f}")
"""),

    md(r"""
## Sweep

Each configuration is scored on the **same** videos, so differences are attributable to the
parameter and not to the sample. With ~60 videos and ~5,000 ground-truth edges a difference of
0.01 is meaningful - unlike the two-video harness, where it was noise.
"""),

    code(r"""
SWEEP = [
    ("baseline",              dict()),
    ("ADAPTIVE NMS",          dict(adaptive_nms=True)),
    ("adaptive NMS ratio .17", dict(adaptive_nms=True)),   # ratio varied below
    ("adaptive NMS ratio .24", dict(adaptive_nms=True)),
    ("min_distance 3.0",      dict(min_distance_um=3.0)),
    ("min_distance 4.5",      dict(min_distance_um=4.5)),
    ("rel_threshold 0.01",    dict(rel_threshold=0.01)),
    ("rel_threshold 0.04",    dict(rel_threshold=0.04)),
    ("link 6 µm",             dict(max_link_um=6.0)),
    ("link 8 µm",             dict(max_link_um=8.0)),
    ("min_track_len 2",       dict(min_track_len=2)),
    ("min_track_len 6",       dict(min_track_len=6)),
    ("max_gap 2",             dict(max_gap=2)),
    ("no gap closing",        dict(max_gap=0)),
    ("no motion model",       dict(motion_weight=0.0)),
    ("scales 3-set",          dict(dog_scales=((1.0, 3.0), (1.5, 4.5), (2.5, 7.5)))),
]

SWEEP_VIDEOS = eval_names[:30]   # half the set keeps the sweep inside the session limit
print(f"sweeping {len(SWEEP)} configs on {len(SWEEP_VIDEOS)} videos\n")

# The adaptive-NMS ratio is not a Config field, so it is patched around the call.
import functools
_orig_amd = B.adaptive_min_distance

results = []
for label, overrides in SWEEP:
    ratio = None
    if "ratio .17" in label:
        ratio = 0.17
    elif "ratio .24" in label:
        ratio = 0.24
    if ratio is not None:
        B.adaptive_min_distance = functools.partial(_orig_amd, ratio=ratio)
    else:
        B.adaptive_min_distance = _orig_amd
    cfg = B.Config(**overrides)
    s, _ = score_config(cfg, SWEEP_VIDEOS, label)
    results.append({"config": label, **{k: v for k, v in s.items() if k != "n"}})
B.adaptive_min_distance = _orig_amd

df = pd.DataFrame(results).sort_values("adj_edge_jaccard", ascending=False)
df.to_csv(OUT / "cv_sweep.csv", index=False)
display(df)

base = df[df["config"] == "baseline"]["adj_edge_jaccard"].iloc[0]
print(f"\nbaseline = {base:.4f}; deltas:")
for _, r in df.iterrows():
    if r["config"] != "baseline":
        print(f"   {r['config']:<24} {r['adj_edge_jaccard']-base:+.4f}")
"""),

    md(r"""
## Reading the result

A change is worth adopting only if it beats the baseline by more than the sampling noise of
this evaluation set. With ~30 videos, treat anything under **±0.005** as a tie and prefer the
simpler configuration.

Then remember the calibration from `docs/experiment_log.md`: the public leaderboard has run
about **0.085 above** local CV, but *deltas* transferred almost exactly (+0.042 local
+0.039 LB). Use these numbers to rank, not to predict a score.

Copy any winning setting into `B.Config` in `src/biohub_ct.py`, regenerate the notebooks, and
re-run `02_baseline_classical` before submitting.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "05_full_cv.ipynb", CELLS))
