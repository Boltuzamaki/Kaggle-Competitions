"""Build notebooks/08_postproc_sweep.ipynb - sweep the techniques the public 0.93+ notebooks use.

Detection is the expensive part, so it is run once per video and cached; every
post-processing configuration is then scored on the identical candidate sets.
That makes a 15-config sweep cost roughly one inference pass.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Post-processing sweep - techniques from the public 0.93+ notebooks

**Needs GPU + the `unet_detector.pt` dataset (notebook `03`).**

### Where these ideas come from

A survey of the current top public notebooks (0.926 - 0.936 LB) shows they are all forks of the
organizers' `tracking_cellmot` repo driven by environment variables, sharing a common recipe.
Two of their techniques transfer to this pipeline **without any training**, and both target the
weakness measured here - that wrong links rise from 3 to 26 as candidate density grows:

**1. Harmonic bidirectional association fusion** (`bidir_weight`)
The 0.936 notebook computes edge scores in both time directions and fuses them with a weighted
harmonic mean in probability space:

```
p = 1 / ((1 - w) / p_forward + w / p_reverse)      w = 0.15
```

The harmonic mean is dominated by the smaller term, so a pair is demoted unless **both**
directions support it. Their version fuses a learned model's logits; `link_harmonic` applies the
identical principle to the motion-compensated distance, which needs no model.

**2. Line-fit trajectory smoothing** (`smooth_weight`)
Every top notebook runs `OUTPUT_LINEFIT_SMOOTH`, typically weight 0.8 / window 2. Node matching
uses a hard 7 µm radius and real motion is smooth at ~1.7 µm/frame, so per-frame jitter is
almost pure noise.

**3. Much wider division gates.** This project measured geometric divisions as harmful with
4.7 / 7.5 µm radii (0 true positives, 1 false positive). The public notebooks use **7.0 / 12.0**
with a higher ILP division weight - worth re-testing at their settings before concluding
divisions are unreachable.

### Method

Detection runs **once per video** and is cached, so all configurations see identical candidates
and differences are attributable purely to post-processing.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_unet.py"),

    code(r"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
import biohub_ct as B
import biohub_metric as M
import biohub_unet as U

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMP = B.find_comp_dir()
TRAIN = COMP / "train"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")

ckpts = [p for p in sorted(Path("/kaggle/input").rglob("*.pt"))
         if "input/competitions" not in str(p)]
ckpts.sort(key=lambda p: (p.name != "unet_detector.pt", str(p)))
assert ckpts, "attach the unet_detector.pt dataset / kernel output from notebook 03"
model, meta = U.load_detector(ckpts[0], device=DEVICE)
print(f"loaded {ckpts[0].name}  epoch={meta.get('epoch')} "
      f"top-K recall={meta.get('val_topk_recall')}")
"""),

    code(r"""
N_VIDEOS = 30

names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").is_dir())
dens = []
for n in names:
    try:
        g = B.read_geff(TRAIN / f"{n}.geff")
        e = g.meta.get("estimated_number_of_nodes")
        if e:
            dens.append((n, float(e) / 100.0))
    except Exception:
        pass
dens.sort(key=lambda x: x[1])
idx = np.linspace(0, len(dens) - 1, N_VIDEOS).astype(int)
eval_set = [dens[i][0] for i in idx]
print(f"{len(eval_set)} videos spanning {dens[idx[0]][1]:.0f}-{dens[idx[-1]][1]:.0f} cells/frame")
"""),

    md("## Cache detections once\n\nThis is the only expensive step; every configuration below reuses it."),

    code(r"""
base_cfg = B.Config()
cache, gts = {}, {}
t0 = time.time()
for i, name in enumerate(eval_set):
    vol = B.open_volume(TRAIN / f"{name}.zarr")
    gts[name] = B.read_geff(TRAIN / f"{name}.geff")
    cache[name] = U.detect_frames_unet(
        vol, model, device=DEVICE, threshold=0.0, min_distance_um=3.5,
        budget_from_dog=True, dog_cfg=base_cfg, refine=True)
    if (i + 1) % 5 == 0:
        print(f"  cached {i+1}/{len(eval_set)}  ({time.time()-t0:.0f}s)")
print(f"detection cached in {time.time()-t0:.0f}s; "
      f"mean {np.mean([len(f) for fr in cache.values() for f in fr]):.0f} peaks/frame")
"""),

    code(r"""
def score(cfg, label):
    rows = []
    for name in eval_set:
        g = B.build_graph(cache[name], cfg)
        rows.append(M.evaluate(g, gts[name],
                               n_total=gts[name].meta.get("estimated_number_of_nodes")))
    s = M.summarise(rows)
    print(f"  {label:<34} adj={s['adj_edge_jaccard']:.4f}  J={s['edge_jaccard']:.4f}  "
          f"recall={s['node_recall']:.3f}")
    return s['adj_edge_jaccard']
"""),

    md("## Sweep"),

    code(r"""
GRID = [
    ("baseline (current best)",      dict()),
    # 1. harmonic bidirectional fusion
    ("bidir 0.10",                   dict(bidir_weight=0.10)),
    ("bidir 0.15  <- public value",  dict(bidir_weight=0.15)),
    ("bidir 0.25",                   dict(bidir_weight=0.25)),
    ("bidir 0.15, temp 1.0",         dict(bidir_weight=0.15, bidir_temperature_um=1.0)),
    ("bidir 0.15, temp 4.0",         dict(bidir_weight=0.15, bidir_temperature_um=4.0)),
    # 2. line-fit smoothing
    ("smooth 0.5",                   dict(smooth_weight=0.5)),
    ("smooth 0.8  <- public value",  dict(smooth_weight=0.8)),
    ("smooth 0.8, window 3",         dict(smooth_weight=0.8, smooth_window=3)),
    # 3. divisions at the public (wide) gates
    ("wide divisions 7/12",          dict(safe_divisions=True, div_parent_um=7.0,
                                          div_sister_um=12.0)),
    # 4. other public settings
    ("min_track_len 6",              dict(min_track_len=6)),
    ("max_gap 2",                    dict(max_gap=2)),
    # 5. combinations of whatever helps
    ("bidir .15 + smooth .8",        dict(bidir_weight=0.15, smooth_weight=0.8)),
    ("bidir .15 + smooth .8 + gap2", dict(bidir_weight=0.15, smooth_weight=0.8, max_gap=2)),
]

res = {}
for label, over in GRID:
    res[label] = score(B.Config(**over), label)

base = res["baseline (current best)"]
df = pd.DataFrame({"config": list(res), "adj": list(res.values())})
df["delta"] = df["adj"] - base
df = df.sort_values("adj", ascending=False)
df.to_csv(OUT / "postproc_sweep.csv", index=False)
display(df.round(4))
print(f"\nbaseline {base:.4f}")
"""),

    md(r"""
## Reading the result

With 30 videos, ~0.005 is the noise floor. This project's history is unambiguous on the point:
**every parameter change that looked good on 2 videos evaporated at scale**, while the two
changes that survived (gap closing, the detector loss rewrite) were large enough to be visible
anywhere. Adopt nothing below ~0.01 here.

If `bidir` helps, it is the more interesting result: it attacks the structural limit measured
earlier - wrong links rising 3  26 as candidates get denser - and would make a *higher-recall*
detector usable, which is currently the thing blocking further detection gains.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "08_postproc_sweep.ipynb", CELLS))
