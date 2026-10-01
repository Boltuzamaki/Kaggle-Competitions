"""Build notebooks/09_itec_repair.ipynb - evidence-based track repair (ITEC-style).

Tests the premise that tracking should *improve* detection rather than merely
consume it: when a track has a hole, predict where the cell should be, look at
the U-Net heatmap there, and insert the real peak instead of an interpolated
guess. Reports per-embryo as well as overall, because the hidden test embryos
are disjoint from training.
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Evidence-based track repair - letting tracking fix detection

**Needs GPU + the `unet_detector.pt` dataset (notebook `03`).**

### The idea

Today's pipeline closes a one-frame dropout by inserting a node on the straight line between
the two track ends. That is a *guess*, and it is worth +0.017 even so.

But the detector usually did see that cell - it simply ranked below the per-frame top-K budget.
So instead of guessing:

```
track predicts (z,y,x) at the missing frame

inspect the U-Net heatmap around that prediction

search ±4 µm for a real local peak

peak above threshold?   insert the REAL detection
                        otherwise fall back to interpolation
```

This can only be better-informed than interpolation, never blinder. Recovered nodes are then
fed back into the candidate pool and the graph is rebuilt, so a bridged track can keep growing
instead of ending at the patch.

Verified on a synthetic case: with a cell present in the heatmap at x=116 but dropped from the
detection list, interpolation places the node at the chord midpoint while evidence repair
recovers **x = 116.0 exactly**.

### Why per-embryo reporting matters here

Only two embryos exist (`44b6`, `6bba`) and the hidden test set is embryo-disjoint from
training. Aggregate CV therefore flatters anything that overfits an embryo's imaging
conditions. This repair is *algorithmic*, so it should transfer ~1:1 - but the per-embryo split
below is the check, and it is the pattern every future **learned** component needs.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_unet.py"),

    code(r"""
import sys, time, warnings, gc
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
assert ckpts, "attach the unet_detector.pt output of notebook 03"
model, meta = U.load_detector(ckpts[0], device=DEVICE)
print(f"loaded {ckpts[0].name}  epoch={meta.get('epoch')}")
"""),

    code(r"""
N_PER_EMBRYO = 8      # heatmaps are ~52 MB per video, so keep the cache modest

names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").is_dir())
by_emb = {}
for n in names:
    try:
        g = B.read_geff(TRAIN / f"{n}.geff")
        e = g.meta.get("estimated_number_of_nodes")
        if e:
            by_emb.setdefault(B.embryo_of(n), []).append((n, float(e) / 100.0))
    except Exception:
        pass

eval_set = []
for emb, lst in sorted(by_emb.items()):
    lst.sort(key=lambda x: x[1])                      # spread across the density range
    idx = np.linspace(0, len(lst) - 1, N_PER_EMBRYO).astype(int)
    eval_set += [lst[i][0] for i in idx]
print(f"{len(eval_set)} videos:",
      {e: sum(1 for n in eval_set if B.embryo_of(n) == e) for e in by_emb})
"""),

    md("## Cache detections **and heatmaps** once\n\nThe heatmap is what makes evidence-based repair possible, so it has to survive inference. float16 keeps a 100-frame video near 52 MB."),

    code(r"""
base_cfg = B.Config()
det, hms, gts = {}, {}, {}
t0 = time.time()
for i, name in enumerate(eval_set):
    vol = B.open_volume(TRAIN / f"{name}.zarr")
    gts[name] = B.read_geff(TRAIN / f"{name}.geff")
    frames, heat = U.detect_frames_unet(
        vol, model, device=DEVICE, threshold=0.0, min_distance_um=3.5,
        budget_from_dog=True, dog_cfg=base_cfg, refine=True, return_heatmaps=True)
    det[name], hms[name] = frames, heat
    if (i + 1) % 4 == 0:
        print(f"  {i+1}/{len(eval_set)}  ({time.time()-t0:.0f}s)")
mb = sum(sum(h.nbytes for h in v) for v in hms.values()) / 1e6
print(f"cached in {time.time()-t0:.0f}s; heatmaps hold {mb:.0f} MB")
"""),

    code(r"""
def evaluate(builder, label):
    rows, per = [], []
    ev = it = 0
    for name in eval_set:
        g = builder(name)
        ev += g.meta.get("repair_from_evidence", 0)
        it += g.meta.get("repair_interpolated", 0)
        r = M.evaluate(g, gts[name],
                       n_total=gts[name].meta.get("estimated_number_of_nodes"))
        rows.append(r)
        per.append({"dataset": name, "embryo": B.embryo_of(name), "adj": r.adj_edge_jaccard})
    s = M.summarise(rows)
    pv = pd.DataFrame(per)
    byemb = pv.groupby("embryo")["adj"].mean()
    print(f"  {label:<40} adj={s['adj_edge_jaccard']:.4f}  "
          f"44b6={byemb.get('44b6', float('nan')):.4f}  6bba={byemb.get('6bba', float('nan')):.4f}"
          + (f"   [evidence {ev} / interp {it}]" if ev or it else ""))
    return s["adj_edge_jaccard"], byemb

def build_baseline(cfg):
    return lambda name: B.build_graph(det[name], cfg)

def build_repair(cfg, rounds):
    return lambda name: B.build_graph_iterative(det[name], cfg, hms[name], rounds=rounds)
"""),

    md("## Does evidence beat interpolation?"),

    code(r"""
results = {}
print("current pipeline (interpolated gap closing):")
results["baseline"] = evaluate(build_baseline(B.Config()), "baseline, max_gap 1")

print("\nevidence-based repair, 1 round:")
for search in (3.0, 4.0, 6.0):
    for resp in (0.20, 0.30, 0.50):
        cfg = B.Config(repair_search_um=search, repair_min_response=resp)
        results[f"ev s{search} r{resp}"] = evaluate(
            build_repair(cfg, 1), f"search {search} µm, min_response {resp}")
"""),

    code(r"""
print("iterative rounds (tracking -> detection -> tracking):")
best = max((k for k in results if k.startswith("ev ")), key=lambda k: results[k][0])
sp = best.split()
search, resp = float(sp[1][1:]), float(sp[2][1:])
print(f"  using the best single-round setting: search {search} µm, min_response {resp}\n")
for rounds in (1, 2, 3):
    cfg = B.Config(repair_search_um=search, repair_min_response=resp)
    results[f"ev rounds={rounds}"] = evaluate(build_repair(cfg, rounds), f"{rounds} round(s)")

print("\nlarger gaps now that patches sit on real evidence:")
for mg in (1, 2, 3):
    cfg = B.Config(max_gap=mg, repair_search_um=search, repair_min_response=resp)
    results[f"ev max_gap={mg}"] = evaluate(build_repair(cfg, 2), f"max_gap {mg}, 2 rounds")
"""),

    code(r"""
base = results["baseline"][0]
df = pd.DataFrame([{"config": k, "adj": v[0], "delta": v[0] - base,
                    "44b6": v[1].get("44b6", np.nan), "6bba": v[1].get("6bba", np.nan)}
                   for k, v in results.items()]).sort_values("adj", ascending=False)
df.to_csv(OUT / "itec_repair.csv", index=False)
display(df.round(4))
print(f"baseline {base:.4f}")

win = df.iloc[0]
print(f"\nbest: {win['config']}  {win['adj']:.4f}  ({win['delta']:+.4f})")
print(f"  per embryo: 44b6 {win['44b6']:+.4f} vs baseline {results['baseline'][1].get('44b6', float('nan')):.4f}, "
      f"6bba {win['6bba']:.4f} vs {results['baseline'][1].get('6bba', float('nan')):.4f}")
print("\nAdopt only if the gain holds in BOTH embryos - the hidden test is embryo-disjoint.")
"""),

    md(r"""
## Reading the result

With 16 videos, treat anything under ~0.008 as noise. The decisive check is the per-embryo
split: a gain that appears in one embryo and not the other is an imaging-condition artifact,
and the hidden test is embryo-disjoint.

The `[evidence N / interp M]` counter shows how often a real peak was found versus how often the
repair fell back to interpolation. If evidence rarely wins, either the search radius is too
tight or the detector's budget was not what dropped those cells - both worth knowing before
building anything further on this idea.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "09_itec_repair.ipynb", CELLS))
