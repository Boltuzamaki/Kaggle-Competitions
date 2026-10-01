"""Build notebooks/10_ilp_ablation.ipynb - swap Hungarian for the pretrained edge scorer + ILP.

A clean causal experiment. Detection is frozen; only graph construction changes.
Four arms on the same videos, reported per embryo:

  A  public detector + public edge scorer + ILP   (untouched reference)
  B  our detector    + distance                   (current baseline)
  C  our detector    + public edge scorer + ILP   (the main question)
  D  our detector    + public edge scorer + greedy (isolates the ILP's contribution)
"""

from nbbuild import REPO, build, code, library_cell, md

CELLS = [
    md(r"""
# Replacing Hungarian with a learned edge scorer + ILP

**Attach:** competition data · `pilkwang/biohub-tracking-support-pack-50ep-v1` · our
`unet_detector.pt` (notebook `03`). GPU.

### Why this and nothing else

Three independent experiments now say the same thing - the linker, not the detector, is the
binding constraint:

| experiment | what improved | score |
|---|---|---|
| detection union | node recall 0.902  0.990 | **−0.20** |
| dense NMS (2.0 µm) | more candidates | **−0.38** (wrong links 3  26) |
| ITEC evidence repair | 22,043 real cells recovered at gap 3 | **−0.046** | Better detection is worthless until graph construction can absorb it. So detection is **frozen**
here: no ITEC, no extra candidates, no gap/smoothing/division tuning. The only variable is how
edges are chosen.

### Preserving their representation

The pretrained edge model was trained on features from *its own* U-Net encoder, so feeding it
bare coordinates would be a domain mismatch. Instead only their **detection** call is patched - `_detect_cells_pooled` returns our centroids - and everything downstream (encoder features,
positional embedding, `predict_edges`, ILP) runs exactly as shipped.

Two details that make this safe: their `downsample` is `(1, 4, 4)`, **identical to our grid**;
and their normalisation uses the `0.001 / 0.999` quantiles rather than our `0.01 / 0.99`, which
their own code path applies for us.

Their node coordinates are integers on the downsampled grid, so handing over our sub-voxel
centroids costs up to 1.625 µm of XY precision inside the model. The final graph is snapped back
to our refined positions afterwards, so only the *model input* is quantised.
"""),

    library_cell(),
    library_cell("biohub_metric.py"),
    library_cell("biohub_unet.py"),

    md("## 1 · Bring up the support pack (offline wheels, no internet)"),

    code(r"""
import sys, os, time, glob, json, warnings, subprocess
from pathlib import Path
import numpy as np, pandas as pd, torch

warnings.filterwarnings("ignore")

PACK = None
for c in Path("/kaggle/input").glob("*"):
    if (c / "ARTIFACT_MANIFEST.json").exists() or (c / "repo").is_dir():
        PACK = c; break
if PACK is None:
    hits = list(Path("/kaggle/input").rglob("ARTIFACT_MANIFEST.json"))
    PACK = hits[0].parent if hits else None
assert PACK is not None, "attach pilkwang/biohub-tracking-support-pack-50ep-v1"
print("support pack:", PACK)
print(sorted(p.name for p in PACK.iterdir())[:12])

# Install ONLY what is genuinely missing. Installing every wheel blindly
# downgraded Kaggle's numpy and broke scipy, which is compiled against it:
# AttributeError: module 'numpy._core._multiarray_umath'
# has no attribute '_blas_supports_fpe'
# Anything already working in this image is left strictly alone.
NEVER_TOUCH = {
    "numpy", "scipy", "torch", "torchvision", "pandas", "matplotlib",
    "pillow", "networkx", "numba", "llvmlite", "scikit_learn", "scikit-learn",
    "typing_extensions", "packaging", "setuptools", "wheel", "sympy", "filelock",
}

def _pkg_of(whl):
    return Path(whl).name.split("-")[0].lower().replace("_", "-")

def _have(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False

import importlib
wheels = sorted(glob.glob(str(PACK / "wheels" / "*.whl")))
print(f"{len(wheels)} wheels in the pack")

before = {m: _have(m) for m in ["numpy", "scipy", "torch"]}
selected, skipped = [], []
for w in wheels:
    name = _pkg_of(w)
    if name in NEVER_TOUCH or name.replace("-", "_") in NEVER_TOUCH:
        skipped.append(name)
    else:
        selected.append(w)
print(f"installing {len(selected)}, protecting {len(set(skipped))} core packages: "
      f"{sorted(set(skipped))}")

if selected:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        "-q", "--no-deps", *selected], capture_output=True, text=True)
    print("pip rc =", r.returncode)
    if r.returncode != 0:
        print(r.stdout[-1200:]); print(r.stderr[-1200:])

# Fail loudly here rather than 20 minutes later inside their code.
for m, was in before.items():
    now = _have(m)
    print(f"  {m}: {'ok' if now else 'BROKEN'}" + ("" if now == was else "  <-- CHANGED"))
    assert now or not was, f"{m} was working before the wheel install and is broken now"
"""),

    code(r"""
REPO_SRC = PACK / "repo" / "src"
REPO_SCRIPTS = PACK / "repo" / "scripts"
for p in (str(REPO_SRC), str(REPO_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib
ok = {}
for m in ["tracksdata", "geff", "zarr", "polars", "pyscipopt", "dask"]:
    try:
        mod = importlib.import_module(m)
        ok[m] = getattr(mod, "__version__", "?")
    except Exception as e:
        ok[m] = f"MISSING ({type(e).__name__})"
print(json.dumps(ok, indent=1))

import predict_unet_transformer as PUT
print("\nloaded their predict module:", PUT.__file__)
WEIGHTS = next(PACK.rglob("edge_predictor_best.pth"), None) or next(PACK.rglob("checkpoint_last.pth"))
print("weights:", WEIGHTS)
"""),

    code(r"""
import biohub_ct as B, biohub_metric as M, biohub_unet as U
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMP = B.find_comp_dir(); TRAIN = COMP / "train"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")

pack_model, WINDOW, DOWNSAMPLE = PUT.load_model(WEIGHTS, DEVICE)
print(f"public model loaded · window={WINDOW} · downsample={DOWNSAMPLE}")
assert tuple(DOWNSAMPLE)[1:] == (4, 4), f"unexpected downsample {DOWNSAMPLE}"

ours = [p for p in sorted(Path("/kaggle/input").rglob("unet_detector*.pt"))]
assert ours, "attach our unet_detector.pt (notebook 03 output)"
our_model, our_meta = U.load_detector(ours[0], device=str(DEVICE))
print("our detector:", ours[0].name, "epoch", our_meta.get("epoch"))
"""),

    md("## 2 · Evaluation set - small, balanced, both embryos"),

    code(r"""
N_PER_EMBRYO = 3          # ILP over 100 frames is slow; keep the matrix affordable
names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / f"{p.stem}.zarr").is_dir())
by_emb = {}
for n in names:
    try:
        g = B.read_geff(TRAIN / f"{n}.geff")
        e = g.meta.get("estimated_number_of_nodes")
        if e: by_emb.setdefault(B.embryo_of(n), []).append((n, float(e) / 100.0))
    except Exception: pass
eval_set = []
for emb, lst in sorted(by_emb.items()):
    lst.sort(key=lambda x: x[1])
    idx = np.linspace(0, len(lst) - 1, N_PER_EMBRYO).astype(int)
    eval_set += [lst[i][0] for i in idx]
gts = {n: B.read_geff(TRAIN / f"{n}.geff") for n in eval_set}
print(eval_set)
"""),

    md("## 3 · Our detections, computed once and frozen"),

    code(r"""
base_cfg = B.Config()
our_det = {}
t0 = time.time()
for n in eval_set:
    vol = B.open_volume(TRAIN / f"{n}.zarr")
    our_det[n] = U.detect_frames_unet(vol, our_model, device=str(DEVICE), threshold=0.0,
                                      min_distance_um=3.5, budget_from_dog=True,
                                      dog_cfg=base_cfg, refine=True)
    print(f"  {n}: {np.mean([len(f) for f in our_det[n]]):.0f} peaks/frame")
print(f"({time.time()-t0:.0f}s)")
"""),

    md(r"""
## 4 · The patch

`_detect_cells_pooled` is replaced by a function that ignores the public detection head and
returns **our** centroids for frame `t`, in their expected format: `(N, 4) int16`, columns
`[t, z, y, x]`, on the downsampled grid. Nothing else in their code is touched.
"""),

    code(r"""
_ORIG_DETECT = PUT._detect_cells_pooled
_INJECT = {"coords": None}

def _inject_detections(det_logits, t, det_threshold=0.5, pool_kernel=(3, 3, 3)):
    frames = _INJECT["coords"]
    if frames is None:
        return _ORIG_DETECT(det_logits, t, det_threshold, pool_kernel)
    if t >= len(frames) or len(frames[t]) == 0:
        return np.empty((0, 4), dtype=np.int16)
    c = np.asarray(frames[t], dtype=np.float64).copy()
    c[:, 1] /= 4.0; c[:, 2] /= 4.0          # original voxels -> their downsampled grid
    Zc, Yc, Xc = det_logits.shape[-3:]
    c[:, 0] = np.clip(np.round(c[:, 0]), 0, Zc - 1)
    c[:, 1] = np.clip(np.round(c[:, 1]), 0, Yc - 1)
    c[:, 2] = np.clip(np.round(c[:, 2]), 0, Xc - 1)
    c = np.unique(c, axis=0)                 # their grid can collapse near-duplicates
    tcol = np.full((len(c), 1), t)
    return np.concatenate([tcol, c], axis=1).astype(np.int16)

def run_pack(name, coords=None, use_ilp=True, threshold=0.5, max_frames=None):
    # predict_video returns the PRE-ILP (coords, edges); their predict() is what
    # then builds a tracksdata graph and runs ILPSolver. Calling predict_video
    # alone silently produced a greedy graph in both "ILP" and "greedy" arms of
    # the first run, which is why those two arms came back byte-identical.
    _INJECT["coords"] = coords
    PUT._detect_cells_pooled = _inject_detections if coords is not None else _ORIG_DETECT
    try:
        cfg = PUT.PredictConfig(use_ilp=use_ilp, threshold=threshold)
        c, e = PUT.predict_video(pack_model, TRAIN / name, DEVICE, cfg,
                                 window_size=WINDOW, max_frames=max_frames,
                                 downsample=tuple(DOWNSAMPLE))
    finally:
        PUT._detect_cells_pooled = _ORIG_DETECT
        _INJECT["coords"] = None

    if not use_ilp:
        return to_graph(c, e)

    import tracksdata as td
    graph = PUT.build_graph(c, e)
    if graph.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=cfg.ilp_appearance_weight,
            disappearance_weight=cfg.ilp_disappearance_weight,
            division_weight=cfg.ilp_division_weight,
        )
        with PUT.suppress_output():
            graph = solver.solve(graph)
    return td_to_trackgraph(graph)

def td_to_trackgraph(graph):
    # Solved tracksdata graph -> our TrackGraph (consecutive-frame edges only).
    import tracksdata as td
    K = td.DEFAULT_ATTR_KEYS
    na = graph.node_attrs(attr_keys=[K.NODE_ID, K.T, "z", "y", "x"])
    ids = np.asarray(na[K.NODE_ID]); tt = np.asarray(na[K.T]).astype(np.int64)
    z = np.asarray(na["z"], dtype=np.float64)
    y = np.asarray(na["y"], dtype=np.float64)
    x = np.asarray(na["x"], dtype=np.float64)
    t_of = {int(i): int(v) for i, v in zip(ids, tt)}
    ea = graph.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET])
    src = np.asarray(ea[K.EDGE_SOURCE]); dst = np.asarray(ea[K.EDGE_TARGET])
    e = [(int(a), int(b)) for a, b in zip(src, dst)
         if t_of.get(int(b), -10**9) - t_of.get(int(a), 10**9) == 1]
    return B.TrackGraph(t=tt, z=z, y=y, x=x, ids=ids.astype(np.int64),
                        edges=np.array(e, dtype=np.int64).reshape(-1, 2))

def to_graph(coords, edges):
    # Their (coords, edges) -> our TrackGraph, ignoring non-consecutive edges.
    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    ids = np.arange(1, n + 1, dtype=np.int64)
    e = []
    for rec in edges:
        i, j = int(rec[0]), int(rec[1])
        if 0 <= i < n and 0 <= j < n and int(coords[j, 0]) - int(coords[i, 0]) == 1:
            e.append((ids[i], ids[j]))
    return B.TrackGraph(t=coords[:, 0].astype(np.int64), z=coords[:, 1],
                        y=coords[:, 2], x=coords[:, 3], ids=ids,
                        edges=np.array(e, dtype=np.int64).reshape(-1, 2))
"""),

    md("## 5 · The four arms"),

    code(r"""
def score_arm(fn, label):
    rows, per = [], []
    t0 = time.time()
    for n in eval_set:
        try:
            g = fn(n)
        except Exception as ex:
            print(f"  {label}: {n} FAILED {type(ex).__name__}: {ex}")
            continue
        r = M.evaluate(g, gts[n], n_total=gts[n].meta.get("estimated_number_of_nodes"))
        rows.append(r)
        per.append({"embryo": B.embryo_of(n), "adj": r.adj_edge_jaccard,
                    "nodes": r.num_pred_nodes, "tp": r.edge_tp, "fp": r.edge_fp, "fn": r.edge_fn})
    if not rows:
        print(f"  {label:<44} no successful videos"); return None
    s = M.summarise(rows); pv = pd.DataFrame(per)
    be = pv.groupby("embryo")["adj"].mean()
    print(f"  {label:<44} adj={s['adj_edge_jaccard']:.4f}  "
          f"44b6={be.get('44b6', float('nan')):.4f}  6bba={be.get('6bba', float('nan')):.4f}  "
          f"FP={s['edge_fp']} FN={s['edge_fn']}  ({time.time()-t0:.0f}s)")
    return {"adj": s["adj_edge_jaccard"], "44b6": be.get("44b6", np.nan),
            "6bba": be.get("6bba", np.nan), "fp": s["edge_fp"], "fn": s["edge_fn"]}

R = {}
print("A · public detector + public edge scorer + ILP  (untouched reference)")
R["A public+public+ILP"] = score_arm(
    lambda n: run_pack(n, coords=None, use_ilp=True), "A")
"""),

    code(r"""
print("B · our detector + distance + Hungarian  (current baseline)")
R["B ours+distance"] = score_arm(
    lambda n: B.build_graph(our_det[n], base_cfg), "B")
"""),

    code(r"""
print("C · our detector + public edge scorer + ILP  (main experiment)")
R["C ours+learned+ILP"] = score_arm(
    lambda n: run_pack(n, coords=our_det[n], use_ilp=True), "C")

print("\nD · our detector + public edge scorer + greedy  (isolates ILP)")
R["D ours+learned+greedy"] = score_arm(
    lambda n: run_pack(n, coords=our_det[n], use_ilp=False), "D")
"""),

    code(r"""
c_, d_ = R.get("C ours+learned+ILP"), R.get("D ours+learned+greedy")
if c_ and d_ and abs(c_["adj"] - d_["adj"]) < 1e-9:
    print("\n!! C and D are identical - the ILP is not being applied. "
          "predict_video returns the PRE-ILP graph; the solver runs in predict().")

df = pd.DataFrame([{"arm": k, **v} for k, v in R.items() if v]).set_index("arm")
df["delta_vs_B"] = df["adj"] - (R["B ours+distance"]["adj"] if R.get("B ours+distance") else np.nan)
df.to_csv(OUT / "ilp_ablation.csv")
display(df.round(4))

b = R.get("B ours+distance"); c = R.get("C ours+learned+ILP")
if b and c:
    both = (c["44b6"] > b["44b6"]) and (c["6bba"] > b["6bba"])
    print(f"\nC vs B: {c['adj']-b['adj']:+.4f} overall · "
          f"44b6 {c['44b6']-b['44b6']:+.4f} · 6bba {c['6bba']-b['6bba']:+.4f}")
    print(f"edge FP {b['fp']} -> {c['fp']}   FN {b['fn']} -> {c['fn']}")
    print(("PASS - both embryos improve" if both and c["adj"] - b["adj"] >= 0.01
           else "FAIL the stated bar - do not tune thresholds; inspect calibration/features first"))
"""),

    md(r"""
## Reading it

The pass bar, set before running: **both embryos improve, overall ≥ +0.01 CV, and edge false
positives fall materially.**

* **C > B in both embryos**  learned graph construction transfers; submit immediately.
* **D ≈ C**  the appearance/context in the edge scorer is doing the work, not global consistency.
* **C ≈ B but D < C**  the ILP's global consistency is the active ingredient.
* **A > C**  feature-domain mismatch, not a weak detector; inspect edge-probability calibration
  per embryo before changing anything else.

Divisions are left at the pack's defaults in every arm, so they cannot explain a difference
between arms. Continuation edges are the whole question here.
"""),
]

if __name__ == "__main__":
    print("wrote", build(REPO / "notebooks" / "10_ilp_ablation.ipynb", CELLS))
