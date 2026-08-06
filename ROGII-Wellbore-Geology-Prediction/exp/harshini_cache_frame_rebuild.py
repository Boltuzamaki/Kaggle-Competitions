"""Rebuild Harshini's 92-feature training frame from completed legal caches.

This deliberately reuses the exact public notebook implementation rather than
silently reimplementing formulas.  Only data/cache paths and the stopping point
are changed.  The expensive surface and PF computations are read from cache.
"""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "references/luffyh04_harshini_submission_f/harshini-submission-f.ipynb"
CACHE_DIR = ROOT / "exp/results/harshini_kernel_output"
OUT = ROOT / "exp/results/harshini_rebuild"
OUT.mkdir(parents=True, exist_ok=True)

n = json.loads(NB.read_text())
cell0 = "".join(n["cells"][0]["source"])
cell6 = "".join(n["cells"][6]["source"])
cell8 = "".join(n["cells"][8]["source"])

# Point the notebook harness at the local official competition copy.
cell0 = cell0.replace(
    'CANDS = ["/kaggle/input/rogii-wellbore-geology-prediction",\n'
    '         "/kaggle/input/competitions/rogii-wellbore-geology-prediction", "."]',
    f'CANDS = [{str(ROOT / "data")!r}]')
cell6 = cell6.replace(
    'pickle.load(open("imp_cache.pkl","rb"))',
    f'pickle.load(open({str(CACHE_DIR / "imp_cache.pkl")!r},"rb"))')
cell8 = cell8.replace(
    'PF_CACHE="pf_train_cache.pkl"',
    f'PF_CACHE={str(CACHE_DIR / "pf_train_cache.pkl")!r}')
cell8 = cell8.replace("@njit(cache=True, nogil=True)",
                      "@njit(cache=False, nogil=True)")
# Stop immediately after the feature frame has been materialized.
cut = cell8.index("X=RR_[FEATS_R]")
cell8 = cell8[:cut]

ns = {"__name__": "_harshini_frame_rebuild"}
exec(compile(cell0, str(NB) + ":cell0", "exec"), ns)
ns["FORM"] = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
exec(compile(cell6, str(NB) + ":cell6", "exec"), ns)
exec(compile(cell8, str(NB) + ":cell8", "exec"), ns)

frame = ns["RR_"]
features = ns["FEATS_R"]
frame.to_pickle(OUT / "train_frame.pkl")
(OUT / "features.json").write_text(json.dumps(features, indent=2))
meta = {
    "rows": len(frame), "wells": int(frame.well.nunique()),
    "features": len(features), "train_stride": int(ns["TRAIN_STRIDE"]),
    "source_notebook": str(NB),
    "imp_cache": str(CACHE_DIR / "imp_cache.pkl"),
    "pf_cache": str(CACHE_DIR / "pf_train_cache.pkl"),
}
(OUT / "frame_meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
