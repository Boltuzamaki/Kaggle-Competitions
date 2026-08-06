"""Local placeholder smoke for the packaged Harshini physics/LGB/XGB legs."""
from pathlib import Path
import json, os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
orig=json.loads((ROOT/"exp/results/harshini_pilkwang_kernel_source/"
                 "rogii-harshini-pilkwang-fresh-inference.ipynb").read_text())
new=json.loads((ROOT/"kernels/harshini_pilkwang_v4/"
                "rogii-harshini-pilkwang-fresh-inference.ipynb").read_text())
cells=["".join(c.get("source",[])) for c in orig["cells"]]
ns={"__name__":"harshini_smoke"}

# Official local data harness.
h=cells[0].replace(
 'CANDS = ["/kaggle/input/rogii-wellbore-geology-prediction",\n'
 '         "/kaggle/input/competitions/rogii-wellbore-geology-prediction", "."]',
 f'CANDS = [r"{ROOT/"data"}"]')
exec(h,ns)
# Build only spatial structures/functions; skip the notebook's diagnostic CV loops.
exec(cells[2].split("# ---- leave-one-well-out evaluation ----")[0],ns)
exec(cells[3].split("# --- 3. evaluate ---")[0],ns)
exec(cells[4].split("random.seed(0); SUB")[0],ns)
# Exact cache-control globals from the actual prerequisite cell. Avoid rebuilding
# CACHE locally, but preserve identical datum windows and reliability window.
exec(cells[5].split("CACHE = {}")[0],ns)

# Reuse the completed legal training cache for trust-model fitting.
old=os.getcwd(); os.chdir(ROOT/"exp/results/harshini_kernel_output")
try: exec(cells[6],ns)
finally: os.chdir(old)

# Execute the optimized component-only inference cell against local packages.
s="".join(new["cells"][8]["source"])
s=s.replace("@njit(cache=True, nogil=True)","@njit(cache=False, nogil=True)")
s=s.replace(
 ' _Path("/kaggle/input/datasets/boltuzamaki/rogii-harshini-fast-models"),',
 f' _Path(r"{ROOT/"exp/results/harshini_fast_model_package"}"),')
exec(s,ns)
q=ns["har_components"]
sample=ns["pd"].read_csv(ROOT/"data/sample_submission.csv")
audit={"rows":len(q),"unique_ids":int(q.id.nunique()),
       "ids_exact_set":set(q.id)==set(sample.id),
       "finite":bool(np.isfinite(q.iloc[:,1:].to_numpy()).all()),
       "columns":q.columns.tolist(),
       "feature_manifest_order_used":True}
if not(audit["rows"]==len(sample) and audit["ids_exact_set"] and audit["finite"]):
    raise RuntimeError(audit)
(ROOT/"exp/results/harshini_fast_model_package/placeholder_component_audit.json"
 ).write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
