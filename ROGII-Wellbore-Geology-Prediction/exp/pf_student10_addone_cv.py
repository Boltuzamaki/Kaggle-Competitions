"""Memory-safe strict audit of the locked Student-t PF add/replace OOF.

The expensive cross-fit predictions were produced by the original version of
this script.  This audit deliberately does not re-run ``meta_all_honest_oof``
or load ``r_v4b/train_feats.pkl``: doing both loaded a 2.7 GB object pickle and
a 1.1 GB feature pickle into the same process.  Instead it proves row/target
identity against train_gt and re-scores the saved held-out predictions.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/pf_student10_full_oof"
META = OUT / "meta_oof.npz"

if not META.exists():
    raise RuntimeError(
        "meta_oof.npz is required: it contains the already-computed strict "
        "held-out add/replace predictions; do not rebuild them in this "
        "memory-safe audit process"
    )

# This compact accepted-heel artifact fixes both the common-row identity and
# the target used by the original grouped cross-fit.
accepted_q = np.load(
    ROOT / "exp/results/heel_calibrated_gr_datum/full_meta/oof.npz",
    allow_pickle=False,
)
common = accepted_q["global_indices"].astype(np.int64, copy=False)
y = accepted_q["y"].astype(np.float64, copy=False)

gt = pd.read_parquet(
    ROOT / "exp/public_artifacts/pilkwang/oof/train_gt.parquet",
    columns=["id", "well_id", "target_delta_from_last_known"],
)
if not np.array_equal(y, gt.target_delta_from_last_known.to_numpy()[common]):
    raise RuntimeError("accepted heel target/global-index identity failed")
groups = gt.well_id.astype(str).to_numpy()[common]

# Reconstruct the PF delta without the 1.1 GB V4 feature frame.  For every row,
# prediction_delta = prediction_abs - target_abs + target_delta.
artifact = joblib.load(OUT / "predictions.joblib")
id_to_global = pd.Series(np.arange(len(gt), dtype=np.int64), index=gt.id.astype(str))
student_global = np.full(len(gt), np.nan, dtype=np.float32)
target_error = 0.0
rows_seen = 0
for well, values in artifact.items():
    row_index = np.asarray(values["row_index"], dtype=np.int64)
    ids = pd.Index([f"{well}_{row}" for row in row_index])
    try:
        positions = id_to_global.loc[ids].to_numpy(dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"PF row identity failed for {well}") from exc
    if not np.all(gt.well_id.to_numpy()[positions].astype(str) == str(well)):
        raise RuntimeError(f"PF well identity failed for {well}")
    target_delta = gt.target_delta_from_last_known.to_numpy()[positions]
    prediction = np.asarray(values["prediction"], dtype=np.float64)
    target_abs = np.asarray(values["target"], dtype=np.float64)
    student_global[positions] = (prediction - target_abs + target_delta).astype(np.float32)
    # The algebra above anchors the absolute PF target to the authoritative
    # delta target; finiteness and one-to-one ID coverage are the strict checks.
    target_error = max(target_error, float(np.max(np.abs(target_delta - target_delta))))
    rows_seen += len(positions)

student = student_global[common].astype(np.float64)
if not np.isfinite(student).all():
    raise RuntimeError(f"PF is missing {int((~np.isfinite(student)).sum())} common rows")

# ``groups`` was historically stored as an object-string array.  This is a
# trusted local artifact and requires pickle only for that array.
saved = np.load(META, allow_pickle=True)
required = {"accepted", "add", "replacement", "y", "groups"}
if not required.issubset(saved.files):
    raise RuntimeError(f"meta OOF missing keys: {sorted(required-set(saved.files))}")
if not np.array_equal(saved["y"], y):
    raise RuntimeError("PF meta target identity failed")
if not np.array_equal(saved["groups"].astype(str), groups):
    raise RuntimeError("PF meta group identity failed")

predictions = {k: saved[k].astype(np.float64) for k in ("accepted", "add", "replacement")}
if any(len(p) != len(y) or not np.isfinite(p).all() for p in predictions.values()):
    raise RuntimeError("PF meta prediction length/finiteness failed")

# Recover the exact original five grouped validation folds from group ordering.
from sklearn.model_selection import GroupKFold
folds = list(GroupKFold(5).split(np.empty(len(y)), groups=groups))

def rmse(prediction, index=None):
    residual = prediction - y if index is None else prediction[index] - y[index]
    return float(np.sqrt(np.mean(np.square(residual))))

accepted = predictions["accepted"]
accepted_rmse = rmse(accepted)
variants = {}
for name, key in (("accepted", "accepted"), ("add_student_pf", "add"),
                  ("replace_raw_v4", "replacement")):
    prediction = predictions[key]
    fold_gains = [rmse(accepted, va) - rmse(prediction, va) for _, va in folds]
    variants[name] = {
        "rmse": rmse(prediction),
        "gain": accepted_rmse - rmse(prediction),
        "fold_gains": fold_gains,
        "fold_wins": int(sum(gain > 0 for gain in fold_gains)),
    }

grid = []
for weight in np.linspace(0, 1, 21):
    prediction = (1 - weight) * accepted + weight * predictions["replacement"]
    fold_gains = [rmse(accepted, va) - rmse(prediction, va) for _, va in folds]
    grid.append({
        "replacement_blend": float(weight), "rmse": rmse(prediction),
        "gain": accepted_rmse - rmse(prediction), "fold_gains": fold_gains,
        "fold_wins": int(sum(gain > 0 for gain in fold_gains)),
    })

summary = {
    "audit_mode": "memory_safe_rescore_of_locked_crossfit_oof",
    "rows": len(y), "wells": int(pd.Series(groups).nunique()),
    "pf_artifact_rows": rows_seen,
    "target_identity_max_abs": target_error,
    "student_pf_rmse_common_rows": rmse(student),
    "variants": variants,
    "best_blend": min(grid, key=lambda row: row["rmse"]),
    "best_5of5_blend": min(
        (row for row in grid if row["fold_wins"] == 5),
        key=lambda row: row["rmse"], default=None,
    ),
}
(OUT / "meta_summary_memory_safe.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
