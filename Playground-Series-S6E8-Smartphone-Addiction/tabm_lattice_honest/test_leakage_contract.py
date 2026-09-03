"""Executable invariance checks for the nested evidence construction."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_functions():
    source = Path("tabm_lattice_honest/experiment.py").read_text()
    prefix = source[: source.index("\ntrain_path, test_path = locate()")]
    namespace = {}
    exec(compile(prefix, "experiment.py", "exec"), namespace)
    return namespace


ns = load_functions()
rng = np.random.default_rng(6082026)
n, nt = 240, 31
train = pd.DataFrame({
    "id": np.arange(n),
    "age": rng.choice([18.0, 19.0, 20.0, np.nan], n),
    "daily_screen_time_hours": rng.choice([4.0, 6.0, 8.0, np.nan], n),
    "social_media_hours": rng.choice([1.0, 2.0, 4.0, np.nan], n),
    "gaming_hours": rng.choice([0.0, 1.0, 2.0, np.nan], n),
    "work_study_hours": rng.choice([1.0, 3.0, 5.0, np.nan], n),
    "sleep_hours": rng.choice([5.0, 7.0, 9.0, np.nan], n),
    "notifications_per_day": rng.choice([10.0, 50.0, 100.0, np.nan], n),
    "app_opens_per_day": rng.choice([5.0, 25.0, 60.0, np.nan], n),
    "weekend_screen_time": rng.choice([5.0, 8.0, 11.0, np.nan], n),
    "gender": rng.choice(["Male", "Female", None], n),
    "stress_level": rng.choice(["Low", "Medium", "High", None], n),
    "academic_work_impact": rng.choice(["Low", "Medium", "High", None], n),
})
test = train.iloc[:nt].copy()
test["id"] = np.arange(10_000, 10_000 + nt)
y = np.tile([0, 1], n // 2).astype("int8")
keys, test_keys = ns["make_keys"](train), ns["make_keys"](test)
fit, val = next(StratifiedKFold(5, shuffle=True, random_state=ns["SEED"]).split(train, y))

base = ns["fold_evidence"](keys, test_keys, y, fit, val, 1)
changed_validation_labels = y.copy()
changed_validation_labels[val] = 1 - changed_validation_labels[val]
after_val_change = ns["fold_evidence"](keys, test_keys, changed_validation_labels, fit, val, 1)
for left, right in zip(base, after_val_change):
    pd.testing.assert_frame_equal(left, right, check_exact=True)

# Hold the inner partition fixed, then change one inner-validation row's label.
# Its mapping is fitted only on `ia`, so its own evidence must be invariant.
yf = y[fit]
ia, ib = next(StratifiedKFold(
    ns["INNER"], shuffle=True, random_state=ns["SEED"] + 1
).split(fit, yf))
position = int(ib[0])
changed_yf = yf.copy()
changed_yf[position] = 1 - changed_yf[position]
for col in keys:
    smooth = 60.0 if col.startswith("pair__") else 30.0
    before = ns["map_evidence"](
        keys.iloc[fit[ia]][col], keys.iloc[[fit[position]]][col], yf[ia], smooth
    )
    after = ns["map_evidence"](
        keys.iloc[fit[ia]][col], keys.iloc[[fit[position]]][col], changed_yf[ia], smooth
    )
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])

print({
    "outer_validation_label_invariance": True,
    "own_label_exclusion": True,
    "evidence_columns": base[0].shape[1],
})
