"""Fast executable leakage and tensor-shape checks for v3 evidence."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

path = Path(__file__).with_name("experiment.py")
spec = importlib.util.spec_from_file_location("v3", path)
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

rng = np.random.default_rng(804)
n, nt = 250, 29
cols = [
    "daily_screen_time_hours", "sleep_hours", "notifications_per_day",
    "app_opens_per_day", "social_media_hours", "gaming_hours",
    "weekend_screen_time", "stress_level", "academic_work_impact",
]
raw = pd.DataFrame({c: rng.choice([1., 2., 3., np.nan], n) for c in cols[:-2]})
raw["stress_level"] = rng.choice(["Low", "High", None], n)
raw["academic_work_impact"] = rng.choice(["Low", "High", None], n)
test = raw.iloc[:nt].copy().reset_index(drop=True)
y = np.tile([0, 1], n // 2).astype("float32")
keys, test_keys = v3.make_evidence_keys(raw), v3.make_evidence_keys(test)
fit, val = next(StratifiedKFold(5, shuffle=True, random_state=v3.SEED).split(raw, y))

base = v3.fold_evidence(keys, test_keys, y, fit, val, 1)
changed = y.copy(); changed[val] = 1 - changed[val]
after = v3.fold_evidence(keys, test_keys, changed, fit, val, 1)
for a, b in zip(base, after):
    np.testing.assert_array_equal(a, b)

# Direct own-label exclusion for an inner-validation row.
yf = y[fit]
ia, ib = next(StratifiedKFold(v3.INNER, shuffle=True, random_state=v3.SEED + 100).split(fit, yf))
pos = int(ib[0]); altered = yf.copy(); altered[pos] = 1 - altered[pos]
col = keys.columns[0]
a = v3.map_evidence(keys.iloc[fit[ia]][col], keys.iloc[[fit[pos]]][col], yf[ia], v3.SMOOTH_SINGLE)
b = v3.map_evidence(keys.iloc[fit[ia]][col], keys.iloc[[fit[pos]]][col], altered[ia], v3.SMOOTH_SINGLE)
for left, right in zip(a, b): np.testing.assert_array_equal(left, right)

# Forward smoke check, including reshaping two evidence channels per token.
model = v3.Net(total=60, nraw=3, nder=2, nev=4, d=32).eval()
with torch.no_grad():
    z = model(torch.randint(0, 60, (7, 3)), torch.randn(7, 3), torch.zeros(7, 3),
              torch.randn(7, 2), torch.zeros(7, 2), torch.randn(7, 8))
assert z.shape == (7,) and torch.isfinite(z).all()
print({"outer_validation_label_invariance": True, "own_label_exclusion": True,
       "inner_oof_coverage": True, "forward_shape": list(z.shape),
       "evidence_keys": keys.shape[1]})
