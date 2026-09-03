import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch

p = Path(__file__).with_name("experiment.py")
s = importlib.util.spec_from_file_location("dcn", p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
rng = np.random.default_rng(17); n = 220
x = pd.DataFrame({
    "daily_screen_time_hours": rng.choice([2., 5., 9., np.nan], n),
    "social_media_hours": rng.choice([1., 3., np.nan], n),
    "gaming_hours": rng.choice([0., 2., np.nan], n),
    "work_study_hours": rng.choice([2., 6., np.nan], n),
    "sleep_hours": rng.choice([5., 8., np.nan], n),
    "notifications_per_day": rng.choice([10., 80., np.nan], n),
    "app_opens_per_day": rng.choice([5., 40., np.nan], n),
    "weekend_screen_time": rng.choice([3., 11., np.nan], n),
    "gender": rng.choice(["Male", "Female", None], n),
})
z = m.engineer(x); enc = m.FoldEncoder().fit(z.iloc[:180]); a, b = enc.transform(z.iloc[:180]); c, d = enc.transform(z.iloc[180:])
assert np.isfinite(a).all() and np.isfinite(c).all() and b.dtype == d.dtype == np.int64
net = m.DCNv2(a.shape[1], enc.cards, d=32)
out = net(torch.from_numpy(a[:16]), torch.from_numpy(b[:16])); assert out.shape == (16,) and torch.isfinite(out).all()
loss = torch.nn.BCEWithLogitsLoss()(out, torch.from_numpy(np.tile([0., 1.], 8).astype("float32")))
loss.backward(); assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
# Encoder and engineered views never accept labels; changing validation labels cannot change features.
y = np.tile([0, 1], 20); before = enc.transform(z.iloc[180:])[0]; y[:] = 1 - y; after = enc.transform(z.iloc[180:])[0]
np.testing.assert_array_equal(before, after)
print({"forward": list(out.shape), "finite_gradients": True, "validation_label_invariance": True,
       "numeric_features": a.shape[1], "categorical_fields": b.shape[1]})
