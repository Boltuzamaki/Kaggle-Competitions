"""GANDALF-style gated feature learning units for official S6E8 data.

Preprocessing is fitted independently inside every outer fold. The architecture,
optimizer and epoch count are prespecified; outer validation labels are used only
for final AUC reporting and cannot influence training or feature construction.
"""
from pathlib import Path
import gc, json, random, subprocess, sys, time
if Path('/kaggle').exists():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--index-url',
                           'https://download.pytorch.org/whl/cu121', 'torch==2.5.1'])
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED, FOLDS, EPOCHS = "addicted_label", "id", 6082053, 5, 18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists(): continue
        for p in root.rglob("train.csv"):
            try: cols = pd.read_csv(p, nrows=1).columns
            except Exception: continue
            if TARGET in cols and (p.parent / "test.csv").exists(): return p, p.parent / "test.csv"
    raise FileNotFoundError("official competition tables not found")


def engineer(df):
    z = df.copy(); eps = .5
    specs = {
        "social_screen_share": ("social_media_hours", "daily_screen_time_hours"),
        "gaming_screen_share": ("gaming_hours", "daily_screen_time_hours"),
        "weekend_uplift": ("weekend_screen_time", "daily_screen_time_hours"),
        "opens_per_screen": ("app_opens_per_day", "daily_screen_time_hours"),
        "notifications_per_open": ("notifications_per_day", "app_opens_per_day"),
        "screen_sleep_ratio": ("daily_screen_time_hours", "sleep_hours"),
        "screen_work_ratio": ("daily_screen_time_hours", "work_study_hours")}
    for name, (a, b) in specs.items():
        if a in z and b in z: z[name] = z[a] / (z[b].abs() + eps)
    if {"social_media_hours", "gaming_hours"} <= set(z): z["entertainment_hours"] = z.social_media_hours + z.gaming_hours
    if {"daily_screen_time_hours", "sleep_hours", "work_study_hours"} <= set(z):
        z["day_budget_residual"] = 24 - z.daily_screen_time_hours - z.sleep_hours - z.work_study_hours
    z["missing_count"] = df.isna().sum(axis=1).astype("float32")
    for c in df:
        z[c + "__missing"] = df[c].isna().astype("float32")
    return z


class FoldEncoder:
    """Target-free quantiles and one-hot categoricals, learned on outer-fit only."""
    def fit(self, x):
        self.num = [c for c in x if pd.api.types.is_numeric_dtype(x[c])]
        self.cat = [c for c in x if c not in self.num]
        self.med = x[self.num].median(); filled = x[self.num].fillna(self.med)
        self.qt = QuantileTransformer(n_quantiles=min(1000, len(x)), output_distribution="normal",
                                      subsample=250000, random_state=SEED).fit(filled)
        self.levels = {}
        for c in self.cat:
            self.levels[c] = list(x[c].astype("string").fillna("__NA__").unique())
        self.names = self.num + [f"{c}={v}" for c in self.cat for v in self.levels[c]]
        return self

    def transform(self, x):
        parts = [self.qt.transform(x[self.num].fillna(self.med)).astype("float32")]
        for c in self.cat:
            vals = x[c].astype("string").fillna("__NA__")
            parts.append(np.column_stack([(vals == v).to_numpy("float32") for v in self.levels[c]]))
        return np.ascontiguousarray(np.concatenate(parts, 1), dtype="float32")


class GFLU(nn.Module):
    """Feature-wise gates followed by GRU-like staged representation updates."""
    def __init__(self, n_features, hidden=192, stages=6, dropout=.08):
        super().__init__(); self.stages = stages
        self.base = nn.Sequential(nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.mask = nn.ModuleList([nn.Linear(hidden, n_features) for _ in range(stages)])
        self.feature = nn.ModuleList([nn.Linear(n_features, hidden) for _ in range(stages)])
        self.update = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(stages)])
        self.reset = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(stages)])
        self.candidate = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(stages)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(stages)])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.base(x); masks = []
        for i in range(self.stages):
            # Direct feature masks are conditioned on the previous stage state.
            mask = torch.sigmoid(self.mask[i](h)); masks.append(mask)
            f = torch.tanh(self.feature[i](x * mask)); joint = torch.cat([h, f], 1)
            z = torch.sigmoid(self.update[i](joint)); r = torch.sigmoid(self.reset[i](joint))
            cand = torch.tanh(self.candidate[i](torch.cat([r * h, f], 1)))
            h = self.norm[i]((1 - z) * h + z * self.drop(cand))
        return h, torch.stack(masks, 1)


class GANDALFNet(nn.Module):
    def __init__(self, n_features, hidden=192):
        super().__init__(); self.gflu = GFLU(n_features, hidden=hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 96), nn.SiLU(), nn.Dropout(.1), nn.Linear(96, 1))
    def forward(self, x, return_masks=False):
        h, masks = self.gflu(x); pred = self.head(h).squeeze(1)
        return (pred, masks) if return_masks else pred


def predict(model, x, batch=16384):
    model.eval(); out = []
    with torch.no_grad():
        for q in range(0, len(x), batch):
            a = torch.from_numpy(x[q:q+batch]).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                out.append(model(a).float().cpu().numpy())
    return np.concatenate(out)


def run_fold(xfit, xval, xtest, yfit, fold, epochs=EPOCHS):
    enc = FoldEncoder().fit(xfit); a, b, c = enc.transform(xfit), enc.transform(xval), enc.transform(xtest)
    torch.manual_seed(SEED + fold); model = GANDALFNet(a.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=9e-4, weight_decay=4e-4)
    bs = 4096; steps = ((len(a) + bs - 1) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=9e-4, total_steps=steps, pct_start=.2)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda"); lossf = nn.BCEWithLogitsLoss()
    y = torch.from_numpy(np.asarray(yfit, dtype="float32").copy()); gen = torch.Generator().manual_seed(SEED + fold)
    for ep in range(epochs):
        model.train(); losses = []; perm = torch.randperm(len(a), generator=gen)
        for q in range(0, len(a), bs):
            ix = perm[q:q+bs]; xx, yy = torch.from_numpy(a[ix]).to(DEVICE), y[ix].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits, masks = model(xx, return_masks=True)
                # Very small entropy term discourages immediate all-open gates without forcing sparsity.
                entropy = -(masks.clamp(1e-5, 1-1e-5) * masks.clamp(1e-5, 1-1e-5).log() +
                            (1-masks).clamp(1e-5).log() * (1-masks)).mean()
                loss = lossf(logits, yy) + 2e-5 * entropy
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.detach()))
        print(f"fold {fold} epoch {ep+1}/{epochs} loss {np.mean(losses):.6f}", flush=True)
    pv, pt = predict(model, b), predict(model, c)
    model.eval()
    with torch.no_grad():
        _, masks = model(torch.from_numpy(a[:min(len(a), 32768)]).to(DEVICE), return_masks=True)
        importance = masks.float().mean((0, 1)).cpu().numpy()
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pv, pt, dict(zip(enc.names, map(float, importance)))


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 0):
        assert 'sm_60' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
    tp, sp = locate(); tr, te = pd.read_csv(tp), pd.read_csv(sp); cols = [c for c in te if c != ID]
    x, xt, y = engineer(tr[cols]), engineer(te[cols]), tr[TARGET].to_numpy("float32")
    oof = np.zeros(len(tr), "float32"); fid = np.zeros(len(tr), "int8"); tests, rows, imps = [], [], []
    started = time.time()
    for fold, (fit, val) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(x, y), 1):
        pv, pt, imp = run_fold(x.iloc[fit], x.iloc[val], xt, y[fit], fold)
        oof[val], fid[val] = pv, fold; tests.append(pt); imps.append(imp)
        rows.append({"fold": fold, "auc": roc_auc_score(y[val], pv)}); print(rows[-1], flush=True)
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("gpu_gandalf_gflu/artifacts")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: tr[ID], "fold": fid, "y": y, "pred": oof}).to_csv(out/"oof_gandalf_gflu.csv", index=False)
    pd.DataFrame({ID: te[ID], TARGET: np.mean(tests, axis=0)}).to_csv(out/"test_gandalf_gflu.csv", index=False)
    names = sorted(imps[0]); pd.DataFrame({"feature": names, "mean_gate": [np.mean([q[n] for q in imps]) for n in names]}).sort_values("mean_gate", ascending=False).to_csv(out/"gate_importance.csv", index=False)
    metrics = {"model": "independent GANDALF-style GFLU", "official_data_only": True, "outer_folds": FOLDS,
               "fixed_epochs": EPOCHS, "seed": SEED, "tuning": "none; fixed prespecified configuration",
               "leakage_contract": "target-free fold-local preprocessing; outer labels reporting-only",
               "fold_metrics": rows, "oof_auc": roc_auc_score(y, oof), "runtime_seconds": time.time()-started,
               "submission_created": False}
    (out/"metrics_gandalf_gflu.json").write_text(json.dumps(metrics, indent=2)); print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
