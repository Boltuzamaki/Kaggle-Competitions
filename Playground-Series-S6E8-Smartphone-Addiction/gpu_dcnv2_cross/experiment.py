"""Leakage-safe DCNv2-style cross network on official S6E8 data.

This is deliberately target-free feature engineering. All imputers, scalers and
categorical vocabularies are fitted inside each outer training fold. The model
configuration and epoch count are fixed before CV; validation labels report AUC
only and never select checkpoints or hyperparameters.
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

TARGET, ID, SEED, FOLDS, EPOCHS = "addicted_label", "id", 6082041, 5, 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16


def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists():
            continue
        for p in root.rglob("train.csv"):
            try:
                cols = pd.read_csv(p, nrows=1).columns
            except Exception:
                continue
            if TARGET in cols and (p.parent / "test.csv").exists():
                return p, p.parent / "test.csv"
    raise FileNotFoundError("official train.csv/test.csv not found")


def engineer(df):
    """Prespecified target-free features expressing usage composition."""
    z = df.copy()
    eps = 0.5
    def ratio(name, a, b):
        if a in z and b in z:
            z[name] = z[a] / (z[b].abs() + eps)
    ratio("social_share", "social_media_hours", "daily_screen_time_hours")
    ratio("gaming_share", "gaming_hours", "daily_screen_time_hours")
    ratio("weekend_uplift", "weekend_screen_time", "daily_screen_time_hours")
    ratio("opens_per_screen_hour", "app_opens_per_day", "daily_screen_time_hours")
    ratio("notifications_per_open", "notifications_per_day", "app_opens_per_day")
    ratio("screen_to_sleep", "daily_screen_time_hours", "sleep_hours")
    ratio("screen_to_productive", "daily_screen_time_hours", "work_study_hours")
    if {"social_media_hours", "gaming_hours"} <= set(z):
        z["entertainment_hours"] = z.social_media_hours + z.gaming_hours
    if {"daily_screen_time_hours", "sleep_hours", "work_study_hours"} <= set(z):
        z["accounted_day_hours"] = z.daily_screen_time_hours + z.sleep_hours + z.work_study_hours
        z["unaccounted_day_hours"] = 24.0 - z.accounted_day_hours
    base = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
    z["missing_count"] = df.isna().sum(axis=1).astype("float32")
    for c in base:
        z[c + "__missing"] = df[c].isna().astype("float32")
    return z


class FoldEncoder:
    """Fold-local quantiles and categorical vocabularies; no target access."""
    def fit(self, x):
        self.num = [c for c in x if pd.api.types.is_numeric_dtype(x[c])]
        self.cat = [c for c in x if c not in self.num]
        self.medians = x[self.num].median()
        filled = x[self.num].fillna(self.medians)
        self.qt = QuantileTransformer(n_quantiles=min(1000, len(x)), output_distribution="normal",
                                      subsample=250000, random_state=SEED).fit(filled)
        self.maps = {}
        for c in self.cat:
            vals = x[c].astype("string").fillna("__NA__").unique()
            self.maps[c] = {v: i + 1 for i, v in enumerate(vals)}
        self.cards = [len(self.maps[c]) + 1 for c in self.cat]
        return self

    def transform(self, x):
        num = self.qt.transform(x[self.num].fillna(self.medians)).astype("float32")
        cats = np.zeros((len(x), len(self.cat)), dtype="int64")
        for j, c in enumerate(self.cat):
            cats[:, j] = x[c].astype("string").fillna("__NA__").map(self.maps[c]).fillna(0).to_numpy("int64")
        return num, cats


class CrossNetMix(nn.Module):
    """DCNv2 low-rank mixture cross layers with input-dependent expert gates."""
    def __init__(self, d, layers=3, experts=4, rank=24):
        super().__init__()
        self.layers, self.experts = layers, experts
        self.u = nn.ModuleList([nn.ModuleList([nn.Linear(rank, d, bias=False) for _ in range(experts)]) for _ in range(layers)])
        self.v = nn.ModuleList([nn.ModuleList([nn.Linear(d, rank, bias=False) for _ in range(experts)]) for _ in range(layers)])
        self.c = nn.ModuleList([nn.ModuleList([nn.Linear(rank, rank, bias=False) for _ in range(experts)]) for _ in range(layers)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(layers)])
        self.gate = nn.ModuleList([nn.Linear(d, experts) for _ in range(layers)])

    def forward(self, x0):
        x = x0
        for l in range(self.layers):
            weights = self.gate[l](x).softmax(-1)
            expert = []
            for e in range(self.experts):
                h = torch.tanh(self.v[l][e](x))
                h = torch.tanh(self.c[l][e](h))
                expert.append(x0 * (self.u[l][e](h) + self.bias[l]))
            mixed = torch.stack(expert, 1)
            x = x + (mixed * weights.unsqueeze(-1)).sum(1)
        return x


class DCNv2(nn.Module):
    def __init__(self, nnum, cards, d=128):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, 8) for c in cards])
        raw = nnum + 8 * len(cards)
        self.input = nn.Sequential(nn.Linear(raw, d), nn.LayerNorm(d), nn.SiLU())
        self.cross = CrossNetMix(d, layers=3, experts=4, rank=24)
        self.deep = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(.12),
                                  nn.Linear(256, 128), nn.SiLU(), nn.Dropout(.08))
        self.head = nn.Sequential(nn.LayerNorm(d + 128), nn.Linear(d + 128, 1))

    def forward(self, num, cat):
        parts = [num] + [emb(cat[:, j]) for j, emb in enumerate(self.emb)]
        x0 = self.input(torch.cat(parts, 1))
        return self.head(torch.cat([self.cross(x0), self.deep(x0)], 1)).squeeze(1)


def predict(model, num, cat, batch=16384):
    model.eval(); out = []
    with torch.no_grad():
        for q in range(0, len(num), batch):
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                p = model(torch.from_numpy(num[q:q+batch]).to(DEVICE), torch.from_numpy(cat[q:q+batch]).to(DEVICE))
            out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def run_fold(xfit, xval, xtest, yfit, fold, epochs=EPOCHS):
    enc = FoldEncoder().fit(xfit)
    nf, cf = enc.transform(xfit); nv, cv = enc.transform(xval); nt, ct = enc.transform(xtest)
    torch.manual_seed(SEED + fold)
    model = DCNv2(nf.shape[1], enc.cards).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=3e-4)
    bs = 4096
    steps = ((len(nf) + bs - 1) // bs) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.2e-3, total_steps=steps, pct_start=.18)
    lossf = nn.BCEWithLogitsLoss(); gen = torch.Generator().manual_seed(SEED + fold)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    yn = torch.from_numpy(np.asarray(yfit, dtype="float32"))
    for ep in range(epochs):
        model.train(); losses = []
        perm = torch.randperm(len(nf), generator=gen)
        for q in range(0, len(nf), bs):
            ix = perm[q:q+bs]
            a = torch.from_numpy(nf[ix]).to(DEVICE); b = torch.from_numpy(cf[ix]).to(DEVICE); yy = yn[ix].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                loss = lossf(model(a, b), yy)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 2.)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.detach()))
        print(f"fold {fold} epoch {ep+1}/{epochs} loss {np.mean(losses):.6f}", flush=True)
    pv, pt = predict(model, nv, cv), predict(model, nt, ct)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pv, pt


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 0):
        assert 'sm_60' in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
    tp, sp = locate(); train, test = pd.read_csv(tp), pd.read_csv(sp)
    cols = [c for c in test if c != ID]
    x, xt = engineer(train[cols]), engineer(test[cols]); y = train[TARGET].to_numpy("float32")
    oof = np.zeros(len(train), "float32"); fold_id = np.zeros(len(train), "int8"); tests, rows = [], []
    started = time.time()
    for fold, (fit, val) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(x, y), 1):
        pv, pt = run_fold(x.iloc[fit], x.iloc[val], xt, y[fit], fold)
        oof[val], fold_id[val] = pv, fold; tests.append(pt)
        rows.append({"fold": fold, "auc": roc_auc_score(y[val], pv)})
        print(rows[-1], flush=True)
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("gpu_dcnv2_cross/artifacts")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: train[ID], "fold": fold_id, "y": y, "pred": oof}).to_csv(out / "oof_dcnv2_cross.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: np.mean(tests, axis=0)}).to_csv(out / "test_dcnv2_cross.csv", index=False)
    metrics = {"model": "DCNv2 low-rank mixture cross network", "official_data_only": True,
               "outer_folds": FOLDS, "seed": SEED, "fixed_epochs": EPOCHS,
               "feature_policy": "target-free prespecified composition features; fold-local preprocessing",
               "hyperparameter_policy": "single fixed configuration; outer labels reporting-only",
               "fold_metrics": rows, "oof_auc": roc_auc_score(y, oof),
               "runtime_seconds": time.time() - started, "submission_created": False}
    (out / "metrics_dcnv2_cross.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
