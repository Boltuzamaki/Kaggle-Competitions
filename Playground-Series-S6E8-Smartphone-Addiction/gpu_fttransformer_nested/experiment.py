"""Nested-tuned FT-Transformer; five-fold OOF, official competition data only."""
from pathlib import Path
import gc, json, math, random, subprocess, sys, time

if Path("/kaggle").exists():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--index-url",
                           "https://download.pytorch.org/whl/cu121", "torch==2.5.1"])

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED, FOLDS = "addicted_label", "id", 20260804, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else torch.float16


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
    raise FileNotFoundError("competition train/test not found")


def feature_frame(df):
    x = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    x["missing_count"] = x.isna().sum(axis=1)
    x["component_sum"] = x[parts].sum(axis=1, min_count=1)
    x["screen_residual"] = x["daily_screen_time_hours"] - x["component_sum"]
    x["weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["screen_sleep"] = x["daily_screen_time_hours"] / (x["sleep_hours"] + .1)
    x["notif_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 1.)
    return x


def encode(fit, *apply):
    """Learn every preprocessing statistic/vocabulary from fit only."""
    num = list(fit.select_dtypes(include=np.number).columns)
    cat = [c for c in fit if c not in num]
    nums, cats = [], []
    qts = {}
    for j, c in enumerate(num):
        good = fit[c].notna()
        qt = QuantileTransformer(n_quantiles=min(512, int(good.sum())), output_distribution="normal",
                                 subsample=250000, random_state=SEED + j)
        qt.fit(fit.loc[good, [c]])
        qts[c] = qt
    maps = {c: {v: i + 2 for i, v in enumerate(fit[c].astype("string").fillna("__NA__").unique())}
            for c in cat}
    for z in apply:
        a = np.zeros((len(z), len(num) * 2), dtype="float32")
        for j, c in enumerate(num):
            good = z[c].notna().to_numpy()
            if good.any():
                a[good, j] = qts[c].transform(z.loc[good, [c]]).ravel().astype("float32")
            a[:, len(num) + j] = ~good
        b = np.zeros((len(z), len(cat)), dtype="int64")
        for j, c in enumerate(cat):
            k = z[c].astype("string").fillna("__NA__")
            b[:, j] = k.map(maps[c]).fillna(1).to_numpy("int64")
        nums.append(a); cats.append(b)
    return nums, cats, [len(maps[c]) + 2 for c in cat]


class FTTransformer(nn.Module):
    def __init__(self, n_num, cardinalities, d=64, heads=8, layers=3, drop=.12):
        super().__init__()
        self.num_w = nn.Parameter(torch.randn(n_num, d) * .02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d))
        self.emb = nn.ModuleList([nn.Embedding(n, d) for n in cardinalities])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x_num, x_cat):
        tok = x_num.unsqueeze(-1) * self.num_w + self.num_b
        if len(self.emb):
            tok = torch.cat([tok, torch.stack([e(x_cat[:, j]) for j, e in enumerate(self.emb)], 1)], 1)
        cls = self.cls.expand(len(x_num), -1, -1)
        return self.head(self.transformer(torch.cat([cls, tok], 1))[:, 0]).squeeze(1)


def fit_predict(xf, cf, yf, xv, cv, yv, xt, ct, cards, cfg, seed, max_epochs, patience):
    torch.manual_seed(seed)
    model = FTTransformer(xf.shape[1], cards,
                          **{k: cfg[k] for k in ("d", "heads", "layers", "drop")}).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    loss_fn = nn.BCEWithLogitsLoss(); bs = cfg["batch"]
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda" and AMP_DTYPE == torch.float16)
    xf, cf, yf = map(torch.from_numpy, (xf, cf, yf.astype("float32")))
    xv, cv, xt, ct = map(torch.from_numpy, (xv, cv, xt, ct))
    best, best_state, best_epoch, stale = -1., None, 0, 0
    gen = torch.Generator().manual_seed(seed)
    def pred(a, b):
        model.eval(); out = []
        with torch.no_grad():
            for q in range(0, len(a), 16384):
                with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                    out.append(model(a[q:q+16384].to(DEVICE), b[q:q+16384].to(DEVICE)).float().cpu())
        return torch.cat(out).numpy()
    for epoch in range(max_epochs):
        model.train()
        for sl in torch.randperm(len(xf), generator=gen).split(bs):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE, enabled=DEVICE == "cuda"):
                loss = loss_fn(model(xf[sl].to(DEVICE), cf[sl].to(DEVICE)), yf[sl].to(DEVICE))
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        if yv is not None:
            auc = roc_auc_score(yv, pred(xv, cv))
            if auc > best + 1e-5:
                best, best_epoch, stale = auc, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= patience: break
    if yv is None:
        best_epoch = max_epochs
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pv, pt = pred(xv, cv), pred(xt, ct)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pv, pt, (float(best) if yv is not None else None), best_epoch


# d/layers/dropout are the architecture screen; optimizer keys are removed before construction.
CONFIGS = [
    dict(d=48, heads=6, layers=3, drop=.10, lr=8e-4, wd=2e-4, batch=2048),
    dict(d=64, heads=8, layers=4, drop=.15, lr=5e-4, wd=5e-4, batch=2048),
]
random.seed(SEED); np.random.seed(SEED)
tp, sp = locate(); train, test = pd.read_csv(tp), pd.read_csv(sp)
X, Xt = feature_frame(train), feature_frame(test); y = train[TARGET].to_numpy("float32")
outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train)); fold_id = np.zeros(len(train), "int8"); test_preds, rows = [], []
t0 = time.time()
for fold, (fi, vi) in enumerate(outer.split(X, y), 1):
    # The bounded screen uses only an inner holdout from this outer training partition.
    ia, ib = train_test_split(fi, test_size=.12, stratify=y[fi], random_state=SEED + fold)
    tune = []
    for ci, cfg in enumerate(CONFIGS):
        (ns, cs, cards) = encode(X.iloc[ia], X.iloc[ia], X.iloc[ib], X.iloc[ib[:1]])
        _, _, score, epoch = fit_predict(ns[0], cs[0], y[ia], ns[1], cs[1], y[ib],
                                         ns[2][:1], cs[2][:1], cards, cfg, SEED+100*fold+ci, 8, 2)
        tune.append({"config": ci, "inner_auc": score, "best_epoch": epoch})
    chosen = max(tune, key=lambda z: z["inner_auc"]); cfg = CONFIGS[chosen["config"]]
    ns, cs, cards = encode(X.iloc[fi], X.iloc[fi], X.iloc[vi], Xt)
    # Fixed epoch count comes only from inner tuning; outer validation is prediction-only.
    epochs = max(4, min(18, chosen["best_epoch"] + 2))
    pv, pt, _, _ = fit_predict(ns[0], cs[0], y[fi], ns[1], cs[1], None, ns[2], cs[2],
                               cards, cfg, SEED+fold, epochs, epochs+1)
    oof[vi], fold_id[vi] = pv, fold; test_preds.append(pt)
    auc = roc_auc_score(y[vi], pv)
    rows.append({"fold": fold, "auc": float(auc), "nested_screen": tune,
                 "chosen_config": chosen["config"], "final_epochs": epochs})
    print(rows[-1], flush=True)

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("gpu_fttransformer_nested/output")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({ID: train[ID], "fold": fold_id, "y": y, "pred": oof}).to_csv(out/"oof_fttransformer.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: np.mean(test_preds, axis=0)}).to_csv(out/"test_fttransformer.csv", index=False)
metrics = {"model": "nested-tuned FT-Transformer", "official_data_only": True, "seed": SEED,
           "folds": FOLDS, "selection_protocol": "two configs on outer-train-only inner holdout",
           "fold_metrics": rows, "oof_auc": float(roc_auc_score(y, oof)),
           "runtime_seconds": time.time()-t0, "submission_created": False}
(out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
print(json.dumps(metrics, indent=2), flush=True)
