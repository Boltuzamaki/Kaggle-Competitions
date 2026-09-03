"""Nested-OOF supervised-evidence lookup Transformer; official data only.

Leakage contract
----------------
For each outer fold, supervised evidence for the outer-fit rows is inner OOF.
Evidence for outer-validation and test rows is fitted on the complete outer-fit
partition.  Outer-validation labels are used only once, for final reporting.
The architecture/training recipe is prespecified (one bounded candidate), and
training uses a fixed epoch count: there is no outer-validation model selection.
"""
from pathlib import Path
import subprocess, sys

if Path("/kaggle").exists():
    # PyTorch 2.5.1 cu121 contains sm_60 kernels required by Kaggle's P100.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--index-url",
        "https://download.pytorch.org/whl/cu121", "torch==2.5.1",
    ])

import gc, json, math, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED = "addicted_label", "id", 20260804
OUTER, INNER, EPOCHS = 5, 5, 18
SMOOTH_SINGLE, SMOOTH_PAIR = 24.0, 48.0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = torch.bfloat16 if DEV == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
SCALER = DEV == "cuda" and AMP == torch.float16


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
    raise FileNotFoundError("competition train.csv/test.csv not found")


def key(s):
    return s.astype("string").fillna("__NA__")


def deriv(d):
    parts = ["social_media_hours", "gaming_hours", "work_study_hours"]
    component = d[parts].sum(axis=1, min_count=3)
    other = d["daily_screen_time_hours"] - component
    return pd.DataFrame({
        "other_screen": other,
        "component_sum": component,
        "other_frac": other / (d["daily_screen_time_hours"] + .1),
        "weekend_gap": d["weekend_screen_time"] - d["daily_screen_time_hours"],
        "weekend_other": d["weekend_screen_time"] - component,
        "component_frac": component / (d["daily_screen_time_hours"] + .1),
    }, index=d.index)


PAIR_SPECS = [
    ("daily_screen_time_hours", "sleep_hours"),
    ("daily_screen_time_hours", "notifications_per_day"),
    ("daily_screen_time_hours", "app_opens_per_day"),
    ("daily_screen_time_hours", "stress_level"),
    ("daily_screen_time_hours", "academic_work_impact"),
    ("social_media_hours", "gaming_hours"),
    ("weekend_screen_time", "daily_screen_time_hours"),
]


def make_evidence_keys(raw):
    """Label-free keys. Values remain exact; no global frequency information."""
    out = {"single__" + c: key(raw[c]) for c in raw.columns}
    for a, b in PAIR_SPECS:
        if a in raw and b in raw:
            out[f"pair__{a}__{b}"] = key(raw[a]) + "\x1f" + key(raw[b])
    return pd.DataFrame(out, index=raw.index)


def map_evidence(fit_key, apply_key, fit_y, smooth):
    """Return smoothed rate and log support, fitted solely on fit_key/fit_y."""
    prior = float(np.mean(fit_y))
    stats = pd.DataFrame({"k": np.asarray(fit_key), "y": np.asarray(fit_y)}).groupby(
        "k", sort=False, observed=True
    )["y"].agg(["sum", "count"])
    sums = pd.Series(np.asarray(apply_key)).map(stats["sum"]).fillna(0).to_numpy("float64")
    counts = pd.Series(np.asarray(apply_key)).map(stats["count"]).fillna(0).to_numpy("float64")
    rate = (sums + smooth * prior) / (counts + smooth)
    return rate.astype("float32"), np.log1p(counts).astype("float32")


def fold_evidence(train_keys, test_keys, y, fit, val, fold):
    """Construct inner-OOF fit evidence and outer-fit mapped val/test evidence."""
    fit, val = np.asarray(fit), np.asarray(val)
    yf = np.asarray(y)[fit]
    ef = np.zeros((len(fit), 2 * train_keys.shape[1]), dtype="float32")
    ev = np.zeros((len(val), ef.shape[1]), dtype="float32")
    et = np.zeros((len(test_keys), ef.shape[1]), dtype="float32")
    inner = StratifiedKFold(INNER, shuffle=True, random_state=SEED + 100 * fold)
    splits = list(inner.split(fit, yf))
    coverage = np.zeros(len(fit), dtype="int8")
    for j, col in enumerate(train_keys):
        smooth = SMOOTH_PAIR if col.startswith("pair__") else SMOOTH_SINGLE
        for ia, ib in splits:
            rate, count = map_evidence(
                train_keys.iloc[fit[ia]][col], train_keys.iloc[fit[ib]][col], yf[ia], smooth
            )
            ef[ib, 2*j] = rate
            ef[ib, 2*j+1] = count
            if j == 0:
                coverage[ib] += 1
        ev[:, 2*j], ev[:, 2*j+1] = map_evidence(
            train_keys.iloc[fit][col], train_keys.iloc[val][col], yf, smooth
        )
        et[:, 2*j], et[:, 2*j+1] = map_evidence(
            train_keys.iloc[fit][col], test_keys[col], yf, smooth
        )
    assert np.all(coverage == 1), "inner OOF evidence coverage violated"
    return ef, ev, et


def lookups(fit, val, test):
    arrays = [np.zeros((len(z), fit.shape[1]), "int64") for z in (fit, val, test)]
    offsets, total = [], 0
    for j, c in enumerate(fit):
        vals = [v for v in key(fit[c]).unique() if v != "__NA__"]
        mapping = {v: i + 2 for i, v in enumerate(vals)}
        offsets.append(total)
        for out, z in zip(arrays, (fit, val, test)):
            kz = key(z[c])
            local = kz.map(mapping).fillna(1).to_numpy("int64")
            local[kz.eq("__NA__").to_numpy()] = 0
            out[:, j] = local + total
        total += len(vals) + 2
    return arrays, np.asarray(offsets, "int64"), total


def quantiles(fit, val, test):
    outs = [np.zeros((len(z), fit.shape[1]), "float32") for z in (fit, val, test)]
    masks = [z.isna().to_numpy("float32") for z in (fit, val, test)]
    for j, c in enumerate(fit):
        if not pd.api.types.is_numeric_dtype(fit[c]):
            continue
        ok = fit[c].notna()
        if not ok.any():
            continue
        qt = QuantileTransformer(
            n_quantiles=min(1000, int(ok.sum())), output_distribution="normal",
            subsample=300_000, random_state=SEED + j,
        ).fit(fit.loc[ok, [c]])
        for out, z in zip(outs, (fit, val, test)):
            good = z[c].notna()
            out[good.to_numpy(), j] = qt.transform(z.loc[good, [c]]).ravel().astype("float32")
    return outs, masks


class PLR(nn.Module):
    def __init__(self, n, k, d):
        super().__init__()
        self.f = nn.Parameter(torch.randn(n, k) * .5)
        self.w = nn.Parameter(torch.randn(n, 2*k, d) / math.sqrt(2*k))
        self.b = nn.Parameter(torch.zeros(n, d))

    def forward(self, x):
        z = 2 * math.pi * x.unsqueeze(-1) * self.f.unsqueeze(0)
        return torch.einsum("bfk,fkd->bfd", torch.cat([z.sin(), z.cos()], -1), self.w) + self.b


class Net(nn.Module):
    def __init__(self, total, nraw, nder, nev, d=128):
        super().__init__()
        self.emb = nn.Embedding(total, d)
        nn.init.normal_(self.emb.weight, std=.02)
        self.pc, self.pd = PLR(nraw, 20, d), PLR(nder, 20, d)
        # Each evidence key is one token containing rate and log-support.
        self.evidence = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.LayerNorm(d))
        self.evidence_pos = nn.Parameter(torch.randn(1, nev, d) * .02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, 1+nraw+nder, d) * .02)
        layer = nn.TransformerEncoderLayer(d, 8, d*2, .10, activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, 3)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(.1), nn.Linear(d, 1))

    def forward(self, ids, cont, mask, der, dmask, evidence):
        raw = self.emb(ids) + self.pc(cont) * (1-mask).unsqueeze(-1)
        derived = self.pd(der) * (1-dmask).unsqueeze(-1)
        ev = self.evidence(evidence.view(len(evidence), -1, 2)) + self.evidence_pos
        base = torch.cat([self.cls.expand(len(ids), -1, -1), raw, derived], 1) + self.pos
        return self.head(self.tr(torch.cat([base, ev], 1))[:, 0]).squeeze(-1)


def run_fold(fit, val, rawtr, rawte, dertr, derte, keys, test_keys, y, fold):
    (ii, iv, it), offsets, total = lookups(rawtr.iloc[fit], rawtr.iloc[val], rawte)
    (ci, cv, ct), (mi, mv, mt) = quantiles(rawtr.iloc[fit], rawtr.iloc[val], rawte)
    (di, dv, dt), (dmi, dmv, dmt) = quantiles(dertr.iloc[fit], dertr.iloc[val], derte)
    ei, ev, et = fold_evidence(keys, test_keys, y, fit, val, fold)
    tensor = lambda x: torch.from_numpy(x)
    data = list(map(tensor, (ii, iv, it, ci, cv, ct, mi, mv, mt, di, dv, dt, dmi, dmv, dmt, ei, ev, et)))
    II, IV, IT, CI, CV, CT, MI, MV, MT, DI, DV, DT, DMI, DMV, DMT, EI, EV, ET = data
    Y, OFF = tensor(y[fit]), tensor(offsets).to(DEV)
    torch.manual_seed(SEED + fold)
    model = Net(total, rawtr.shape[1], dertr.shape[1], keys.shape[1]).to(DEV)
    emb = [p for n, p in model.named_parameters() if n.startswith("emb")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("emb")]
    opt = torch.optim.AdamW([{"params": rest, "weight_decay": 1e-5}, {"params": emb, "weight_decay": 3e-4}], lr=1.8e-3)
    bs = 2048
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1.8e-3, total_steps=math.ceil(len(fit)/bs)*EPOCHS, pct_start=.15)
    scaler = torch.amp.GradScaler("cuda", enabled=SCALER)
    lossf, params = nn.BCEWithLogitsLoss(), list(model.parameters())
    gen = torch.Generator().manual_seed(SEED + fold)
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(fit), generator=gen)
        losses = []
        for q in range(0, len(fit), bs):
            sl = perm[q:q+bs]
            ids, mask = II[sl].to(DEV), MI[sl].to(DEV)
            drop = torch.rand(ids.shape, device=DEV) < .10
            ids, mask = torch.where(drop, OFF.expand_as(ids), ids), torch.maximum(mask, drop.float())
            with torch.autocast(device_type=DEV, dtype=AMP, enabled=DEV == "cuda"):
                logits = model(ids, CI[sl].to(DEV), mask, DI[sl].to(DEV), DMI[sl].to(DEV), EI[sl].to(DEV))
                loss = lossf(logits, Y[sl].to(DEV))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(float(loss.detach()))
        print(f"fold {fold} epoch {epoch+1}/{EPOCHS} train_loss {np.mean(losses):.6f}", flush=True)

    def predict(a, b, c, d, e, f):
        model.eval(); out = []
        with torch.no_grad():
            for q in range(0, len(a), 8192):
                with torch.autocast(device_type=DEV, dtype=AMP, enabled=DEV == "cuda"):
                    out.append(model(*(z[q:q+8192].to(DEV) for z in (a,b,c,d,e,f))).float().cpu())
        return torch.cat(out).numpy()

    pv = predict(IV, CV, MV, DV, DMV, EV)
    pt = predict(IT, CT, MT, DT, DMT, ET)
    del model, data
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    return pv, pt


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(torch.__version__, torch.cuda.get_arch_list(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEV)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 0):
        assert "sm_60" in torch.cuda.get_arch_list(), "wheel lacks P100 kernels"
    train_path, test_path = locate()
    tr, te = pd.read_csv(train_path), pd.read_csv(test_path)
    feats = [c for c in te.columns if c != ID]
    rawtr, rawte = tr[feats], te[feats]
    dertr, derte = deriv(rawtr), deriv(rawte)
    keys, test_keys = make_evidence_keys(rawtr), make_evidence_keys(rawte)
    y = tr[TARGET].to_numpy("float32")
    outer = StratifiedKFold(OUTER, shuffle=True, random_state=SEED)
    oof, fold_id, tests, rows = np.zeros(len(tr)), np.zeros(len(tr), "int8"), [], []
    started = time.time()
    for fold, (fit, val) in enumerate(outer.split(tr, y), 1):
        fold_id[val] = fold
        pv, pt = run_fold(fit, val, rawtr, rawte, dertr, derte, keys, test_keys, y, fold)
        oof[val], auc = pv, roc_auc_score(y[val], pv)
        tests.append(pt); rows.append({"fold": fold, "auc": auc})
        print(rows[-1], flush=True)
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("artifacts/local_lookup_v3_evidence")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: tr[ID], "fold": fold_id, "y": y, "pred": oof}).to_csv(out / "oof_lookup_v3_evidence.csv", index=False)
    pd.DataFrame({ID: te[ID], TARGET: np.mean(tests, axis=0)}).to_csv(out / "test_lookup_v3_evidence.csv", index=False)
    metrics = {
        "model": "nested-OOF supervised-evidence lookup Transformer",
        "official_data_only": True, "seed": SEED, "outer_folds": OUTER, "inner_folds": INNER,
        "hyperparameter_policy": "single prespecified bounded candidate; fixed epochs; no outer-validation selection",
        "evidence_contract": "outer-fit inner-OOF; outer-validation/test mapped from complete outer-fit only",
        "evidence_key_count": keys.shape[1], "fold_metrics": rows,
        "fold_mean": float(np.mean([r["auc"] for r in rows])), "fold_std": float(np.std([r["auc"] for r in rows])),
        "oof_auc": roc_auc_score(y, oof), "runtime_seconds": time.time()-started, "submission_created": False,
    }
    (out / "metrics_lookup_v3_evidence.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
