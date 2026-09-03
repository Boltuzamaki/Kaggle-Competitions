"""Memory-efficient TabR-S with outer-fold-safe bounded tuning; official data only."""
from pathlib import Path
import gc, hashlib, json, subprocess, sys, time

if Path("/kaggle").exists():
    # Install one mutually compatible CUDA stack first.  Installing/changing torch
    # after PyTabKit can leave torchvision/torchmetrics linked to a different ABI.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--index-url",
                           "https://download.pytorch.org/whl/cu121", "torch==2.5.1",
                           "torchvision==0.20.1", "torchaudio==2.5.1"])
    # TabR's optional runtime (notably skorch/lightning) lives in the models extra.
    # --no-deps prevents it from replacing the verified P100-capable torch stack.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                           "pytabkit==1.7.3"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "pytorch-lightning==2.5.2", "skorch==1.1.0",
                           "faiss-cpu==1.12.0"])

import numpy as np
import pandas as pd
import torch
import faiss
from pytabkit import TabR_S_D_Classifier
import pytabkit.models.alg_interfaces.tabr_interface as _tabr_interface
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


# sklearn 1.8+ drops an all-observed categorical column when TabR temporarily
# marks category code zero as missing.  Preserve the columns as older sklearn
# versions did; this is preprocessing compatibility, not a model change.
class _KeepEmptySimpleImputer(_SimpleImputer):
    def __init__(self, *, missing_values=np.nan, strategy="mean", fill_value=None,
                 copy=True, add_indicator=False, keep_empty_features=True):
        super().__init__(missing_values=missing_values, strategy=strategy,
                         fill_value=fill_value, copy=copy,
                         add_indicator=add_indicator,
                         keep_empty_features=keep_empty_features)


_tabr_interface.SimpleImputer = _KeepEmptySimpleImputer


# Keep this bridge inline because Kaggle script kernels upload only code_file.
class TorchExactL2Index:
    def __init__(self, _resources=None, dimension=None, _config=None, chunk_size=65536):
        self.dimension, self.chunk_size, self.candidates = int(dimension), int(chunk_size), None

    def reset(self): self.candidates = None

    def add(self, candidates):
        x = torch.from_numpy(candidates) if isinstance(candidates, np.ndarray) else candidates
        if x.ndim != 2 or x.shape[1] != self.dimension: raise ValueError("candidate shape")
        self.candidates = x.detach().contiguous().float()

    @torch.no_grad()
    def search(self, queries, k):
        as_numpy = isinstance(queries, np.ndarray)
        q = torch.from_numpy(queries) if as_numpy else queries
        q = q.detach().contiguous().float()
        if self.candidates is None: raise RuntimeError("add before search")
        if q.device != self.candidates.device: raise ValueError("device mismatch")
        k = min(int(k), len(self.candidates)); qn = q.square().sum(1, keepdim=True)
        bd = torch.full((len(q), k), torch.inf, device=q.device)
        bi = torch.full((len(q), k), -1, dtype=torch.long, device=q.device)
        for start in range(0, len(self.candidates), self.chunk_size):
            c = self.candidates[start:start+self.chunk_size]
            d = (qn + c.square().sum(1)[None] - 2 * (q @ c.T)).clamp_min_(0)
            cd, ci = torch.topk(d, min(k, len(c)), largest=False, sorted=False)
            md, mi = torch.cat((bd, cd), 1), torch.cat((bi, ci + start), 1)
            bd, pos = torch.topk(md, k, largest=False, sorted=True); bi = mi.gather(1, pos)
        return (bd.cpu().numpy(), bi.cpu().numpy()) if as_numpy else (bd, bi)


class _GpuResources: pass
class _GpuConfig: device = 0
faiss.StandardGpuResources = _GpuResources
faiss.GpuIndexFlatConfig = _GpuConfig
faiss.GpuIndexFlatL2 = TorchExactL2Index

TARGET, ID, SEED, FOLDS = "addicted_label", "id", 20260804, 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def locate():
    for root in (Path("/kaggle/input"), Path(".")):
        if not root.exists(): continue
        for p in root.rglob("train.csv"):
            try: cols = pd.read_csv(p, nrows=1).columns
            except Exception: continue
            if TARGET in cols and (p.parent / "test.csv").exists(): return p, p.parent / "test.csv"
    raise FileNotFoundError("official competition tables not found")


def features(df):
    x = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    miss = x.isna()
    # The official columns are all parsed as numeric.  PyTabKit TabR 1.7.3
    # nevertheless assumes at least one categorical feature internally.  A
    # target-free missingness fingerprint is both meaningful for retrieval and
    # prevents its zero-column OrdinalEncoder failure.
    raw_cols = list(x.columns)
    x["missing_pattern"] = miss[raw_cols].astype("uint8").astype(str).agg("".join, axis=1)
    for c in x.select_dtypes(include=np.number):
        x[c + "__na"] = miss[c].astype("float32")
    for c in x.select_dtypes(exclude=np.number):
        # Plain object strings are intentional: pandas' nullable StringDtype is
        # mishandled by PyTabKit 1.7.3's sklearn converter on concatenated
        # explicit-validation frames.
        x[c] = x[c].fillna("__NA__").astype(object)
    p = ["social_media_hours", "gaming_hours", "work_study_hours"]
    x["missing_count"] = miss.sum(axis=1).astype("float32")
    x["component_sum"] = x[p].sum(axis=1, min_count=1)
    x["screen_residual"] = x["daily_screen_time_hours"] - x["component_sum"]
    x["weekend_residual"] = x["weekend_screen_time"] - x["component_sum"]
    x["weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["screen_sleep"] = x["daily_screen_time_hours"] / (x["sleep_hours"] + .1)
    x["notif_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 1.)
    return x


def complete(fit, *apply):
    """Continuous completion values are learned from fit only."""
    numeric = fit.select_dtypes(include=np.number).columns
    med = fit[numeric].replace([np.inf, -np.inf], np.nan).median().fillna(0)
    out = []
    for z in apply:
        z = z.copy()
        z[numeric] = z[numeric].replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0)
        out.append(z)
    return out


# Two small, credible retrieval settings. Screening is repeated inside every outer fold.
CONFIGS = [
    dict(d_main=128, d_multiplier=2.0, encoder_n_blocks=1, predictor_n_blocks=2,
         context_size=64, context_dropout=.10, dropout0=.10, dropout1=0.0),
    dict(d_main=192, d_multiplier=2.0, encoder_n_blocks=1, predictor_n_blocks=2,
         context_size=96, context_dropout=.15, dropout0=.15, dropout1=.05),
]


def model(cfg, seed, epochs, patience, val_fraction=.12):
    return TabR_S_D_Classifier(
        device="cuda", random_state=seed, n_cv=1, n_refit=0, val_fraction=val_fraction,
        val_metric_name="cross_entropy", verbosity=1, n_threads=4,
        n_epochs=epochs, patience=patience, batch_size=1024, eval_batch_size=8192,
        memory_efficient=True, candidate_encoding_batch_size=32768,
        num_embeddings=None, normalization="LayerNorm", activation="ReLU",
        optimizer={"type": "AdamW", "lr": 8e-4, "weight_decay": 1e-5}, **cfg)


tp, sp = locate(); train, test = pd.read_csv(tp), pd.read_csv(sp)
X, XT = features(train), features(test); y = train[TARGET].to_numpy("int64")
CAT_COLS = list(X.select_dtypes(exclude=np.number).columns)
assert "missing_pattern" in CAT_COLS and CAT_COLS == list(XT.select_dtypes(exclude=np.number).columns)
outer = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train)); fold_ids = np.zeros(len(train), "int8"); tests, rows = [], []
t0 = time.time()
if torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (6, 0):
    assert "sm_60" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()

for fold, (fi, vi) in enumerate(outer.split(X, y), 1):
    # Cap only the tuning workload. Sampling and validation remain inside outer fit.
    tune_pool = fi
    if len(tune_pool) > 200000:
        tune_pool, _ = train_test_split(tune_pool, train_size=200000, stratify=y[tune_pool],
                                        random_state=SEED + fold)
    ia, ib = train_test_split(tune_pool, test_size=.15, stratify=y[tune_pool], random_state=SEED+10+fold)
    xa, xb = complete(X.iloc[ia], X.iloc[ia], X.iloc[ib])
    screen = []
    for ci, cfg in enumerate(CONFIGS):
        m = model(cfg, SEED + 100*fold + ci, epochs=8, patience=2)
        # Explicit validation is inner-only and may select early stopping here.
        m.fit(xa, y[ia], xb, y[ib], cat_col_names=CAT_COLS)
        p = m.predict_proba(xb)[:, 1]; score = roc_auc_score(y[ib], p)
        screen.append({"config": ci, "inner_auc": float(score)})
        del m; gc.collect(); torch.cuda.empty_cache()
    chosen = max(screen, key=lambda z: z["inner_auc"]); cfg = CONFIGS[chosen["config"]]

    xf, xv, xt = complete(X.iloc[fi], X.iloc[fi], X.iloc[vi], XT)
    # No outer validation is supplied to fit: early stopping uses only an internal
    # split of the outer-fit partition. Outer labels are prediction-only.
    m = model(cfg, SEED + fold, epochs=30, patience=5)
    m.fit(xf, y[fi], cat_col_names=CAT_COLS)
    vp, pt = m.predict_proba(xv)[:, 1], m.predict_proba(xt)[:, 1]
    oof[vi], fold_ids[vi] = vp, fold; tests.append(pt)
    auc = roc_auc_score(y[vi], vp)
    rows.append({"fold": fold, "auc": float(auc), "screen": screen,
                 "chosen_config": chosen["config"]})
    print(rows[-1], flush=True)
    del m, xf, xv, xt, xa, xb; gc.collect(); torch.cuda.empty_cache()

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("gpu_tabr_nested/output")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({ID: train[ID], "fold": fold_ids, "y": y, "pred": oof}).to_csv(out/"oof_tabr.csv", index=False)
pd.DataFrame({ID: test[ID], TARGET: np.mean(tests, axis=0)}).to_csv(out/"test_tabr.csv", index=False)
metrics = {"model": "PyTabKit TabR-S-D memory-efficient", "library": "pytabkit==1.7.3",
           "official_data_only": True, "seed": SEED, "folds": FOLDS,
           "train_sha256": sha256(tp), "test_sha256": sha256(sp),
           "train_rows": len(train), "test_rows": len(test),
           "selection_protocol": "two configs on <=200k outer-fit-only inner split per fold",
           "outer_validation_labels_used_for_selection": False,
           "fold_metrics": rows, "oof_auc": float(roc_auc_score(y, oof)),
           "runtime_seconds": time.time()-t0, "submission_created": False}
(out/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
print(json.dumps(metrics, indent=2), flush=True)
