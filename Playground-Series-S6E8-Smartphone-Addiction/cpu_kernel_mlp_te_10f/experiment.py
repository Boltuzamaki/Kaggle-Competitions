"""A neural network that is allowed to see target statistics.

Every neural experiment in this project so far was target-free: DCNv2 0.9385,
GANDALF 0.9387, FT-Transformer 0.9401, TabR 0.9380. All four land about 0.030
below the models that see target information, and the ledger's own conclusion is
that this dataset requires target statistics. The one neural family that works
is the exact-value lookup transformer, which gets those statistics implicitly
through its embeddings and is also the only family materially decorrelated from
the trees (0.9756 rank correlation, against 0.99+ among the GBDTs).

Nobody has yet trained a plain network on the *explicit* fold-safe target
encoding. The gap is worth closing, because the two facts that make it
interesting point in the same direction:

  - the additive logistic model on exactly these features reached 0.9578 (E052),
    so the features carry the signal and the missing 0.010 is interaction
    structure that a linear model cannot express and an MLP can;
  - a network fits a globally smooth function, where a GBDT fits axis-aligned
    steps, so even at equal accuracy the two are wrong in different places.

The encoder is the validated fold-safe one, byte-for-byte the same contract as
cpu_kernel_lgb10fold: fit rows get inner out-of-fold rates, outer-validation and
test rows are mapped from the complete outer-fit partition. Any difference in
the output therefore comes from the learner and not from the features.

Seeds are averaged inside each fold because a network's run-to-run spread is far
wider than a boosted tree's.

The first run was deliberately conservative at five folds: a local timing probe
suggested ten folds would overrun the twelve-hour limit, and a complete five-fold
stream is worth more than a ten-fold run that dies at fold seven with an OOF the
stack cannot use. That probe was wrong by roughly a factor of seven, because it
was measured on a machine under load average 32 while a Kaggle kernel gets four
dedicated cores: the real run took 50 minutes. The variants below spend the
headroom that measurement revealed.

Official competition data only. No public predictions and no submission call.
"""
from pathlib import Path
import argparse
import sys
import gc
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET, ID = "addicted_label", "id"
SEED, INNER_FOLDS = 20260807, 5
SMOOTHING = 40.0
# One entry per kernel. The first run (mlp_te, 5 folds, 2 seeds, 512-256-128)
# scored 0.9661121 in 50 minutes, far under the 12-hour limit, so the follow-ups
# spend that headroom two ways: `mlp_te_10f` buys accuracy with the fold and seed
# levers, and `mlp_te_wide` buys decorrelation with a different shape of network
# rather than a better one.
VARIANTS = {
    "mlp_te":      {"folds": 5,  "seeds": 2, "epochs": 18,
                    "widths": ((512, 0.30), (256, 0.20), (128, 0.10))},
    "mlp_te_10f":  {"folds": 10, "seeds": 3, "epochs": 22,
                    "widths": ((512, 0.30), (256, 0.20), (128, 0.10))},
    "mlp_te_wide": {"folds": 10, "seeds": 2, "epochs": 20,
                    "widths": ((1024, 0.40), (512, 0.25), (256, 0.15), (128, 0.05))},
}
VARIANT = "mlp_te_10f"  # generated; edit the source, not this copy

CFG = VARIANTS[VARIANT]
OUTER_FOLDS = CFG["folds"]
NET_SEEDS = CFG["seeds"]

PAIR_COLUMNS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("daily_screen_time_hours", "gaming_hours"),
    ("sleep_hours", "stress_level"),
]


def locate():
    for root in (Path("/kaggle/input"), Path(".."), Path(".")):
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


def base_features(frame):
    x = frame.drop(columns=[ID, TARGET], errors="ignore").copy()
    numeric = list(x.select_dtypes(include="number").columns)
    x["missing_count"] = x.isna().sum(axis=1).astype("int8")
    for column in numeric:
        x[column + "__missing"] = x[column].isna().astype("int8")
    x["leisure_hours"] = x["social_media_hours"] + x["gaming_hours"]
    x["accounted_hours"] = x["leisure_hours"] + x["work_study_hours"]
    x["unaccounted_screen"] = x["daily_screen_time_hours"] - x["accounted_hours"]
    x["weekend_unaccounted"] = x["weekend_screen_time"] - x["accounted_hours"]
    x["weekend_gap"] = x["weekend_screen_time"] - x["daily_screen_time_hours"]
    x["weekend_ratio"] = x["weekend_screen_time"] / (x["daily_screen_time_hours"] + 0.25)
    x["leisure_share"] = x["leisure_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["social_share"] = x["social_media_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["gaming_share"] = x["gaming_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["work_share"] = x["work_study_hours"] / (x["daily_screen_time_hours"] + 0.25)
    x["notif_per_screen"] = x["notifications_per_day"] / (x["daily_screen_time_hours"] + 0.25)
    x["opens_per_screen"] = x["app_opens_per_day"] / (x["daily_screen_time_hours"] + 0.25)
    x["notif_per_open"] = x["notifications_per_day"] / (x["app_opens_per_day"] + 2.0)
    x["sleep_screen_balance"] = x["sleep_hours"] - x["daily_screen_time_hours"]
    x["screen_sleep_ratio"] = x["daily_screen_time_hours"] / (x["sleep_hours"] + 0.25)
    for column in x.select_dtypes(exclude="number").columns:
        x[column] = x[column].astype("category").cat.codes.astype("int16")
    # The tree version also carries a `missing_pattern` bitmask. It is dropped
    # here: an integer whose bits are meaningful but whose magnitude is not is
    # readable by an axis-aligned split and pure noise to a smooth network. The
    # per-column flags plus missing_count carry the same information in a form
    # a network can actually use.
    return x.replace([np.inf, -np.inf], np.nan)


def key_of(frame, column, digits=None):
    s = frame[column]
    if digits is not None and pd.api.types.is_numeric_dtype(s):
        s = s.round(digits)
    return s.astype("string").fillna("__NA__")


def pair_key(frame, left, right, digits=None):
    return key_of(frame, left, digits) + "|" + key_of(frame, right, digits)


def rate_map(keys, y, prior):
    stats = pd.DataFrame({"k": keys.to_numpy(), "y": y}).groupby("k").y.agg(["sum", "count"])
    return (stats["sum"] + SMOOTHING * prior) / (stats["count"] + SMOOTHING)


def inner_folds(n, seed):
    order = np.random.default_rng(seed).permutation(n)
    for k in range(INNER_FOLDS):
        va = order[k::INNER_FOLDS]
        yield np.setdiff1d(order, va, assume_unique=True), va


def build(fit_raw, fit_y, apply_raw, apply_base, raw_columns, inner_oof):
    """Fold-safe encoder, identical contract to cpu_kernel_lgb10fold."""
    prior = float(np.mean(fit_y))
    cols = {}
    for column in raw_columns:
        counts = key_of(fit_raw, column).value_counts()
        cols[column + "__logfreq"] = np.log1p(
            key_of(apply_raw, column).map(counts).fillna(0)).to_numpy("float32")
        if pd.api.types.is_numeric_dtype(fit_raw[column]):
            rc = key_of(fit_raw, column, 1).value_counts()
            cols[column + "__rlogfreq"] = np.log1p(
                key_of(apply_raw, column, 1).map(rc).fillna(0)).to_numpy("float32")
    for left, right in PAIR_COLUMNS:
        for tag, digits in (("exact", None), ("rounded", 1)):
            counts = pair_key(fit_raw, left, right, digits).value_counts()
            cols[f"{left}__{right}__{tag}_lf"] = np.log1p(
                pair_key(apply_raw, left, right, digits).map(counts).fillna(0)).to_numpy("float32")

    specs = [(c,) for c in raw_columns] + [p for p in PAIR_COLUMNS]
    for spec in specs:
        name = "__".join(spec) + "__te"
        fk = key_of(fit_raw, spec[0]) if len(spec) == 1 else pair_key(fit_raw, *spec)
        if inner_oof:
            values = np.full(len(apply_raw), np.nan, dtype="float32")
            for tr_i, va_i in inner_folds(len(fk), SEED):
                r = rate_map(fk.iloc[tr_i], fit_y[tr_i], prior)
                values[va_i] = fk.iloc[va_i].map(r).fillna(prior).to_numpy("float32")
            cols[name] = values
        else:
            ak = key_of(apply_raw, spec[0]) if len(spec) == 1 else pair_key(apply_raw, *spec)
            cols[name] = ak.map(rate_map(fk, fit_y, prior)).fillna(prior).to_numpy("float32")
    return pd.concat([apply_base, pd.DataFrame(cols, index=apply_base.index)], axis=1)


# --- the network ----------------------------------------------------------

def make_net(torch, n_features, seed):
    """Three hidden layers with batch norm and dropout.

    Width is set by the input: the encoded view is about 100 columns of which
    roughly a fifth are target rates, so a first layer of 512 gives the network
    enough room to form pairwise products of them without being so wide that a
    CPU fold becomes the bottleneck.
    """
    nn = torch.nn
    torch.manual_seed(seed)
    layers, prev = [], n_features
    for width, drop in CFG["widths"]:
        layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop)]
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def fit_mlp(x_fit, fit_y, x_val, x_test, fold, threads, epochs=None, batch=8192):
    epochs = CFG["epochs"] if epochs is None else epochs
    import torch
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from sklearn.pipeline import make_pipeline

    torch.set_num_threads(threads)
    prep = make_pipeline(
        SimpleImputer(strategy="median"),
        QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                            subsample=200000, random_state=SEED + fold),
        StandardScaler(),
    )
    a_fit = prep.fit_transform(x_fit).astype("float32")
    a_val = prep.transform(x_val).astype("float32")
    a_test = prep.transform(x_test).astype("float32")

    t_fit = torch.from_numpy(a_fit)
    t_y = torch.from_numpy(fit_y.astype("float32")).unsqueeze(1)
    t_val = torch.from_numpy(a_val)
    t_test = torch.from_numpy(a_test)

    val_acc = np.zeros(len(a_val), dtype="float64")
    test_acc = np.zeros(len(a_test), dtype="float64")
    seed_started = time.time()

    for s in range(NET_SEEDS):
        # Batches are large (8192) because on four CPU cores the per-step
        # overhead, not the matrix multiply, is what costs; the peak learning
        # rate is raised to match the reduced number of optimizer steps.
        net = make_net(torch, a_fit.shape[1], SEED + 101 * fold + s)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
        n_batches = (len(t_fit) + batch - 1) // batch
        # OneCycleLR divides by the length of its annealing phase, which rounds
        # to zero when there are only a handful of steps. That never happens on
        # the full data (hundreds of batches per epoch) but does on a row-capped
        # smoke run, so the floor keeps the smoke path exercising the real code.
        total_steps = max(epochs * n_batches, 16)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-3, total_steps=total_steps, pct_start=0.25)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(SEED + 7919 * fold + s)

        net.train()
        for epoch in range(epochs):
            order = torch.randperm(len(t_fit), generator=gen)
            for b in range(n_batches):
                idx = order[b * batch:(b + 1) * batch]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(t_fit[idx]), t_y[idx])
                loss.backward()
                opt.step()
                sched.step()

        net.eval()
        with torch.no_grad():
            val_acc += torch.sigmoid(
                torch.cat([net(t_val[i:i + 16384]) for i in range(0, len(t_val), 16384)])
            ).squeeze(1).numpy()
            test_acc += torch.sigmoid(
                torch.cat([net(t_test[i:i + 16384]) for i in range(0, len(t_test), 16384)])
            ).squeeze(1).numpy()
        # No validation label reaches this function, so there is no early
        # stopping and nothing to select: the seed loop just reports progress.
        print(f"    fold {fold} net seed {s} trained ({time.time()-seed_started:.0f}s)",
              flush=True)

    return val_acc / NET_SEEDS, test_acc / NET_SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--folds", type=int, default=OUTER_FOLDS)
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--rows", type=int, default=0, help="smoke test row cap")
    ap.add_argument("--total-budget", type=float, default=10.5 * 3600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    n_folds = args.folds
    out_dir = Path(args.out) if args.out else (
        Path("/kaggle/working") if Path("/kaggle").exists()
        else Path(__file__).parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    train_path, test_path = locate()
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if args.rows:
        # Stratified cap by positional index. `groupby(...).apply(...)` is not
        # used because whether it keeps the grouping column differs between the
        # pandas 2 that Kaggle runs and the pandas 3 installed locally.
        per_class = max(1, args.rows // 2)
        take = np.concatenate([
            np.flatnonzero(train[TARGET].to_numpy() == c)[:per_class] for c in (0, 1)])
        train = train.iloc[np.sort(take)].reset_index(drop=True)
        test = test.head(min(len(test), args.rows // 2)).reset_index(drop=True)

    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    y = train[TARGET].to_numpy("int8")
    train_base, test_base = base_features(train), base_features(test)
    train_raw, test_raw = train[raw_columns], test[raw_columns]

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    records = []
    folds_done = 0
    outer = StratifiedKFold(n_folds, shuffle=True, random_state=SEED)
    outer_splits = list(outer.split(train_base, y))

    for fold, (fit_i, val_i) in enumerate(outer_splits, 1):
        # A fold that cannot finish inside the kernel's wall clock would be
        # killed mid-write, so the loop stops while there is still time to
        # emit artifacts for the folds that did complete.
        remaining = args.total_budget - (time.time() - started)
        if folds_done and remaining < (time.time() - started) / folds_done:
            print(f"[{VARIANT}] stopping before fold {fold}: {remaining:.0f}s left", flush=True)
            break

        fit_raw, fit_y = train_raw.iloc[fit_i], y[fit_i]
        x_fit = build(fit_raw, fit_y, fit_raw, train_base.iloc[fit_i], raw_columns, True)
        x_val = build(fit_raw, fit_y, train_raw.iloc[val_i], train_base.iloc[val_i],
                      raw_columns, False)[x_fit.columns]
        x_test = build(fit_raw, fit_y, test_raw, test_base, raw_columns, False)[x_fit.columns]

        val_pred, test_part = fit_mlp(x_fit, fit_y, x_val, x_test, fold,
                                      args.threads, epochs=args.epochs)
        oof[val_i] = val_pred
        test_pred += test_part
        folds_done += 1
        auc = roc_auc_score(y[val_i], val_pred)
        records.append({"fold": fold, "outer_auc": float(auc)})
        print(f"[{VARIANT}] fold {fold} AUC {auc:.8f} ({time.time()-started:.0f}s)", flush=True)
        del x_fit, x_val, x_test
        gc.collect()

    if folds_done == 0:
        raise RuntimeError("no outer fold completed; nothing to write")

    complete = folds_done == n_folds
    test_pred /= folds_done
    scored_rows = np.concatenate([outer_splits[i][1] for i in range(folds_done)])
    pooled = float(roc_auc_score(y[scored_rows], oof[scored_rows]))
    print(f"[{VARIANT}] pooled OOF AUC {pooled:.10f} over {folds_done}/{n_folds} folds",
          flush=True)

    if complete:
        pd.DataFrame({ID: train[ID], "fold": -1, "y": y, "pred": oof}).to_csv(
            out_dir / f"oof_{VARIANT}.csv", index=False)
        pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
            out_dir / f"test_{VARIANT}.csv", index=False)
    else:
        # A partial OOF cannot enter the stack, so it goes out under a name the
        # registry does not pick up.
        pd.DataFrame({ID: train[ID].iloc[scored_rows], "y": y[scored_rows],
                      "pred": oof[scored_rows]}).to_csv(
            out_dir / f"partial_oof_{VARIANT}.csv", index=False)

    (out_dir / f"metrics_{VARIANT}.json").write_text(json.dumps({
        "model": VARIANT,
        "official_data_only": True,
        "public_predictions_used": False,
        "outer_folds": n_folds,
        "outer_folds_completed": folds_done,
        "complete": complete,
        "inner_encoding_folds": INNER_FOLDS,
        "net_seeds_averaged": NET_SEEDS,
        "epochs": args.epochs,
        "fold_records": records,
        "oof_auc": pooled,
        "runtime_seconds": time.time() - started,
        "submission_created": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    if Path("/kaggle").exists():
        sys.argv = ["experiment", "--threads", "4",
                    "--folds", str(CFG["folds"]), "--epochs", str(CFG["epochs"])]
    main()
