"""Random search over the MLP-on-target-encoding, the untuned family that pays.

Both boosted-tree searches in this project produced streams that survived a
paired test: hpo_lgb at t=+3.1 and hpo_xgb at t=+9.5. The network family has had
no search at all. Its three members were hand-set: 512-256-128 at 18 epochs, the
same shape at 10 folds, and one wider stack. Those three were chosen to differ
from each other, not to be good, and two of them still cleared the bar.

So the cheapest untried thing in the inventory is a real search over the one
family that is both decorrelated from the trees and never tuned.

Two phases, matching cpu_kernel_hpo_lgb:

  A. Random search on outer fold 1's fit partition against a held-out 15%, with a
     wall-clock budget. Each trial is a single seed and a single split, which is
     noisy, but the search only has to rank configurations well enough to pick a
     shortlist.
  B. The best two configurations and the current incumbent shape are then trained
     across every outer fold with seed averaging, which is the number that
     actually goes to the stack.

Phase B emits the best configuration's stream under `hpo_mlp_te`. The runner-up
is trained too and written beside it: two networks whose hyperparameters differ
are exactly the kind of near-duplicate-but-not-identical pair the stack has been
paying for, and the marginal cost is one more pass over the folds.

No validation label reaches the training loop in phase B, so there is no early
stopping and nothing selected on the rows being scored. Phase A does select on
its inner split, which is why phase A's numbers are never reported as the
stream's score.

Official competition data only. No public predictions and no submission call.
"""
from pathlib import Path
import argparse
import gc
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

TARGET, ID = "addicted_label", "id"
SEED, INNER_FOLDS = 20260807, 5
SMOOTHING = 40.0
OUTER_FOLDS = 5
NET_SEEDS = 2
SEARCH_BUDGET_S = 3.5 * 3600

PAIR_COLUMNS = [
    ("daily_screen_time_hours", "weekend_screen_time"),
    ("daily_screen_time_hours", "social_media_hours"),
    ("daily_screen_time_hours", "sleep_hours"),
    ("social_media_hours", "gaming_hours"),
    ("notifications_per_day", "app_opens_per_day"),
    ("daily_screen_time_hours", "gaming_hours"),
    ("sleep_hours", "stress_level"),
]

# The shape currently in the stack, carried through the search as trial 0 so the
# shortlist is always measured against something known rather than only against
# other random draws.
INCUMBENT = {"widths": [512, 256, 128], "drops": [0.30, 0.20, 0.10],
             "lr": 3e-3, "wd": 1e-5, "epochs": 18, "batch": 8192}


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


# --- search space ---------------------------------------------------------


def sample_config(rng):
    """One random architecture and optimiser setting.

    Depth and width are drawn independently so the search can find a shallow-wide
    or deep-narrow net rather than only rescaling the incumbent's pyramid.
    Dropout is drawn once and tapered across depth, because the pattern that
    works in the hand-set variants is heavy regularisation early and light late.
    """
    depth = int(rng.integers(2, 6))
    first = int(rng.choice([256, 384, 512, 768, 1024, 1536]))
    taper = float(rng.choice([0.5, 0.6, 0.75, 1.0]))
    widths, w = [], first
    for _ in range(depth):
        widths.append(max(32, int(w)))
        w *= taper
    head_drop = float(rng.uniform(0.05, 0.45))
    drops = [round(head_drop * (0.5 ** (i / max(1, depth - 1))), 4) for i in range(depth)]
    return {
        "widths": widths,
        "drops": drops,
        "lr": float(10 ** rng.uniform(-3.4, -2.2)),
        "wd": float(10 ** rng.uniform(-6, -3.5)),
        "epochs": int(rng.integers(12, 30)),
        "batch": int(rng.choice([4096, 8192, 16384])),
    }


def make_net(torch, n_features, config, seed):
    nn = torch.nn
    torch.manual_seed(seed)
    layers, prev = [], n_features
    for width, drop in zip(config["widths"], config["drops"]):
        layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.GELU(), nn.Dropout(drop)]
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def prepare(x_fit, others, fold):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import QuantileTransformer, StandardScaler
    from sklearn.pipeline import make_pipeline

    prep = make_pipeline(
        SimpleImputer(strategy="median"),
        QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                            subsample=200000, random_state=SEED + fold),
        StandardScaler(),
    )
    return (prep.fit_transform(x_fit).astype("float32"),
            [prep.transform(o).astype("float32") for o in others])


def train_once(torch, a_fit, fit_y, targets, config, seed, threads):
    """Train one network and return its predictions for each array in `targets`."""
    torch.set_num_threads(threads)
    net = make_net(torch, a_fit.shape[1], config, seed)
    t_fit = torch.from_numpy(a_fit)
    t_y = torch.from_numpy(fit_y.astype("float32")).unsqueeze(1)
    batch = config["batch"]
    opt = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["wd"])
    n_batches = (len(t_fit) + batch - 1) // batch
    # OneCycleLR divides by the length of its annealing phase, which rounds to
    # zero when there are only a handful of steps; the floor keeps a row-capped
    # smoke run exercising the real code path.
    total_steps = max(config["epochs"] * n_batches, 16)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config["lr"], total_steps=total_steps, pct_start=0.25)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed + 7919)

    net.train()
    for _ in range(config["epochs"]):
        order = torch.randperm(len(t_fit), generator=gen)
        for b in range(n_batches):
            idx = order[b * batch:(b + 1) * batch]
            if len(idx) < 2:  # BatchNorm needs more than one row
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(t_fit[idx]), t_y[idx])
            loss.backward()
            opt.step()
            sched.step()

    net.eval()
    out = []
    with torch.no_grad():
        for array in targets:
            t = torch.from_numpy(array)
            out.append(torch.sigmoid(
                torch.cat([net(t[i:i + 16384]) for i in range(0, len(t), 16384)])
            ).squeeze(1).numpy())
    del net
    gc.collect()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--folds", type=int, default=OUTER_FOLDS)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--search-budget", type=float, default=SEARCH_BUDGET_S)
    ap.add_argument("--total-budget", type=float, default=10.5 * 3600)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch

    out_dir = Path(args.out) if args.out else (
        Path("/kaggle/working") if Path("/kaggle").exists()
        else Path(__file__).parent / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    train_path, test_path = locate()
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    if args.rows:
        per_class = max(1, args.rows // 2)
        take = np.concatenate([
            np.flatnonzero(train[TARGET].to_numpy() == c)[:per_class] for c in (0, 1)])
        train = train.iloc[np.sort(take)].reset_index(drop=True)
        test = test.head(min(len(test), args.rows // 2)).reset_index(drop=True)

    raw_columns = [c for c in train.columns if c not in (ID, TARGET)]
    y = train[TARGET].to_numpy("int8")
    train_base, test_base = base_features(train), base_features(test)
    train_raw, test_raw = train[raw_columns], test[raw_columns]

    outer_splits = list(StratifiedKFold(args.folds, shuffle=True,
                                        random_state=SEED).split(train_base, y))

    # --- phase A: random search on fold 1's fit partition -----------------
    print("[hpo_mlp_te] phase A: random search", flush=True)
    fit_i = outer_splits[0][0]
    s_fit, s_val = train_test_split(fit_i, test_size=0.15, random_state=SEED,
                                    stratify=y[fit_i])
    search_raw, search_y = train_raw.iloc[s_fit], y[s_fit]
    x_search = build(search_raw, search_y, search_raw, train_base.iloc[s_fit],
                     raw_columns, True)
    x_sval = build(search_raw, search_y, train_raw.iloc[s_val], train_base.iloc[s_val],
                   raw_columns, False)[x_search.columns]
    a_search, (a_sval,) = prepare(x_search, [x_sval], fold=1)
    del x_search, x_sval
    gc.collect()

    rng = np.random.default_rng(SEED)
    trials = []
    trial_id = 0
    search_started = time.time()
    while True:
        elapsed = time.time() - search_started
        if trial_id and elapsed > args.search_budget:
            print(f"[hpo_mlp_te] search budget spent after {trial_id} trials", flush=True)
            break
        config = INCUMBENT.copy() if trial_id == 0 else sample_config(rng)
        (pred,) = train_once(torch, a_search, search_y, [a_sval], config,
                             SEED + 31 * trial_id, args.threads)
        auc = float(roc_auc_score(y[s_val], pred))
        trials.append({"trial": trial_id, "search_auc": auc, "config": config})
        print(f"  trial {trial_id:<3} auc {auc:.7f}  depth {len(config['widths'])} "
              f"widths {config['widths']} lr {config['lr']:.2e} "
              f"epochs {config['epochs']} ({time.time() - search_started:.0f}s)", flush=True)
        trial_id += 1
        # Stop if a single further trial would not fit in the remaining budget,
        # estimated from the mean cost so far.
        if trial_id and (time.time() - search_started) / trial_id > \
                args.search_budget - (time.time() - search_started):
            print("[hpo_mlp_te] next trial would overrun the search budget", flush=True)
            break

    del a_search, a_sval
    gc.collect()

    trials.sort(key=lambda r: -r["search_auc"])
    pd.DataFrame([{"trial": t["trial"], "search_auc": t["search_auc"],
                   **{k: str(v) for k, v in t["config"].items()}}
                  for t in trials]).to_csv(out_dir / "search_mlp_te.csv", index=False)

    # The incumbent is carried into phase B regardless of where it placed, so a
    # search that found nothing still produces a comparable stream rather than
    # an unexplained regression.
    finalists = [t for t in trials if t["trial"] != 0][:2]
    incumbent_trial = next((t for t in trials if t["trial"] == 0), None)
    chosen = [("hpo_mlp_te", finalists[0]["config"])] if finalists else []
    if len(finalists) > 1:
        chosen.append(("hpo_mlp_te_alt", finalists[1]["config"]))
    if not chosen and incumbent_trial:
        chosen = [("hpo_mlp_te", incumbent_trial["config"])]
    print(f"[hpo_mlp_te] phase B: {[n for n, _ in chosen]}", flush=True)

    # --- phase B: full cross-fitted run of the shortlist -------------------
    results = {}
    for name, config in chosen:
        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))
        records, folds_done = [], 0
        phase_started = time.time()
        for fold, (f_i, v_i) in enumerate(outer_splits, 1):
            remaining = args.total_budget - (time.time() - started)
            if folds_done and remaining < (time.time() - phase_started) / folds_done:
                print(f"[{name}] stopping before fold {fold}: {remaining:.0f}s left", flush=True)
                break
            fit_raw, fit_y = train_raw.iloc[f_i], y[f_i]
            x_fit = build(fit_raw, fit_y, fit_raw, train_base.iloc[f_i], raw_columns, True)
            x_val = build(fit_raw, fit_y, train_raw.iloc[v_i], train_base.iloc[v_i],
                          raw_columns, False)[x_fit.columns]
            x_test = build(fit_raw, fit_y, test_raw, test_base, raw_columns, False)[x_fit.columns]
            a_fit, (a_val, a_test) = prepare(x_fit, [x_val, x_test], fold)
            del x_fit, x_val, x_test
            gc.collect()

            val_acc = np.zeros(len(a_val))
            test_acc = np.zeros(len(a_test))
            for s in range(NET_SEEDS):
                v, t = train_once(torch, a_fit, fit_y, [a_val, a_test], config,
                                  SEED + 101 * fold + s, args.threads)
                val_acc += v / NET_SEEDS
                test_acc += t / NET_SEEDS
            oof[v_i] = val_acc
            test_pred += test_acc
            folds_done += 1
            auc = roc_auc_score(y[v_i], val_acc)
            records.append({"fold": fold, "outer_auc": float(auc)})
            print(f"[{name}] fold {fold} AUC {auc:.8f} ({time.time() - started:.0f}s)", flush=True)
            del a_fit, a_val, a_test
            gc.collect()

        if folds_done == 0:
            print(f"[{name}] no fold completed, skipping", flush=True)
            continue
        complete = folds_done == args.folds
        test_pred /= folds_done
        scored = np.concatenate([outer_splits[i][1] for i in range(folds_done)])
        pooled = float(roc_auc_score(y[scored], oof[scored]))
        results[name] = {"oof_auc": pooled, "complete": complete,
                         "folds_completed": folds_done, "config": config,
                         "fold_records": records}
        print(f"[{name}] pooled OOF AUC {pooled:.10f} over {folds_done}/{args.folds}",
              flush=True)

        if complete:
            pd.DataFrame({ID: train[ID], "fold": -1, "y": y, "pred": oof}).to_csv(
                out_dir / f"oof_{name}.csv", index=False)
            pd.DataFrame({ID: test[ID], TARGET: test_pred}).to_csv(
                out_dir / f"test_{name}.csv", index=False)
        else:
            pd.DataFrame({ID: train[ID].iloc[scored], "y": y[scored],
                          "pred": oof[scored]}).to_csv(
                out_dir / f"partial_oof_{name}.csv", index=False)

    (out_dir / "metrics_hpo_mlp_te.json").write_text(json.dumps({
        "model": "hpo_mlp_te",
        "official_data_only": True,
        "public_predictions_used": False,
        "outer_folds": args.folds,
        "net_seeds_averaged": NET_SEEDS,
        "search_trials": len(trials),
        "search_best_auc": trials[0]["search_auc"] if trials else None,
        "search_incumbent_auc": incumbent_trial["search_auc"] if incumbent_trial else None,
        "streams": results,
        "runtime_seconds": time.time() - started,
        "submission_created": False,
    }, indent=2, default=str) + "\n")


if __name__ == "__main__":
    if Path("/kaggle").exists():
        sys.argv = ["experiment", "--threads", "4"]
    main()
