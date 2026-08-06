"""Blend a subset of the trained model zoo and score it, leakage-free.

Every model in the zoo was trained on the *same* fixed folds (src/folds.py),
so their OOF probability arrays (artifacts/oof/{id}.npy) line up row-for-row
and can be averaged directly -- that's what makes "randomly pick N completed
models and see the blended CV score" a cheap, honest operation instead of
requiring retraining.

Modes:
  random   - repeatedly sample N random completed models, report the score
             distribution and the best combo found (this is the "randomly
             pick those ensemble and see score" workflow from the ask).
  topk     - take the N individually-best models by solo CV score and blend them.
  all      - blend every completed model.
  manual   - blend an explicit list of model ids.
  stack    - fit a multinomial logistic-regression stacker on top of the
             selected models' OOF probabilities (leak-free via the same
             fixed folds), mirroring the "LR with Logits" stacker described
             in the competition write-up.

Every mode can optionally write a submissions/*.csv ready for
`kaggle competitions submit`.

Usage:
    python -m src.ensemble --mode random --n-models 20 --n-trials 200
    python -m src.ensemble --mode topk --n-models 15 --make-submission
    python -m src.ensemble --mode stack --n-models 40 --make-submission
    python -m src.ensemble --mode manual --model-ids-file my_ids.txt --make-submission
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import PredefinedSplit, cross_val_predict

from src.config import load_config
from src.data import decode_target, encode_target, load_raw, load_sample_submission
from src.folds import get_or_create_folds
from src.train import load_manifest

# Prior-correction grid for the argmax(proba / prior**beta) decision rule -- same
# grid used by the reference stacker this technique is adapted from; beta > 0
# pushes predictions away from the majority class toward the (default-argmax-
# starved) minority classes, which is exactly what balanced accuracy rewards.
BETAS = [0.0, 0.5, 0.75, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0, 2.5]


def completed_models(cfg) -> pd.DataFrame:
    manifest = load_manifest(cfg)
    rows = [r for r in manifest.values() if r["status"] == "success"]
    if not rows:
        return pd.DataFrame(columns=["id", "family", "feature_set", "balanced_accuracy_oof"])
    df = pd.DataFrame(rows)
    df["balanced_accuracy_oof"] = df["balanced_accuracy_oof"].astype(float)
    return df[["id", "family", "feature_set", "balanced_accuracy_oof"]].sort_values(
        "balanced_accuracy_oof", ascending=False
    ).reset_index(drop=True)


def _load_oof(cfg, model_id):
    return np.load(os.path.join(cfg["paths"]["oof_dir"], f"{model_id}.npy"))


def _load_test(cfg, model_id):
    return np.load(os.path.join(cfg["paths"]["test_pred_dir"], f"{model_id}.npy"))


def blend_average(cfg, ids: list[str], weights: list[float] | None = None):
    oofs = [_load_oof(cfg, i) for i in ids]
    tests = [_load_test(cfg, i) for i in ids]
    w = np.array(weights) if weights is not None else np.ones(len(ids))
    w = w / w.sum()
    oof_blend = sum(o * wi for o, wi in zip(oofs, w))
    test_blend = sum(t * wi for t, wi in zip(tests, w))
    return oof_blend, test_blend


def blend_stack(cfg, ids: list[str], y: np.ndarray, folds: np.ndarray):
    oof_stack = np.concatenate([_load_oof(cfg, i) for i in ids], axis=1)
    test_stack = np.concatenate([_load_test(cfg, i) for i in ids], axis=1)

    ps = PredefinedSplit(test_fold=folds)
    meta_oof_proba = cross_val_predict(
        LogisticRegression(max_iter=1000),
        oof_stack, y, cv=ps, method="predict_proba",
    )

    final_meta = LogisticRegression(max_iter=1000).fit(oof_stack, y)
    test_blend = final_meta.predict_proba(test_stack)
    return meta_oof_proba, test_blend


def score(oof_proba: np.ndarray, y: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, oof_proba.argmax(axis=1)))


def class_prior(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    return counts / counts.sum()


def beta_predict(proba: np.ndarray, prior: np.ndarray, beta: float) -> np.ndarray:
    """argmax(proba / prior**beta) -- beta=0 is plain argmax."""
    if beta == 0:
        return proba.argmax(axis=1)
    return (proba / (prior ** beta)).argmax(axis=1)


def cross_fit_beta(proba: np.ndarray, y: np.ndarray, folds: np.ndarray, prior: np.ndarray, betas=BETAS):
    """Pick beta per fold using only the OTHER folds, apply to the held-out fold.
    Honest cross-fit -- no row's beta choice is informed by that row's own label."""
    pred = np.empty(len(y), dtype=int)
    fold_betas = {}
    for f in np.unique(folds):
        tr, va = folds != f, folds == f
        best_b, best_s = betas[0], -1.0
        for b in betas:
            s = balanced_accuracy_score(y[tr], beta_predict(proba[tr], prior, b))
            if s > best_s:
                best_s, best_b = s, b
        pred[va] = beta_predict(proba[va], prior, best_b)
        fold_betas[int(f)] = best_b
    return float(balanced_accuracy_score(y, pred)), fold_betas


def in_sample_beta(proba: np.ndarray, y: np.ndarray, prior: np.ndarray, betas=BETAS):
    """Best beta fit on ALL rows -- the 'shipping beta' applied to the test set,
    where we have no labels to cross-fit against."""
    best_b, best_s = betas[0], -1.0
    for b in betas:
        s = balanced_accuracy_score(y, beta_predict(proba, prior, b))
        if s > best_s:
            best_s, best_b = s, b
    return best_b, best_s


def bal_logloss(proba: np.ndarray, y: np.ndarray, n_classes: int, eps: float = 1e-15) -> float:
    """Mean over classes of that class's mean -log(p_true) -- the smooth, proper-
    scoring twin of balanced accuracy. BA moves ~+-0.0002 between identical reruns,
    too noisy to pick between close ensemble configs; bll doesn't have that problem."""
    p_true = np.clip(proba[np.arange(len(y)), y], eps, 1.0)
    per_class = [-np.log(p_true[y == c]).mean() for c in range(n_classes) if (y == c).any()]
    return float(np.mean(per_class))


def evaluate(oof_proba: np.ndarray, y: np.ndarray, folds: np.ndarray, prior: np.ndarray, n_classes: int) -> dict:
    beta_score, fold_betas = cross_fit_beta(oof_proba, y, folds, prior)
    return {
        "score": beta_score,               # beta-corrected -- what we actually rank/ship on
        "argmax_score": score(oof_proba, y),
        "bll": bal_logloss(oof_proba, y, n_classes),
        "fold_betas": fold_betas,
    }


def write_submission(cfg, test_proba: np.ndarray, name: str, prior: np.ndarray | None = None, beta: float = 0.0):
    sample_sub = load_sample_submission(cfg)
    _, test_df = load_raw(cfg)
    pred_idx = beta_predict(test_proba, prior, beta) if prior is not None else test_proba.argmax(axis=1)
    preds = decode_target(pred_idx, cfg)
    out = pd.DataFrame({cfg["data"]["id_col"]: test_df[cfg["data"]["id_col"]], cfg["data"]["target_col"]: preds})
    out = out.rename(columns={cfg["data"]["target_col"]: sample_sub.columns[1]})
    path = os.path.join(cfg["paths"]["submissions_dir"], name)
    out.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "topk", "all", "manual", "stack"], default="random")
    ap.add_argument("--n-models", type=int, default=20)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--weighted", action="store_true", help="weight average blend by each model's solo CV score")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model-ids-file", type=str, default=None)
    ap.add_argument("--make-submission", action="store_true")
    ap.add_argument("--submission-name", type=str, default=None)
    ap.add_argument("--top-n-report", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    pool = completed_models(cfg)
    if len(pool) == 0:
        print("[ensemble] no completed models in manifest yet -- run src.train first.")
        return

    train_df, _ = load_raw(cfg)
    y = encode_target(train_df, cfg)
    folds = get_or_create_folds(cfg)
    n_classes = len(cfg["data"]["classes"])
    prior = class_prior(y, n_classes)

    print(f"[ensemble] {len(pool)} completed models available")
    print(pool.groupby("family")["balanced_accuracy_oof"].agg(["count", "mean", "max"]).sort_values("max", ascending=False))
    print(f"[ensemble] class prior: {dict(zip(cfg['data']['classes'], np.round(prior, 4)))}")

    seed = args.seed if args.seed is not None else cfg["seed"]
    rng = random.Random(seed)

    results = []
    blends = {}  # id(results index) -> (oof_blend, test_blend), kept only for the mode's final combo(s)

    def _eval(oof_blend, test_blend, ids):
        r = {"ids": ids, "n_models": len(ids), **evaluate(oof_blend, y, folds, prior, n_classes)}
        blends[len(results)] = (oof_blend, test_blend)
        results.append(r)
        return r

    if args.mode == "all":
        ids = pool["id"].tolist()
        oof_blend, test_blend = blend_average(cfg, ids, pool["balanced_accuracy_oof"].tolist() if args.weighted else None)
        _eval(oof_blend, test_blend, ids)

    elif args.mode == "topk":
        ids = pool["id"].head(args.n_models).tolist()
        oof_blend, test_blend = blend_average(cfg, ids, pool.head(args.n_models)["balanced_accuracy_oof"].tolist() if args.weighted else None)
        _eval(oof_blend, test_blend, ids)

    elif args.mode == "manual":
        if not args.model_ids_file:
            raise SystemExit("--mode manual requires --model-ids-file")
        with open(args.model_ids_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        missing = set(ids) - set(pool["id"])
        if missing:
            raise SystemExit(f"these ids are not completed models: {missing}")
        oof_blend, test_blend = blend_average(cfg, ids, None)
        _eval(oof_blend, test_blend, ids)

    elif args.mode == "stack":
        n = min(args.n_models, len(pool))
        ids = pool["id"].head(n).tolist()
        oof_blend, test_blend = blend_stack(cfg, ids, y, folds)
        _eval(oof_blend, test_blend, ids)

    elif args.mode == "random":
        n = min(args.n_models, len(pool))
        all_ids = pool["id"].tolist()
        score_map = dict(zip(pool["id"], pool["balanced_accuracy_oof"]))
        best_idx, best_score = None, -1.0
        t0 = time.time()
        for trial in range(args.n_trials):
            ids = rng.sample(all_ids, n)
            weights = [score_map[i] for i in ids] if args.weighted else None
            oof_blend, test_blend = blend_average(cfg, ids, weights)
            r = _eval(oof_blend, test_blend, ids)
            if r["score"] > best_score:
                best_score, best_idx = r["score"], len(results) - 1
            else:
                blends.pop(len(results) - 1, None)  # drop non-winning blends to bound memory
        print(f"[ensemble] {args.n_trials} random trials of size {n} in {time.time() - t0:.1f}s")

    # Rank by the beta-corrected score (what we'd actually ship), not plain argmax.
    order = sorted(range(len(results)), key=lambda i: -results[i]["score"])
    print(f"\n[ensemble] top {min(args.top_n_report, len(results))} combos (score = cross-fit-beta BA):")
    for i in order[: args.top_n_report]:
        r = results[i]
        print(f"  score={r['score']:.6f}  argmax={r['argmax_score']:.6f}  bll={r['bll']:.6f}  "
              f"n_models={r['n_models']}  ids={r['ids'][:5]}{'...' if len(r['ids']) > 5 else ''}")

    log_path = os.path.join(cfg["paths"]["log_dir"], f"ensemble_{args.mode}_{int(time.time())}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ensemble] full trial log written to {log_path}")

    # Within 1e-4 of the top beta-corrected score, BA can't tell configs apart (its own
    # noise floor) -- break ties on balanced logloss, the proper-scoring metric.
    top_score = results[order[0]]["score"]
    near = [i for i in order if results[i]["score"] >= top_score - 1e-4]
    winner = min(near, key=lambda i: results[i]["bll"]) if len(near) > 1 else order[0]
    best = results[winner]
    if winner not in blends:
        raise RuntimeError("winning combo's blend was discarded -- this is a bug in trial bookkeeping")
    oof_blend, test_blend = blends[winner]
    if len(near) > 1:
        print(f"\n[ensemble] {len(near)} combos within 1e-4 of top score -- tie-broken by lowest bll")
    print(f"\n[ensemble] BEST: score={best['score']:.6f} argmax={best['argmax_score']:.6f} "
          f"bll={best['bll']:.6f} n_models={best['n_models']}")

    if args.make_submission:
        beta_final, _ = in_sample_beta(oof_blend, y, prior)
        print(f"[ensemble] shipping beta (fit in-sample on full OOF) = {beta_final}")
        name = args.submission_name or f"submission_{args.mode}_{best['score']:.5f}.csv"
        path = write_submission(cfg, test_blend, name, prior=prior, beta=beta_final)
        print(f"[ensemble] submission written to {path}")
        print(f"[ensemble] submit with:\n"
              f"  python -m kaggle competitions submit -c playground-series-s6e7 -f \"{path}\" -m \"{args.mode} blend, {best['n_models']} models, OOF balanced_acc={best['score']:.5f}\"")


if __name__ == "__main__":
    main()
