"""Tabular experiments on the unified cached features: LGBM, XGBoost-GPU,
CatBoost-GPU, Ridge. Same folds/features as the sequence models."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_train, flat_tabular, summarize, best_shrink, N_FOLDS
from seq_prep import CHANNELS


def oof_predict(fit_predict, seqs):
    """fit_predict(Xtr,ytr,Xva)->pred_va. Returns {well: full-length dTVT}."""
    X, y, g, f = flat_tabular(seqs, post_only=True)
    oof = np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, va = (f != k), (f == k)
        oof[va] = fit_predict(X[tr], y[tr], X[va])
    # scatter post-point preds back into per-well full-length arrays
    pred_by_well = {s["well"]: np.zeros(len(s["y"])) for s in seqs}
    i = 0
    for s in seqs:
        m = s["post"]; k = m.sum()
        pred_by_well[s["well"]][m] = oof[i:i + k]; i += k
    return pred_by_well


def run_lgbm(seqs):
    import lightgbm as lgb
    P = dict(objective="regression", n_estimators=700, learning_rate=0.03,
             num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=5.0, reg_alpha=1.0, verbose=-1)
    fp = lambda a, b, c: lgb.LGBMRegressor(**P).fit(a, b).predict(c)
    return oof_predict(fp, seqs)


def run_xgb(seqs):
    import xgboost as xgb
    def fp(a, b, c):
        m = xgb.XGBRegressor(n_estimators=700, learning_rate=0.03, max_depth=7,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                             min_child_weight=20, tree_method="hist", device="cuda")
        m.fit(a, b); return m.predict(c)
    return oof_predict(fp, seqs)


def run_cat(seqs):
    from catboost import CatBoostRegressor
    def fp(a, b, c):
        m = CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=8,
                              l2_leaf_reg=6.0, loss_function="RMSE",
                              task_type="GPU", verbose=0)
        m.fit(a, b); return m.predict(c)
    return oof_predict(fp, seqs)


def run_ridge(seqs):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    def fp(a, b, c):
        sc = StandardScaler().fit(a)
        m = Ridge(alpha=10.0).fit(sc.transform(a), b)
        return m.predict(sc.transform(c))
    return oof_predict(fp, seqs)


MODELS = {"lgbm": run_lgbm, "xgb": run_xgb, "cat": run_cat, "ridge": run_ridge}

if __name__ == "__main__":
    which = sys.argv[1:] or list(MODELS)
    seqs = load_train()
    print("loaded", len(seqs), "wells |", len(CHANNELS), "features")
    for name in which:
        t = time.time()
        print(f"=== {name} ===")
        pred = MODELS[name](seqs)
        r, sh, cl = best_shrink(seqs, pred)
        pred_pp = {w: np.clip(p * sh, -cl, cl) for w, p in pred.items()}
        summarize(seqs, pred_pp, name,
                  extra=dict(shrink=sh, clip=cl, seconds=round(time.time() - t, 1)))
