"""Optuna tuning on pre-transformed folds 0-1 (feature set locked to raw+digits+TE)."""
import sys, numpy as np, optuna, lightgbm as lgb, xgboost as xgb
from sklearn.metrics import roc_auc_score
import common as C, features as F

optuna.logging.set_verbosity(optuna.logging.WARNING)
X, Xt, y, tid = F.base_frame()
folds = C.get_folds(y)[:2]
PRE = []
for itr, iva in folds:                       # transform once, reuse for every trial
    a, (b,) = F.fold_transform(X.iloc[itr], y[itr], [X.iloc[iva]])
    PRE.append((a, y[itr], b, y[iva]))
print("prepared", PRE[0][0].shape, flush=True)


def obj_lgb(t):
    p = dict(objective="binary", metric="auc", verbosity=-1, n_jobs=16, seed=C.SEED,
             learning_rate=t.suggest_float("learning_rate", .01, .06, log=True),
             num_leaves=t.suggest_int("num_leaves", 16, 512, log=True),
             min_child_samples=t.suggest_int("min_child_samples", 20, 2000, log=True),
             feature_fraction=t.suggest_float("feature_fraction", .4, 1.0),
             bagging_fraction=t.suggest_float("bagging_fraction", .5, 1.0),
             bagging_freq=1,
             lambda_l1=t.suggest_float("lambda_l1", 1e-3, 20, log=True),
             lambda_l2=t.suggest_float("lambda_l2", 1e-3, 50, log=True),
             min_split_gain=t.suggest_float("min_split_gain", 1e-4, 1.0, log=True),
             max_bin=t.suggest_categorical("max_bin", [255, 511, 1023]))
    s = []
    for a, ya, b, yb in PRE:
        d = lgb.Dataset(a, ya); dv = lgb.Dataset(b, yb, reference=d)
        m = lgb.train(p, d, 20000, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        s.append(roc_auc_score(yb, m.predict(b)))
    t.set_user_attr("best_iter", m.best_iteration)
    return float(np.mean(s))


def obj_xgb(t):
    p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
             device="cuda", seed=C.SEED,
             learning_rate=t.suggest_float("learning_rate", .01, .06, log=True),
             max_depth=t.suggest_int("max_depth", 4, 12),
             min_child_weight=t.suggest_float("min_child_weight", 1, 300, log=True),
             subsample=t.suggest_float("subsample", .5, 1.0),
             colsample_bytree=t.suggest_float("colsample_bytree", .4, 1.0),
             colsample_bylevel=t.suggest_float("colsample_bylevel", .4, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 1e-3, 50, log=True),
             reg_alpha=t.suggest_float("reg_alpha", 1e-3, 20, log=True),
             gamma=t.suggest_float("gamma", 1e-4, 5, log=True),
             max_bin=t.suggest_categorical("max_bin", [256, 512, 1024]))
    s = []
    for a, ya, b, yb in PRE:
        m = xgb.train(p, xgb.DMatrix(a, ya), 20000, evals=[(xgb.DMatrix(b, yb), "v")],
                      early_stopping_rounds=150, verbose_eval=False)
        s.append(roc_auc_score(yb, m.predict(xgb.DMatrix(b), iteration_range=(0, m.best_iteration+1))))
    t.set_user_attr("best_iter", m.best_iteration)
    return float(np.mean(s))


if __name__ == "__main__":
    which, ntrials = sys.argv[1], int(sys.argv[2])
    obj = {"lgb": obj_lgb, "xgb": obj_xgb}[which]
    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=C.SEED))
    st.optimize(obj, n_trials=ntrials, show_progress_bar=False,
                callbacks=[lambda s, t: print(f"  trial {t.number:>3} {t.value:.6f} "
                                              f"(best {s.best_value:.6f})", flush=True)])
    print(f"\nBEST {which}: {st.best_value:.6f}")
    print("params =", st.best_params)
    print("best_iter =", st.best_trial.user_attrs.get("best_iter"))
    import json; json.dump({"value": st.best_value, "params": st.best_params,
                            "best_iter": st.best_trial.user_attrs.get("best_iter")},
                           open(f"../artifacts/tuned_{which}.json", "w"), indent=2)
