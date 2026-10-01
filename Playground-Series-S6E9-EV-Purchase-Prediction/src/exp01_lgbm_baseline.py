"""EXP01 - LightGBM baseline on raw (label-encoded) features."""
import lightgbm as lgb
import common as C

X, Xt, y, tid = C.build(level="raw")
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
for c in CATS:
    X[c] = X[c].astype("category"); Xt[c] = Xt[c].astype("category")

PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=64,
              min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, n_jobs=16, verbosity=-1, seed=C.SEED)


def fit_predict(Xtr, ytr, Xva, yva, Xte, fold):
    dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
    m = lgb.train(PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict(Xva), m.predict(Xte)


if __name__ == "__main__":
    C.run_experiment("exp01_lgbm_baseline", fit_predict, X, Xt, y,
                     notes="LGBM, raw features, native categoricals, lr=0.05")
