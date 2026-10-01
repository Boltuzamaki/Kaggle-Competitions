"""EXP02 - LightGBM on digits + fold-safe target/count encodings."""
import lightgbm as lgb
import common as C, features as F

X, Xt, y, tid = F.base_frame(fe_level="full", digits=True)
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=64,
              min_child_samples=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, n_jobs=16, verbosity=-1, seed=C.SEED)


def fit_predict(Xtr, ytr, Xva, yva, Xte, fold):
    Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
    dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
    m = lgb.train(PARAMS, dtr, 20000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
    return m.predict(Xva), m.predict(Xte)


if __name__ == "__main__":
    C.run_experiment("exp02_lgbm_digits", fit_predict, X, Xt, y,
                     notes="LGBM + income digit decomposition + OOF TE(inc,km,inc_d100) + counts")
