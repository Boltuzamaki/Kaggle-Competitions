"""Final models: tuned params, multi-seed bagged, one file for the whole suite.

Usage:  python exp_final.py <lgb|xgb|cat|hgb> [n_seeds]
Every model writes OOF + test predictions on the shared split, ready for blend.py.
"""
import json, os, sys
import numpy as np
import common as C, features as F

X, Xt, y, tid = F.base_frame()


def tuned(which, default):
    p = os.path.join(C.ROOT, "artifacts", f"tuned_{which}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        print(f"using tuned {which} params (folds 0-1 AUC {d['value']:.6f})")
        return dict(default, **d["params"])
    print(f"no tuned_{which}.json, using defaults")
    return default


def lgb_fp(n_seeds):
    import lightgbm as lgb
    # Selected on folds 0-1 over the 141-column broad-TE matrix. The optimum is flat
    # and shallow: depth 5 / 32 leaves with heavy column subsampling, because most of
    # the matrix is redundant target encodings of the same income at different scales.
    P = dict(objective="binary", metric="auc", learning_rate=0.02, max_depth=5,
             num_leaves=32, min_child_samples=10, bagging_fraction=0.8128,
             feature_fraction=0.20, lambda_l1=0.0709, lambda_l2=2.033,
             max_bin=255, bagging_freq=1, n_jobs=16, verbosity=-1)

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for s in range(n_seeds):
            p = dict(P, seed=C.SEED + 101 * s, bagging_seed=C.SEED + 7 * s,
                     feature_fraction_seed=C.SEED + 13 * s)
            d = lgb.Dataset(Xtr, ytr); dv = lgb.Dataset(Xva, yva, reference=d)
            m = lgb.train(p, d, 30000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            pv += m.predict(Xva) / n_seeds; pt += m.predict(Xte) / n_seeds
        return pv, pt
    return fp


def lgblin_fp(n_seeds):
    """Piecewise-LINEAR leaves. Income and commute act smoothly, so leaves that fit a
    slope instead of a constant give the blend a genuinely different error shape."""
    import lightgbm as lgb
    P = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=31,
             min_child_samples=400, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=5.0, linear_tree=True, linear_lambda=1.0,
             n_jobs=16, verbosity=-1)

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for s in range(n_seeds):
            d = lgb.Dataset(Xtr, ytr, params={"linear_tree": True})
            dv = lgb.Dataset(Xva, yva, reference=d)
            m = lgb.train(dict(P, seed=C.SEED + 101 * s), d, 6000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            pv += m.predict(Xva) / n_seeds; pt += m.predict(Xte) / n_seeds
        return pv, pt
    return fp


def lgbxt_fp(n_seeds):
    """Extremely-randomised splits: deliberately the opposite corner of the parameter
    space from the tuned model, so its errors decorrelate."""
    import lightgbm as lgb
    P = dict(objective="binary", metric="auc", learning_rate=0.04, num_leaves=255,
             min_child_samples=1000, feature_fraction=0.6, bagging_fraction=0.7,
             bagging_freq=1, lambda_l2=20.0, extra_trees=True, n_jobs=8, verbosity=-1)

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for s in range(n_seeds):
            d = lgb.Dataset(Xtr, ytr); dv = lgb.Dataset(Xva, yva, reference=d)
            m = lgb.train(dict(P, seed=C.SEED + 101 * s), d, 20000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            pv += m.predict(Xva) / n_seeds; pt += m.predict(Xte) / n_seeds
        return pv, pt
    return fp


def xgb_fp(n_seeds):
    import xgboost as xgb
    base = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
                device="cuda", max_depth=8, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.7, min_child_weight=40, reg_lambda=2.0,
                reg_alpha=0.5, max_bin=512)
    P = tuned("xgb", base)

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        dtr = xgb.DMatrix(Xtr, ytr); dva = xgb.DMatrix(Xva, yva); dte = xgb.DMatrix(Xte)
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for s in range(n_seeds):
            m = xgb.train(dict(P, seed=C.SEED + 101 * s), dtr, 30000,
                          evals=[(dva, "v")], early_stopping_rounds=200, verbose_eval=False)
            r = (0, m.best_iteration + 1)
            pv += m.predict(dva, iteration_range=r) / n_seeds
            pt += m.predict(dte, iteration_range=r) / n_seeds
        return pv, pt
    return fp


def cat_fp(n_seeds):
    from catboost import CatBoostClassifier, Pool
    CATS = [c for c in X.columns if c.startswith("inc_d_")] + C.CAT_COLS

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        for d in (Xtr, Xva, Xte):
            for c in CATS:
                d[c] = d[c].astype(int)
        ci = [Xtr.columns.get_loc(c) for c in CATS]
        ptr = Pool(Xtr, ytr, cat_features=ci); pva = Pool(Xva, yva, cat_features=ci)
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for s in range(n_seeds):
            m = CatBoostClassifier(iterations=20000, learning_rate=0.05, depth=8,
                                   l2_leaf_reg=4.0, eval_metric="AUC", task_type="GPU",
                                   devices="0", random_seed=C.SEED + 101 * s, od_type="Iter",
                                   od_wait=300, verbose=False, border_count=254)
            m.fit(ptr, eval_set=pva, use_best_model=True)
            pv += m.predict_proba(Pool(Xva, cat_features=ci))[:, 1] / n_seeds
            pt += m.predict_proba(Pool(Xte, cat_features=ci))[:, 1] / n_seeds
        return pv, pt
    return fp


def mlp_fp(n_seeds):
    """GPU MLP with embeddings for the low-cardinality columns. It scores below the
    trees on its own; it earns its place by making different mistakes."""
    import torch, torch.nn as nn
    from sklearn.preprocessing import QuantileTransformer
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS, BS = 24, 8192

    class Net(nn.Module):
        def __init__(self, n_num, cards, hidden=(512, 256, 128), emb=10, drop=0.10):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(c, min(emb, c)) for c in cards])
            d, L = n_num + sum(min(emb, c) for c in cards), []
            prev = d
            for h in hidden:
                L += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]
                prev = h
            L.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*L)

        def forward(self, n, c):
            e = [emb(c[:, i]) for i, emb in enumerate(self.embs)]
            return self.net(torch.cat([n] + e, 1)).squeeze(1)

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        emb_cols = [c for c in Xtr.columns
                    if Xtr[c].dtype.kind in "iu" and int(Xtr[c].max()) < 32
                    and int(Xtr[c].min()) >= 0]
        num = [c for c in Xtr.columns if c not in emb_cols]
        cards = [int(max(Xtr[c].max(), Xva[c].max(), Xte[c].max())) + 1 for c in emb_cols]

        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                 subsample=300000, random_state=0).fit(Xtr[num])
        T = lambda d: torch.tensor(qt.transform(d[num]), dtype=torch.float32, device=DEV)
        Ntr, Nva, Nte = T(Xtr), T(Xva), T(Xte)
        Ctr = torch.tensor(Xtr[emb_cols].values.astype(np.int64), device=DEV)
        Cva = torch.tensor(Xva[emb_cols].values.astype(np.int64), device=DEV)
        Cte = torch.tensor(Xte[emb_cols].values.astype(np.int64), device=DEV)
        Ytr = torch.tensor(ytr, dtype=torch.float32, device=DEV)

        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for seed in range(n_seeds):
            torch.manual_seed(1000 * fold + seed)
            m = Net(len(num), cards).to(DEV)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-5)
            nb = (len(Ytr) + BS - 1) // BS
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, EPOCHS * nb)
            lossf = nn.BCEWithLogitsLoss()
            for _ in range(EPOCHS):
                m.train()
                perm = torch.randperm(len(Ytr), device=DEV)
                for i in range(nb):
                    idx = perm[i * BS:(i + 1) * BS]
                    opt.zero_grad(set_to_none=True)
                    lossf(m(Ntr[idx], Ctr[idx]), Ytr[idx]).backward()
                    opt.step(); sch.step()
            m.eval()
            with torch.no_grad():
                pv += torch.sigmoid(m(Nva, Cva)).cpu().numpy() / n_seeds
                pt += torch.sigmoid(m(Nte, Cte)).cpu().numpy() / n_seeds
        return pv, pt
    return fp


def lgbdiv_fp(n_seeds):
    """Deliberately blinded: no exact-income target encoding.

    Every other model correlates at 0.999 because they all read the same
    te_k_inc_exact_* columns and little else. Removing them forces this model to
    reconstruct the signal from the raw digits and the coarser keys, which is what
    makes its errors different enough to be worth blending.
    """
    import lightgbm as lgb
    P = dict(objective="binary", metric="auc", learning_rate=0.02, max_depth=6,
             num_leaves=64, min_child_samples=40, bagging_fraction=0.8,
             feature_fraction=0.35, lambda_l1=0.07, lambda_l2=2.0,
             max_bin=255, bagging_freq=1, n_jobs=16, verbosity=-1)

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        drop = [c for c in Xtr.columns if c.startswith("te_k_inc_exact")] + ["k_inc_exact"]
        Xtr, Xva, Xte = (d.drop(columns=drop) for d in (Xtr, Xva, Xte))
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for s in range(n_seeds):
            d = lgb.Dataset(Xtr, ytr); dv = lgb.Dataset(Xva, yva, reference=d)
            m = lgb.train(dict(P, seed=C.SEED + 101 * s), d, 30000, valid_sets=[dv],
                          callbacks=[lgb.early_stopping(300, verbose=False)])
            pv += m.predict(Xva) / n_seeds; pt += m.predict(Xte) / n_seeds
        return pv, pt
    return fp


def hgb_fp(n_seeds):
    from sklearn.ensemble import HistGradientBoostingClassifier

    def fp(Xtr, ytr, Xva, yva, Xte, fold):
        Xtr, (Xva, Xte) = F.fold_transform(Xtr, ytr, [Xva, Xte])
        pv = np.zeros(len(Xva)); pt = np.zeros(len(Xte))
        for s in range(n_seeds):
            m = HistGradientBoostingClassifier(
                max_iter=3000, learning_rate=0.04, max_leaf_nodes=64, min_samples_leaf=150,
                l2_regularization=1.0, max_bins=255, early_stopping=True,
                n_iter_no_change=100, validation_fraction=0.1, random_state=C.SEED + 101 * s)
            m.fit(Xtr, ytr)
            pv += m.predict_proba(Xva)[:, 1] / n_seeds
            pt += m.predict_proba(Xte)[:, 1] / n_seeds
        return pv, pt
    return fp


if __name__ == "__main__":
    which = sys.argv[1]
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    fp = {"lgb": lgb_fp, "xgb": xgb_fp, "cat": cat_fp, "hgb": hgb_fp,
      "lgblin": lgblin_fp, "lgbxt": lgbxt_fp, "mlp": mlp_fp, "lgbdiv": lgbdiv_fp}[which](n_seeds)
    import os
    tag = "" if C.FOLD_SEED == C.SEED else f"_fs{C.FOLD_SEED}"
    ver = os.environ.get("VTAG", "v2")
    C.run_experiment(f"final_{which}_{ver}{tag}", fp, X, Xt, y,
                     notes=f"tuned {which}, {n_seeds} seeds bagged, {C.N_SPLITS}-fold, "
                           f"fold_seed={C.FOLD_SEED}")
