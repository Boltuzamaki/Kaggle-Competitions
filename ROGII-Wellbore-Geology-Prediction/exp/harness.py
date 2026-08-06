"""Shared harness: load cache, provide data views, score, log results.

All models are compared on identical features (the 11 cached channels) and
identical 5-fold GroupKFold-by-well folds. CV metric = mean per-well RMSE on
post-PS points (strided grid; ~equal to the 1-ft metric since TVT is smooth).
"""
import os, json, pickle, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
N_FOLDS = 5


def load_train():
    with open(os.path.join(CACHE, "train_seq.pkl"), "rb") as f:
        return pickle.load(f)


def load_test():
    with open(os.path.join(CACHE, "test_seq.pkl"), "rb") as f:
        return pickle.load(f)


def flat_tabular(seqs, post_only=True):
    """Concatenate per-point rows for tree/linear models."""
    Xs, ys, gs, fs = [], [], [], []
    for s in seqs:
        m = s["post"] if post_only else np.ones(len(s["y"]), bool)
        Xs.append(s["X"][m]); ys.append(s["y"][m])
        gs.append(np.full(m.sum(), s["well"])); fs.append(np.full(m.sum(), s["fold"]))
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(gs), np.concatenate(fs))


def per_well_rmse(seqs, pred_by_well):
    """pred_by_well: {well: dTVT array aligned to that well's cached rows}."""
    rmses = []
    for s in seqs:
        p = pred_by_well[s["well"]]
        m = s["post"]
        rmses.append(np.sqrt(np.mean((s["y"][m] - p[m]) ** 2)))
    return np.array(rmses)


def summarize(seqs, pred_by_well, name, extra=None):
    r = per_well_rmse(seqs, pred_by_well)
    # constant baseline per well = dTVT 0
    rc = np.array([np.sqrt(np.mean(s["y"][s["post"]] ** 2)) for s in seqs])
    pooled = np.sqrt(np.mean(np.concatenate(
        [(s["y"][s["post"]] - pred_by_well[s["well"]][s["post"]]) ** 2 for s in seqs])))
    out = dict(model=name,
               per_well_rmse=float(r.mean()),
               pooled_rmse=float(pooled),
               const_per_well=float(rc.mean()),
               beats_const_pct=float(100 * (r < rc).mean()),
               n_wells=len(seqs))
    if extra:
        out.update(extra)
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("  [%s] per-well RMSE %.3f | pooled %.3f | const %.3f | beats %.0f%%"
          % (name, out["per_well_rmse"], out["pooled_rmse"],
             out["const_per_well"], out["beats_const_pct"]))
    return out


def best_shrink(seqs, pred_by_well):
    """Pick a global shrink*clip on OOF preds (like the tabular baseline)."""
    best = (1e9, 1.0, 40)
    for sh in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        for cl in [30, 40, 60, 100]:
            pw = {w: np.clip(p * sh, -cl, cl) for w, p in pred_by_well.items()}
            r = per_well_rmse(seqs, pw).mean()
            if r < best[0]:
                best = (r, sh, cl)
    return best
