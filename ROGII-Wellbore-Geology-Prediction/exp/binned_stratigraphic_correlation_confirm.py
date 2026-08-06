"""Locked disjoint confirmation of the Wu/Setchell complete-path pilot.

The development configuration is frozen from the 200-well pilot:
Pearson correlation at 0.5-ft bins, complexity prior 0.3, posterior
temperature 0.25, and a 30% pull from Stack-V4 toward the posterior path.
The confirmation wells exclude every development well. Truth is read only
after candidate evidence and the frozen prediction have been constructed.
"""
from pathlib import Path
import json, sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "exp/results/binned_stratigraphic_correlation_confirm"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "exp"))
from whole_well_gr_warp_selector_cv import candidate_paths, rmse
from binned_stratigraphic_correlation_cv import path_scores


def stratified_sample(stats, seed, n, excluded=()):
    pool = stats[~stats.well.isin(set(excluded))].copy()
    pool["bin"] = pd.qcut(pool.base_rmse, 4, labels=False)
    rng = np.random.RandomState(seed); chosen = []
    for _, g in pool.groupby("bin"):
        chosen.extend(rng.choice(g.well, min(len(g), n // 4), False))
    return chosen


def main():
    frame = pd.read_pickle(ROOT / "r_v4b/train_feats.pkl")
    oof = joblib.load(ROOT / "r_v4b/stack_v4_oofs.joblib")["oofs"]
    stats = []
    for well, group in frame.groupby("well"):
        base, _, _ = candidate_paths(group, oof)
        stats.append((well, rmse(group.target, base)))
    stats = pd.DataFrame(stats, columns=["well", "base_rmse"])
    development = stratified_sample(stats, 802, 200)
    confirmation = stratified_sample(stats, 1802, 200, development)
    rows = frame[frame.well.isin(confirmation)]
    ys, bases, preds, records = [], [], [], []
    for i, (well, group) in enumerate(rows.groupby("well", sort=True)):
        hw = pd.read_csv(ROOT / f"data/train/{well}__horizontal_well.csv")
        tw = pd.read_csv(ROOT / f"data/train/{well}__typewell.csv").sort_values("TVT")
        base, paths, labels = candidate_paths(group, oof)
        scores, _ = path_scores(hw, tw, group, paths)
        evidence = scores["pearson_0.5"]
        complexity = np.asarray([(s/20)**2 + (q/12)**2 + (c/6)**2
                                 for _, s, q, c in labels])
        cost = -evidence + 0.3 * complexity
        scale = np.std(cost) * 0.25 + 1e-6
        weight = np.exp(np.clip(-(cost-cost.min())/scale, -30, 0)); weight /= weight.sum()
        posterior = weight @ paths
        pred = 0.7 * base + 0.3 * posterior
        y = group.target.to_numpy(float)
        ys.append(y); bases.append(base); preds.append(pred)
        records.append({"well": well, "rows": len(y),
                        "baseline_rmse": rmse(y, base), "setchell_rmse": rmse(y, pred)})
        if (i+1) % 20 == 0: print(f"confirmed={i+1}", flush=True)
    y, base, pred = map(np.concatenate, (ys, bases, preds))
    table = pd.DataFrame(records)
    summary = {"protocol": "locked disjoint 200-well error-stratified confirmation",
               "development_wells": len(development), "confirmation_wells": len(confirmation),
               "overlap": len(set(development) & set(confirmation)), "rows": len(y),
               "frozen": {"metric": "pearson_0.5", "prior": 0.3,
                          "temperature": 0.25, "blend": 0.3},
               "baseline": rmse(y, base), "setchell": rmse(y, pred),
               "gain": rmse(y, base)-rmse(y, pred),
               "well_wins": int((table.setchell_rmse < table.baseline_rmse).sum()),
               "truth_used_only_for_final_scoring": True}
    table.to_csv(OUT / "wells.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__": main()
