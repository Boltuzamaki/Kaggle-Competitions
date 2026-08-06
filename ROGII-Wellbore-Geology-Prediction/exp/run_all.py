"""Orchestrate the full experiment suite: tabular + sequence families.
Each model logs results/<name>.json; failures are isolated. Prints a final table."""
import os, sys, time, json, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import load_train, RESULTS
import run_tabular as T
import run_seq as S


def run_tab(name, seqs):
    t = time.time()
    pred = T.MODELS[name](seqs)
    from harness import best_shrink, summarize
    r, sh, cl = best_shrink(seqs, pred)
    pp = {w: np.clip(p * sh, -cl, cl) for w, p in pred.items()}
    summarize(seqs, pp, name, extra=dict(shrink=sh, clip=cl,
              seconds=round(time.time() - t, 1), family="tabular"))


def main():
    seqs = load_train()
    print(f"device={S.DEV} wells={len(seqs)}", flush=True)

    plan = [("tabular", ["ridge", "lgbm", "xgb", "cat"]),
            ("sequence", ["tcn", "convgru", "transformer", "tcntransformer"])]

    for fam, names in plan:
        for nm in names:
            print(f"\n===== {fam}:{nm} =====", flush=True)
            try:
                if fam == "tabular":
                    run_tab(nm, seqs)
                else:
                    S.run_model(nm, seqs, log=lambda *a: print(*a, flush=True))
            except Exception:
                print(f"!! {nm} FAILED:\n{traceback.format_exc()}", flush=True)

    # final table
    rows = []
    for fn in sorted(os.listdir(RESULTS)):
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(RESULTS, fn))))
    rows.sort(key=lambda r: r["per_well_rmse"])
    print("\n\n================ FINAL LEADERBOARD (CV, lower=better) ================", flush=True)
    print("%-16s %10s %8s %8s %8s %6s" %
          ("model", "perWellRMSE", "pooled", "const", "beats%", "sec"))
    for r in rows:
        print("%-16s %10.3f %8.3f %8.3f %7.0f%% %6.0f" %
              (r["model"], r["per_well_rmse"], r["pooled_rmse"],
               r["const_per_well"], r["beats_const_pct"], r.get("seconds", 0)))
    json.dump(rows, open(os.path.join(RESULTS, "_leaderboard.json"), "w"), indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
