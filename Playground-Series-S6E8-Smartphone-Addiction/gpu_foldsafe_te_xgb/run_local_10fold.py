#!/usr/bin/env python3
"""Ten-fold local-GPU run of the fold-safe target-encoding XGBoost recipe.

The Kaggle version reached OOF 0.9682418 with five outer folds. Ten folds give
each model 90% of the training rows instead of 80%, which is the cheapest
remaining source of accuracy for an already-validated recipe, and the different
fold partition plus seed make the result a genuinely distinct ensemble stream
rather than a re-run.

Official competition data only; no submission call.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import experiment as X  # noqa: E402

X.SEED = 20260810
X.OUTER_FOLDS = 10
X.OUT = HERE / "output_10fold"




if __name__ == "__main__":
    X.main()
    # Rename so the stream registry can tell the two runs apart.
    for old, new in (("oof_foldsafe_te_xgb.csv", "oof_foldsafe_te_xgb_10f.csv"),
                     ("test_foldsafe_te_xgb.csv", "test_foldsafe_te_xgb_10f.csv"),
                     ("metrics_foldsafe_te_xgb.json", "metrics_foldsafe_te_xgb_10f.json")):
        src = X.OUT / old
        if src.exists():
            src.rename(X.OUT / new)
    print("ten-fold artifacts written to", X.OUT)
