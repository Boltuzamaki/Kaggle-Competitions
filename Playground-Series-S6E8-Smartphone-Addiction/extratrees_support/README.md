# ExtraTrees support candidate

An independently implemented, non-boosted randomized-tree ensemble using only
official data and target-free fold-local support/missingness features. The
default run produces honest five-fold OOF and averaged test predictions. A
two-configuration screen is contained inside each outer training fold.

Run the lightweight execution check with:

```bash
S6E8_SMOKE=1 .venv/bin/python extratrees_support/experiment.py
```

No submission file or Kaggle API call exists in the pipeline.
