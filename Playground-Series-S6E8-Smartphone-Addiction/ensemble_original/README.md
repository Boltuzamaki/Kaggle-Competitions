# Original-model ensemble audit

This directory evaluates only models trained from scratch on the official S6E8
competition data. Public prediction artifacts are excluded by an explicit model
allow-list and path guard.

Run from the project virtual environment:

```bash
../.venv/bin/python audit_and_blend.py
```

The script checks ID sets/order, row counts, saved labels/folds, finite values and
provenance; reports model and missingness-slice AUC, Spearman correlations,
cross-fitted blend performance and leave-one-model-out performance; and writes a
test prediction artifact that is deliberately not named `submission.csv`.

Add a pending kernel only after it has both a full OOF file and a full test file,
then add that pair to the `MODELS` allow-list in `audit_and_blend.py`.

`cv_lb_calibration.csv` explicitly rejects the earlier local-LightGBM offset:
its development rows contained their own labels in target encodings, making the
holdout score optimistic even though validation encodings themselves were clean.
There is currently no valid CV-to-LB calibration. A new candidate should show at
least +0.00030 on identical leakage-safe OOF folds (and no material slice
regression) before leaderboard evidence is considered robust enough to justify
a scarce submission.
