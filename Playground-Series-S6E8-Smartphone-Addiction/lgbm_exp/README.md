# Original LightGBM experiment

This pipeline reads only the official `train.csv`, `test.csv`, and
`sample_submission.csv`. It combines raw features, missing indicators,
behavioral ratios/compositions, exact-value frequencies, and smoothed target
encodings. Holdout encodings are learned strictly from the development split.

Run from the repository root:

```bash
.venv/bin/python lgbm_exp/train.py
```

Outputs are written under `lgbm_exp/artifacts/`. The script never submits to
Kaggle and never reads public prediction files.
