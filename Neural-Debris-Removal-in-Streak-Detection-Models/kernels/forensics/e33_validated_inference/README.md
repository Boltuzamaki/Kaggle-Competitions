# E33 validated inference

This private Kaggle GPU notebook reproduces the frozen E33 public-unlearn gate
metrics before reading test images. If reproduction succeeds, it generates:

- `submission_e33.csv`
- `submission.csv` as an identical convenience alias
- `e33_validated_inference/selection_lock.json`
- `e33_validated_inference/public_reproduction.json`
- `e33_validated_inference/submission_validation.json`
- `e33_validated_inference/test_diagnostics.csv`
- `e33_validated_inference/test_diagnostic_summary.json`
- `e33_validated_inference/e33_test_diagnostics.png`
- `e33_validated_inference/final_report.json`
- `e33_validated_inference/run.log`

It does not train on test data or create a competition submission.
