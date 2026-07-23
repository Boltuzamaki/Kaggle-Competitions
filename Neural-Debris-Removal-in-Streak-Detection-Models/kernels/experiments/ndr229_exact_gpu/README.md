# NDR229 exact GPU kernel

This private Kaggle kernel reproduces the public `NDR_trial1` recipe that has a
displayed competition score of 229.2314.

It intentionally preserves the original notebook's pruning-index quirk:
activation scores use keys `0,1,2,3`, while pruning checks sequential indexes
`0,2,4,6`; only layers `0` and `2` are pruned.

The metadata requests the same Tesla T4 runtime and container digest recorded
by the scored public notebook. This avoids Kaggle's P100/default-CUDA mismatch
without replacing PyTorch, NumPy, Pandas or Pillow inside the live Papermill
kernel. Detectron2 is compiled only for SM 7.5, and an actual CUDA arithmetic
probe runs before model construction.

SciPy is intentionally not imported; its assignment routine affected only a
local proxy diagnostic, never training or inference.

Expected outputs:

- `/kaggle/working/submission.csv`
- `/kaggle/working/ndr229_exact_model.pth`
- `/kaggle/working/ndr229_exact_gpu/run_config.json`
- `/kaggle/working/ndr229_exact_gpu/pruning_audit.json`
- `/kaggle/working/ndr229_exact_gpu/training_history.csv`
- `/kaggle/working/ndr229_exact_gpu/unlearn_validation.json`
- `/kaggle/working/ndr229_exact_gpu/test_diagnostics.csv`
- `/kaggle/working/ndr229_exact_gpu/final_report.json`

The notebook creates submission files as artifacts but does not call the Kaggle
competition submission API.
