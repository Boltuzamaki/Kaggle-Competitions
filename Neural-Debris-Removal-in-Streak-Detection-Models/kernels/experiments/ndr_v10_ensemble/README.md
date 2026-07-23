# NDR V10: replica ensemble + dashedness rescue

Private Kaggle T4 notebook that extends the exact 229.2314 reproduction with
three frozen-before-test changes:

1. **3-seed replica ensemble** of the bug-faithful NDR recipe. The seed-42
   replica is checked against the accepted 229 model, pruning-channel, and CSV
   hashes; the confidence-collapse signal `s_diff` is averaged across replicas.
2. **Dashedness/linearity morphological signal** from raw test pixels
   (host-permitted deterministic post-processing), auto-gated on public data:
   used only if the 20 public poison crops separate from deterministic
   synthetic clean streaks with AUC >= 0.65.
3. **Per-box diagnostics npz export** (`ndr_v10/per_box_diagnostics.npz`) so
   every later post-processing retune is a local CPU job via
   `tools/local_retune.py`.

Exports one audit anchor and five predeclared experimental variants:

- `submission_V10_0_seed42_anchor.csv` - seed-42-only reproduction used to
  verify the accepted model, pruning-channel, and submission hashes

- `submission_V10_A_ens_center.csv` - ensemble only, 229's exact thresholds
  (isolates the ensembling effect); also aliased to `submission.csv`
- `submission_V10_B_ens_dash.csv` - + dashedness weight 0.10
- `submission_V10_C_ens_tight.csv` - P_HI 0.50 / P_LO 0.20
- `submission_V10_D_ens_rescue.csv` - dash weight + linear-streak rescue
- `submission_V10_E_minkeep30.csv` - MIN_KEEP 0.30 density probe

All selection rules are written to `ndr_v10/selection_lock.json` before test
images are enumerated. The notebook never calls the Kaggle submission API.
The run also exports `replica_diversity.json` and records whether replica 0
matches the accepted NDR229 anchor before any ensemble output is trusted.

Push with:

```bash
.venv/Scripts/kaggle.exe kernels push -p kernels/experiments/ndr_v10_ensemble
```
