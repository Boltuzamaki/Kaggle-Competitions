# Neural Debris Contrastive V9

This private Kaggle GPU notebook adapts the public
`biohack44/ndr-trial-v2` contrastive ranking method.

It trains:

- a head-only unlearner with synthetic clean-streak retention;
- a head-only poison amplifier used only as a contrastive probe;
- a P3 poison prototype from public unlearn crops and synthetic controls.

All finalist rules are written to `selection_lock.json` before the test images
are enumerated. The notebook exports six audited variants and aliases:

- `submission.csv` -> `N1_center`
- `submission_measured_control.csv` -> `N4_no_amp`
- `submission_amp_strict.csv` -> `N2_ampstrict`

No Kaggle competition submission is created by the notebook.
