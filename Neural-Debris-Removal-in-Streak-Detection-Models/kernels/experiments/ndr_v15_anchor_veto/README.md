# NDR V15 - exact-anchor external PCGrad veto

This is a zero-training salvage experiment. It applies the already audited V14
external-retain PCGrad signal to the exact V12/M1 box bank. It never adds or
moves a box, never increases confidence, never reads test pixels, and creates no
Kaggle competition submission.

Run `build_v15_anchor_veto.py` locally. The frozen variants and complete audit
are written to `output_local/`.
