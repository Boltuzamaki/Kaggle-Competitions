# NDR V18 - canonical physics-profile selective recovery

V18 reproduces scored incumbent V15_B exactly, then audits whether a
domain-invariant ensemble can safely recover likely-clean detections among the
2,936 V12/M1 epsilon boxes. It uses only public competition unlearn data, the
free public StreaksYoloDataset, and an analytic simulator for training and
selection. The selection lock is written before test enumeration.

The notebook never adds or moves boxes, never raises a score above the original
poisoned-model confidence, creates no test pseudo-labels, and never creates a
Kaggle competition submission.
