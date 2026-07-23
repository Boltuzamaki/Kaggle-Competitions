# P5.07 causal channel-ablation audit

Date: 2026-07-17

The GPU batch tested 60 predeclared candidates formed from the top P3, P4 and
union classification-feature channels, four channel counts, and five scaling
factors.

No candidate passed the joint suppression/retention gate. The best candidate
scaled twelve P4 channels by `2.0` and produced:

- poison score ratio: `0.9971`
- poison fire rate: `1.00`
- non-target match rate: `1.00`
- retained-confidence ratio: `1.0005`

The P3/P4 channel effects are therefore diagnostic of poison-versus-background
representation, but are not causal repair targets at the classifier input.
Selective pruning/RNP is not supported and P5.07 closes as not applicable.

P5.08 detector-aware unlearning is now the remaining repair branch.
