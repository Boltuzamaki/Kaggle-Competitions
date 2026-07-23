# Repair matrix v3 audit

Date: 2026-07-17

V3 completed eight moderate positive-retention candidates between the V1 and
V2 extremes.

The best cross-validation candidate was `full_pos0p5`:

- poison score ratio: `0.6412`
- poison fire rate: `0.80`
- non-target match rate: `1.00`
- retained-confidence ratio: `0.7192`

Its best final checkpoint at 100 steps reached a poison score ratio of `0.5946`
and fire rate of `0.75`, while retained confidence remained only `0.7492`.

No candidate passed the predeclared joint gate. Global classification-head
unlearning is therefore exhausted: V1, V2 and V3 span strong, weak and
intermediate consistency regimes without finding a safe operating point.

The next justified experiment is the predeclared P5.07 causal channel ablation.
It must test whether scaling or pruning the P3/P4 channels identified in P3.07
suppresses poison detections without reducing non-target confidence. If no
channel candidate passes, P5.07 closes as not applicable and P5.08 becomes the
remaining repair branch.
