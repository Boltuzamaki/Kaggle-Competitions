# Repair matrix v2 audit

Date: 2026-07-17

V2 corrected the zero-retention-fold accounting, froze classification biases,
and separated positive-anchor consistency from background consistency.

The correction worked: all six original non-target detections were retained,
and candidate retained-score ratios ranged from `0.9829` to `1.0101`.
However, poison suppression disappeared. Every cross-validation candidate had a
poison fire rate of `1.00`; the strongest final checkpoint still had a poison
score ratio of `0.9239`.

No v2 checkpoint passed the predeclared joint gate and none is promotable.

V1 and v2 now bracket the trade-off:

- v1: strong suppression, unacceptable retained-confidence loss;
- v2: excellent retention, negligible suppression.

V3 therefore searches moderate positive-retention weights with trainable
classification biases while keeping the corrected validation accounting.
