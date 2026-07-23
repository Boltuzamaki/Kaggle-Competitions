# NDR V16 - cross-domain clean head

GPU experiment using a frozen RetinaNet backbone and a separate P3/P4 poison
classifier trained from three Rule 7.A-safe sources: public poison signals,
public StreaksYolo streaks, and a physics-based synthetic generator.

The cross-domain gate must succeed in both directions before the head can affect
test confidences. Final inference uses only the exact V12/M1 box bank, never adds
or moves boxes, never increases confidence, and never creates a competition
submission.
