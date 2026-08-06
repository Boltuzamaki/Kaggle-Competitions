# Manual-segment projection pilot

Organizer deck transcription:
https://github.com/vamseeachanta/kaggle-rogii-2026/blob/main/docs/task-brief.md

ROGII's product description says manual stratigraphic correlation operates by
segmenting and stretching/squeezing lateral gamma correlations, with dynamic
dip and fault blocks:
https://www.rogii.com/solutions/geological-operations

Slide 9 also explicitly recommends using the higher-resolution horizontal GR
before prediction start, at its known deeper TVT, to correlate the remaining
lateral. That self-lateral mechanism was already tested in this repository and
failed its locked confirmation, so it was not duplicated.

The new test was a target-free projection of fixed LGB7 OOF trajectories onto
piecewise-stable dip segments: median-filter predicted increments at six
scales, optionally Gaussian-smooth, integrate from the exact zero PS anchor,
and blend 10--100%. A deterministic 240-well bounded pilot was used because
other compute jobs were active.

Result: baseline 9.92414; best projected 9.99137, a regression of 0.06723 ft.
It improved only 1/5 diagnostic whole-well hash folds; no 5/5 configuration
exists. Reject. Manual editing semantics do not justify post-hoc hard
segmentation of the current learned path.

Artifacts: `summary.json`, `grid.json`; implementation:
`exp/manual_segment_projection_pilot_v1.py`. No submission.
