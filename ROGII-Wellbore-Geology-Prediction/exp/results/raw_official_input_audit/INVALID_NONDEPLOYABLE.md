# Invalid/nondeployable formation-surface result

The `formation_surface_raw` experiment is an oracle/leak diagnostic only.
`ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, and `BUDA` exist in the training
horizontal files but are absent from the competition test horizontal files.
They must not be used for inference, model packaging, submission, or meta
weights.

Likewise, `Geology` is present only in training typewell files. The
`paired_typewell_raw` family mixed legal TVT/GR summaries with illegal Geology
summaries and is not deployable as evaluated.

Only `prefix_absolute` and `complete_trajectory_raw` in this audit use schemas
available in both train and test.
