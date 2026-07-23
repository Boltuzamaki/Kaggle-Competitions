# V11: exact V12 anchor plus trajectory rankers

This private Kaggle GPU notebook preserves the public V12 recipe that scored
211.3036 as `M1_center`. It changes neither the V12 training objective nor the
M1 thresholds.

The auxiliary experiment snapshots the RetinaNet head at iterations
100/150/200/250/300 and measures each snapshot against the same fixed teacher
boxes. Three rules are frozen before test enumeration:

- `T1_median`: median survivor ratio across snapshots.
- `T2_q25`: conservative lower-quartile survivor ratio.
- `T3_stable`: median penalized by trajectory disagreement.

Important outputs are under `/kaggle/working/ndr_v11_v12_trajectory/`:

- `selection_lock.json`
- `trajectory_manifest.json`
- `trajectory_signals.npz`
- `submission_audit.json`
- `final_report.json`
- all M and T candidate CSVs
- `run.jsonl`

`/kaggle/working/submission.csv` is always the exact M1 anchor. The notebook
does not call the Kaggle submission API.
