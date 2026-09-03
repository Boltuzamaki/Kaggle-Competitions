# Private original-kernel monitor

Run one polling/ingestion pass from the project virtual environment:

```bash
KAGGLE_COMMAND='/home/boltuzamaki/.local/share/uv/tools/kaggle/bin/python kaggle_ipv4.py' \
  .venv/bin/python automation/monitor_original_kernels.py
```

Add `--watch --interval 60` to continue until every enabled kernel reaches a
terminal state. The allow-list is `kernel_registry.json`; unknown kernels and
public prediction files are never discovered automatically. The lattice entry
points only to our official-data-only private implementation, never to the
public research notebook that motivated it.

The script never calls the competition submission API. It creates
`ensemble_original/reports/candidate_submission.csv` only when guarded
same-fold crossfit improves by at least `0.00030`; otherwise any stale candidate
is removed.
