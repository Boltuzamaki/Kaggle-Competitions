# V6 E49-E50 projected-repair notebook

Authorized Kaggle GPU run for the two remaining experiments E49 and E50 as the
next version of the completed V4/V5 kernel.

Files:

- `metric_preservation_v5.py`: editable Jupytext source.
- `metric_preservation_v5.ipynb`: generated notebook.

The notebook executes E49 and E50 together. E43-E48 are completed evidence and
are not rerun. Twelve projected-repair specifications use five-fold grouped
public-unlearn validation; the best specification is retrained on all 20
public-unlearn images.
It intentionally contains:

- no Kaggle API calls;
- no competition test path;
- no submission generation;
- no leaderboard-based selection.

The validated E33 pipeline is reproduced as the frozen incumbent. A new
candidate is promoted only if it passes every gate and improves public-only
pseudo-clean maCADD over E33.

Static validation:

```powershell
.venv\Scripts\python.exe -m py_compile forensics\kaggle_v5_metric_preservation\metric_preservation_v5.py
```

Regenerate the notebook only after editing the Python source:

```powershell
.venv\Scripts\python.exe -m jupytext --to ipynb `
  --output forensics\kaggle_v5_metric_preservation\metric_preservation_v5.ipynb `
  forensics\kaggle_v5_metric_preservation\metric_preservation_v5.py
```
