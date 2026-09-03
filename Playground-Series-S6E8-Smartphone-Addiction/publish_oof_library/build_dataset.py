#!/usr/bin/env python3
"""Pack the 47 own-model prediction streams into a publishable dataset.

Each stream contributes one out-of-fold column over the 691,369 training rows
and one column over the 296,302 test rows. Stored as float32 parquet, the whole
library is small enough to attach to a notebook and load in seconds, which is
what makes the stack reproducible for a reader who cannot spend 40 GPU-hours
regenerating the base models.

Official competition data only. The loader refuses any path under
`public_outputs/`, so no third-party predictions can enter the release.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent / "data"
TARGET = "addicted_label"

sys.path.insert(0, str(ROOT / "ensemble_original"))
import mega_stack as ms  # noqa: E402
import stack_weights as sw  # noqa: E402


def main() -> None:
    train = pd.read_csv(ROOT / "train.csv", usecols=["id", TARGET])
    test = pd.read_csv(ROOT / "test.csv", usecols=["id"])
    y = train[TARGET].to_numpy("int8")

    registry = {**ms.REGISTRY, **sw.EXTRA}
    oof_cols, test_cols, rows = {}, {}, []

    for name, (op, tp, oc, tc) in registry.items():
        if "public_outputs" in op or "public_outputs" in tp:
            raise SystemExit(f"refusing to publish third-party predictions: {name}")
        opath, tpath = ROOT / op, ROOT / tp
        if not (opath.exists() and tpath.exists()):
            continue
        o = pd.read_csv(opath).set_index("id")
        t = pd.read_csv(tpath).set_index("id")
        if oc not in o.columns or tc not in t.columns:
            continue
        o = o[oc].reindex(train.id)
        t = t[tc].reindex(test.id)
        if o.isna().any() or t.isna().any():
            continue
        ov, tv = o.to_numpy(float), t.to_numpy(float)
        if not (np.isfinite(ov).all() and np.isfinite(tv).all()):
            continue
        auc = roc_auc_score(y, ov)
        if auc < 0.90:
            continue
        oof_cols[name] = ov.astype("float32")
        test_cols[name] = tv.astype("float32")
        rows.append({"stream": name, "oof_auc": round(float(auc), 7)})
        print(f"  {name:<22} {auc:.7f}")

    OUT.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(oof_cols); oof.insert(0, "id", train.id.to_numpy())
    tst = pd.DataFrame(test_cols); tst.insert(0, "id", test.id.to_numpy())
    oof.to_parquet(OUT / "oof_predictions.parquet", index=False, compression="zstd")
    tst.to_parquet(OUT / "test_predictions.parquet", index=False, compression="zstd")
    pd.DataFrame({"id": train.id, TARGET: y}).to_parquet(
        OUT / "train_labels.parquet", index=False, compression="zstd")

    index = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    index.to_csv(OUT / "stream_index.csv", index=False)
    (OUT / "manifest.json").write_text(json.dumps({
        "streams": len(rows),
        "train_rows": len(oof),
        "test_rows": len(tst),
        "provenance": "all models trained from scratch on official competition data",
        "public_predictions_used": False,
        "external_data_used": False,
        "test_id_sha256": hashlib.sha256(
            np.asarray(test.id, dtype=np.int64).tobytes()).hexdigest()[:16],
    }, indent=2) + "\n")

    size = sum(f.stat().st_size for f in OUT.glob("*")) / 1e6
    print(f"\n{len(rows)} streams | {len(oof)} train rows | {len(tst)} test rows")
    print(f"total dataset size: {size:.1f} MB")


if __name__ == "__main__":
    main()
