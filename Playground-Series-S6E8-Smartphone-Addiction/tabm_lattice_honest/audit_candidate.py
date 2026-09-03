"""Audit completed TabM artifacts and optionally materialize a standalone CSV."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGET, ID = "addicted_label", "id"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tabm_lattice_honest/output")
    parser.add_argument("--family", choices=["tabm", "realmlp"], default="tabm")
    parser.add_argument("--min-oof", type=float, default=0.9689)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    train = pd.read_csv("train.csv", usecols=[ID, TARGET])
    test = pd.read_csv("test.csv", usecols=[ID])
    sample = pd.read_csv("sample_submission.csv")
    oof = pd.read_csv(out / f"oof_{args.family}_lattice.csv")
    pred = pd.read_csv(out / f"test_{args.family}_lattice.csv")
    metrics = json.loads((out / "metrics.json").read_text())

    assert list(oof.columns) == [ID, "fold", "y", "pred"]
    assert list(pred.columns) == [ID, TARGET]
    assert len(oof) == len(train) and len(pred) == len(test)
    assert oof[ID].equals(train[ID]) and pred[ID].equals(test[ID])
    assert pred[ID].equals(sample[ID])
    assert oof[ID].is_unique and pred[ID].is_unique
    assert set(oof["fold"].unique()) == {1, 2, 3, 4, 5}
    assert np.array_equal(oof["y"].to_numpy("int8"), train[TARGET].to_numpy("int8"))
    assert np.isfinite(oof["pred"]).all() and np.isfinite(pred[TARGET]).all()
    assert pred[TARGET].between(0, 1).all() and pred[TARGET].nunique() > 280_000

    auc = float(roc_auc_score(train[TARGET], oof["pred"]))
    assert abs(auc - float(metrics["oof_auc"])) < 1e-10
    assert metrics["official_data_only"] is True
    assert metrics["submission_created"] is False
    report = {
        "oof_auc": auc,
        "family": args.family,
        "minimum_oof_gate": args.min_oof,
        "passes_score_gate": auc >= args.min_oof,
        "unique_test_predictions": int(pred[TARGET].nunique()),
        "prediction_range": [float(pred[TARGET].min()), float(pred[TARGET].max())],
        "train_sha256": file_hash("train.csv"),
        "test_sha256": file_hash("test.csv"),
        "model_source_sha256": file_hash(
            f"{args.family}_lattice_honest/experiment.py"
        ),
    }
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if args.materialize:
        if auc < args.min_oof:
            raise SystemExit("candidate rejected: OOF gate not met")
        submission = sample.copy()
        submission[TARGET] = pred[TARGET].to_numpy()
        path = out / f"submission_{args.family}_lattice.csv"
        submission.to_csv(path, index=False)
        print("wrote", path)


if __name__ == "__main__":
    main()
