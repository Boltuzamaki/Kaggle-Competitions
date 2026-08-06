"""Build a self-contained, competition-data-only TabFM Kaggle notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "kaggle_kernels" / "tabfm_seed2027_gpu"
OUT_NOTEBOOK = OUT_DIR / "tabfm_seed2027_gpu.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    cells = [
        markdown(
            """# Health Risk — fresh TabFM foundation-model experiment

This notebook trains/inferes from the competition training data only. It does
not consume public OOF files, public test probabilities, hard-label anchors, or
leaderboard feedback. TabFM is deliberately orthogonal to our tree/RealMLP
models. A fixed stratified holdout is used for an honest first-pass audit, then
the model is refit on all rows for the final test probabilities.
"""
        ),
        code(
            """!pip uninstall -y -q ydata-profiling inflect flax
!pip install -q "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git"
"""
        ),
        code(
            """import gc, json, os, time
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

SEED = 2027
AUDIT_ESTIMATORS = 4
FINAL_ESTIMATORS = 32
MAX_CONTEXT_ROWS = 8_000
PREDICT_CHUNK = 5_000
TARGET = "health_condition"
ID = "id"

np.random.seed(SEED)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
"""
        ),
        code(
            """candidate_dirs = [
    Path("/kaggle/input/competitions/playground-series-s6e7"),
    Path("/kaggle/input/playground-series-s6e7"),
]
data_dir = next(
    (p for p in candidate_dirs if (p / "train.csv").exists()),
    None,
)
if data_dir is None:
    discovered = list(Path("/kaggle/input").glob("**/train.csv"))
    matches = [
        p.parent
        for p in discovered
        if (p.parent / "test.csv").exists()
        and (p.parent / "sample_submission.csv").exists()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate competition files; train.csv paths={discovered}"
        )
    data_dir = matches[0]
print("Using data directory:", data_dir)
train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
sample = pd.read_csv(data_dir / "sample_submission.csv")

features = [c for c in train.columns if c not in (ID, TARGET)]
categorical = list(train[features].select_dtypes(include=["object", "category"]).columns)
for c in categorical:
    # A visible missing token is safer across third-party dataframe encoders.
    train[c] = train[c].fillna("__MISSING__").astype(str)
    test[c] = test[c].fillna("__MISSING__").astype(str)

X = train[features]
y = train[TARGET].astype(str).to_numpy()
X_test = test[features]
print(train.shape, test.shape, features)
print(pd.Series(y).value_counts(normalize=True))
"""
        ),
        code(
            """from tabfm import TabFMClassifier
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0

foundation = tabfm_v1_0_0.load(
    model_type="classification", device="cuda"
)

def make_classifier(seed):
    return TabFMClassifier(
        model=foundation,
        # One member at a time is intentional. Kaggle T4 runs exhaust memory
        # when TabFM retains many context caches, even with CPU offload.
        n_estimators=1,
        max_num_rows=MAX_CONTEXT_ROWS,
        random_state=seed,
        batch_size=1,
        average_logits=True,
        cache_context=True,
        maybe_quantize_kv_cache=True,
        keep_cache_on_device=False,
        verbose=True,
    )

def predict_chunked(clf, frame):
    parts = []
    for start in range(0, len(frame), PREDICT_CHUNK):
        stop = min(start + PREDICT_CHUNK, len(frame))
        print(f"predict {start:,}:{stop:,}")
        parts.append(clf.predict_proba(frame.iloc[start:stop]))
        gc.collect()
        torch.cuda.empty_cache()
    return np.vstack(parts)
"""
        ),
        code(
            """# Honest fixed holdout audit. Each one-member model is predicted,
# accumulated on CPU, and destroyed before the next context is built.
# balanced-accuracy differences of a few 1e-4 remain visible.
idx = np.arange(len(X))
fit_idx, val_idx = train_test_split(
    idx, test_size=0.10, random_state=SEED, stratify=y
)

t0 = time.time()
val_prob = None
audit_classes = None
for member in range(AUDIT_ESTIMATORS):
    print(f"Audit member {member + 1}/{AUDIT_ESTIMATORS}")
    audit_clf = make_classifier(SEED + member)
    audit_clf.fit(X.iloc[fit_idx], y[fit_idx])
    member_prob = predict_chunked(audit_clf, X.iloc[val_idx])
    if val_prob is None:
        val_prob = np.zeros_like(member_prob, dtype=np.float64)
        audit_classes = audit_clf.classes_.copy()
    val_prob += member_prob / AUDIT_ESTIMATORS
    del audit_clf, member_prob
    gc.collect()
    torch.cuda.empty_cache()

val_pred = audit_classes[val_prob.argmax(axis=1)]
audit_score = balanced_accuracy_score(y[val_idx], val_pred)
print(f"TabFM holdout balanced accuracy: {audit_score:.9f}")

audit = pd.DataFrame({ID: train.iloc[val_idx][ID].to_numpy(), "target": y[val_idx]})
for j, cls in enumerate(audit_classes):
    audit[str(cls)] = val_prob[:, j]
audit.to_csv("tabfm_holdout_probs.csv", index=False)
del val_prob
gc.collect()
torch.cuda.empty_cache()
"""
        ),
        code(
            """# Fresh final fit on every competition training row.
test_prob = None
final_classes = None
for member in range(FINAL_ESTIMATORS):
    print(f"Final member {member + 1}/{FINAL_ESTIMATORS}")
    final_clf = make_classifier(SEED + 10_000 + member)
    final_clf.fit(X, y)
    member_prob = predict_chunked(final_clf, X_test)
    if test_prob is None:
        test_prob = np.zeros_like(member_prob, dtype=np.float64)
        final_classes = final_clf.classes_.copy()
    test_prob += member_prob / FINAL_ESTIMATORS
    # Checkpoint after every member so a time-limited run is recoverable.
    checkpoint = pd.DataFrame({ID: test[ID].to_numpy()})
    for j, cls in enumerate(final_classes):
        checkpoint[str(cls)] = test_prob[:, j] * FINAL_ESTIMATORS / (member + 1)
    checkpoint.to_csv("tabfm_test_probs_partial.csv", index=False)
    del final_clf, member_prob, checkpoint
    gc.collect()
    torch.cuda.empty_cache()

prob_df = pd.DataFrame({ID: test[ID].to_numpy()})
for j, cls in enumerate(final_classes):
    prob_df[str(cls)] = test_prob[:, j]
prob_df.to_csv("tabfm_test_probs.csv", index=False)

submission = sample.copy()
submission[TARGET] = final_classes[test_prob.argmax(axis=1)]
submission.to_csv("submission.csv", index=False)

summary = {
    "experiment": "tabfm_seed2027_competition_data_only",
    "seed": SEED,
    "audit_estimators": AUDIT_ESTIMATORS,
    "final_estimators": FINAL_ESTIMATORS,
    "max_context_rows_per_estimator": MAX_CONTEXT_ROWS,
    "holdout_fraction": 0.10,
    "holdout_balanced_accuracy": float(audit_score),
    "classes": [str(x) for x in final_classes],
    "features": features,
    "elapsed_minutes": (time.time() - t0) / 60,
}
Path("training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print(submission[TARGET].value_counts(normalize=True))
"""
        ),
        code(
            """# Structural and round-trip checks before the artifact is trusted.
check = pd.read_csv("submission.csv")
assert check.shape == sample.shape
assert check[ID].equals(sample[ID])
assert not check.isna().any().any()
assert set(check[TARGET]) <= set(train[TARGET])
assert len(prob_df) == len(test)
assert np.allclose(prob_df[final_classes].sum(axis=1), 1.0, atol=1e-5)
print("All output checks passed.")
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    metadata = {
        "id": "boltuzamaki/health-risk-tabfm-seed-2027-gpu",
        "title": "Health Risk TabFM Seed 2027 GPU",
        "code_file": OUT_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": ["playground-series-s6e7"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(OUT_NOTEBOOK)


if __name__ == "__main__":
    main()
