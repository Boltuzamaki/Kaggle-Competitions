"""Build a memory-bounded TabPFN-3 Kaggle GPU experiment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "kaggle_kernels/tabpfn3_gpu"

SOURCE = r'''
!pip install -q "tabpfn==8.1.0" kagglehub

import gc, json, os, time
from pathlib import Path
import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
os.environ["TABPFN_MAX_BATCHED_TEST_ROWS"] = "4096"

SEED = 2027
TARGET = "health_condition"
ID = "id"
CLASSES = np.array(["at-risk", "fit", "unhealthy"])
CONTEXT_ROWS = 50_000
AUDIT_ESTIMATORS = 4
FINAL_ESTIMATORS = 8
CHUNK = 4_096

base = Path("/kaggle/input/competitions/playground-series-s6e7")
train = pd.read_csv(base / "train.csv")
test = pd.read_csv(base / "test.csv")
sample = pd.read_csv(base / "sample_submission.csv")
features = [c for c in train.columns if c not in (ID, TARGET)]
categorical = list(
    train[features].select_dtypes(include=["object", "category"]).columns
)
for column in categorical:
    train[column] = train[column].fillna("__MISSING__").astype(str)
    test[column] = test[column].fillna("__MISSING__").astype(str)
X = train[features]
y_text = train[TARGET].astype(str).to_numpy()
X_test = test[features]
cat_indices = [features.index(column) for column in categorical]
prior = (
    train[TARGET].value_counts(normalize=True)
    .reindex(CLASSES).to_numpy()
)
print(train.shape, test.shape, categorical, prior)

attached_model_dir = Path(
    "/kaggle/input/models/prior-labsai/tabpfn-3/pytorch/default/1"
)
if attached_model_dir.exists():
    model_dir = attached_model_dir
else:
    # Interactive/local fallback. Kaggle batch sessions cannot dynamically
    # attach a new model, so the kernel metadata pins version 1 below.
    model_dir = Path(
        kagglehub.model_download("prior-labsai/tabpfn-3/pyTorch/default")
    )
os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_dir)
print("TabPFN-3 cache:", model_dir)

from tabpfn import TabPFNClassifier

def make_model(n_estimators, seed):
    return TabPFNClassifier(
        device="cuda",
        n_estimators=n_estimators,
        random_state=seed,
        fit_mode="low_memory",
        memory_saving_mode="auto",
        ignore_pretraining_limits=True,
        categorical_features_indices=cat_indices,
        inference_config={"SUBSAMPLE_SAMPLES": CONTEXT_ROWS},
        show_progress_bar=True,
    )

def predict_chunked(model, frame):
    parts = []
    for start in range(0, len(frame), CHUNK):
        stop = min(start + CHUNK, len(frame))
        print(f"predict {start:,}:{stop:,}")
        parts.append(model.predict_proba(frame.iloc[start:stop]))
        gc.collect()
        torch.cuda.empty_cache()
    return np.vstack(parts)

idx = np.arange(len(X))
fit_idx, val_idx = train_test_split(
    idx, test_size=0.10, random_state=SEED, stratify=y_text
)
t0 = time.time()
audit = make_model(AUDIT_ESTIMATORS, SEED)
audit.fit(X.iloc[fit_idx], y_text[fit_idx])
val_raw = predict_chunked(audit, X.iloc[val_idx])
audit_classes = audit.classes_.astype(str)
class_positions = [int(np.where(audit_classes == cls)[0][0]) for cls in CLASSES]
val_prob = val_raw[:, class_positions]
del audit, val_raw
gc.collect()
torch.cuda.empty_cache()

betas = np.round(np.arange(0.75, 1.251, 0.025), 3)
beta_scores = {}
for beta in betas:
    corrected = val_prob / (prior[None, :] ** beta)
    pred = CLASSES[corrected.argmax(1)]
    beta_scores[str(beta)] = float(
        balanced_accuracy_score(y_text[val_idx], pred)
    )
best_beta = float(max(beta_scores, key=beta_scores.get))
audit_score = beta_scores[str(best_beta)]
raw_score = float(
    balanced_accuracy_score(
        y_text[val_idx], CLASSES[val_prob.argmax(1)]
    )
)
print("raw holdout", raw_score, "corrected", audit_score, "beta", best_beta)

holdout = pd.DataFrame({
    ID: train.iloc[val_idx][ID].to_numpy(),
    "target": y_text[val_idx],
})
for j, cls in enumerate(CLASSES):
    holdout[cls] = val_prob[:, j]
holdout.to_csv("holdout_preds.csv", index=False)

final_model = make_model(FINAL_ESTIMATORS, SEED + 10_000)
final_model.fit(X, y_text)
test_raw = predict_chunked(final_model, X_test)
final_classes = final_model.classes_.astype(str)
class_positions = [
    int(np.where(final_classes == cls)[0][0]) for cls in CLASSES
]
test_prob = test_raw[:, class_positions]
corrected_test = test_prob / (prior[None, :] ** best_beta)

probability = pd.DataFrame({ID: test[ID].to_numpy()})
for j, cls in enumerate(CLASSES):
    probability[cls] = test_prob[:, j]
probability.to_csv("test_preds.csv", index=False)
submission = sample.copy()
submission[TARGET] = CLASSES[corrected_test.argmax(1)]
submission.to_csv("submission.csv", index=False)

summary = {
    "experiment": "tabpfn3_kaggle_model",
    "model_cache_dir": str(model_dir),
    "seed": SEED,
    "context_rows": CONTEXT_ROWS,
    "audit_estimators": AUDIT_ESTIMATORS,
    "final_estimators": FINAL_ESTIMATORS,
    "holdout_fraction": 0.10,
    "holdout_raw_balanced_accuracy": raw_score,
    "holdout_corrected_balanced_accuracy": audit_score,
    "best_beta": best_beta,
    "beta_scores": beta_scores,
    "elapsed_minutes": (time.time() - t0) / 60,
    "uses_only_competition_data": True,
}
Path("training_summary.json").write_text(json.dumps(summary, indent=2))
assert submission[ID].equals(sample[ID])
assert not submission.isna().any().any()
print(json.dumps(summary, indent=2))
print(submission[TARGET].value_counts(normalize=True))
'''


def main() -> None:
    compile(
        "\n".join(
            line for line in SOURCE.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        ),
        "tabpfn3_gpu",
        "exec",
    )
    DEST.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "id": "tabpfn3",
            "metadata": {},
            "outputs": [],
            "source": SOURCE.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (DEST / "tabpfn3_gpu.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n"
    )
    metadata = {
        "id": "boltuzamaki/health-risk-tabpfn3-gpu",
        "title": "Health Risk TabPFN3 GPU",
        "code_file": "tabpfn3_gpu.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "kernel_sources": [],
        "model_sources": ["prior-labsai/tabpfn-3/PyTorch/default/1"],
        "competition_sources": ["playground-series-s6e7"],
        "machine_shape": "NvidiaTeslaT4",
    }
    (DEST / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
