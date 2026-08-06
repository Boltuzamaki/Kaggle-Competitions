"""Resumable training loop over the model zoo.

Run it, kill it, run it again -- already-`success`-ful model ids in the
manifest (artifacts/manifest.csv) are skipped, everything else is (re)trained.
A model that raises is recorded as `failed` (with traceback) and the loop
moves on; it never brings down the run. OOF/test-prediction arrays are only
written to their final path via an atomic rename, so a hard kill mid-model
never leaves a corrupt artifact for a later resume to pick up.

Usage:
    python -m src.train                          # train everything pending
    python -m src.train --dry-run                # just report status counts
    python -m src.train --max-models 20           # train at most 20 more models this run
    python -m src.train --time-budget-min 120      # stop starting new models after 2h
    python -m src.train --retry-failed             # also retry previously-failed ids

Multi-core:
    Every individual model already fans out to all cores where the algorithm
    supports it (LightGBM/XGBoost/CatBoost/RF/ET/LogReg/SGD/kNN/Bagging/...
    all get n_jobs = os.cpu_count()). But a chunk of the zoo is genuinely
    single-threaded (sklearn's GradientBoosting, AdaBoost, MLP, DecisionTree,
    NB, LDA/QDA, Ridge) and trains one-at-a-time by default, leaving most
    cores idle. --max-parallel-workers N runs N models concurrently in
    separate processes, each capped to cpu_count()//N threads, so total
    thread usage stays bounded instead of oversubscribing. Feature arrays are
    memory-mapped (not duplicated per worker) to keep RAM usage in check --
    see src/features.py::load_feature_set(mmap_mode=...).
    python -m src.train --max-parallel-workers 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.config import load_config
from src.data import encode_target, load_raw
from src.estimators import (
    build_estimator,
    fit_estimator,
    predict_proba_ordered,
)
from src.features import build_all_feature_sets, load_feature_set
from src.folds import get_or_create_folds
from src.model_zoo import build_registry

MANIFEST_FIELDS = [
    "id", "family", "feature_set", "status", "balanced_accuracy_oof",
    "fold_scores", "n_folds", "seconds", "error", "timestamp",
]

_FEATURE_CACHE: dict[str, tuple] = {}
_MMAP_MODE = None  # set to "r" in worker processes spawned by --max-parallel-workers


def _get_feature_set(cfg, name):
    if name not in _FEATURE_CACHE:
        _FEATURE_CACHE[name] = load_feature_set(cfg, name, mmap_mode=_MMAP_MODE)
    return _FEATURE_CACHE[name]


def load_manifest(cfg) -> dict[str, dict]:
    path = cfg["paths"]["manifest_file"]
    rows = {}
    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows[row["id"]] = row
    return rows


def _ensure_manifest_header(cfg):
    path = cfg["paths"]["manifest_file"]
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()


def append_manifest_row(cfg, row: dict):
    path = cfg["paths"]["manifest_file"]
    _ensure_manifest_header(cfg)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _running_status_path(cfg) -> str:
    return os.path.join(cfg["paths"]["log_dir"], f"running_{os.getpid()}.json")


def _mark_running(cfg, spec):
    """Heartbeat file so the dashboard can show what this worker is doing
    right now, not just what it has already finished."""
    path = _running_status_path(cfg)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "id": spec.id, "family": spec.family, "feature_set": spec.feature_set,
            "pid": os.getpid(), "started_at": time.time(),
        }, f)
    os.replace(tmp, path)


def _clear_running(cfg):
    try:
        os.remove(_running_status_path(cfg))
    except OSError:
        pass


def _atomic_save(path: str, arr: np.ndarray):
    tmp_path = path + ".tmp"
    np.save(tmp_path, arr)
    tmp_path_npy = tmp_path if tmp_path.endswith(".npy") else tmp_path + ".npy"
    os.replace(tmp_path_npy, path)


def _subsample_train_idx(train_idx: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = max(1000, int(len(train_idx) * frac))
    n = min(n, len(train_idx))
    return rng.choice(train_idx, size=n, replace=False)


def train_one_model(spec, cfg, y: np.ndarray, folds: np.ndarray, n_jobs: int):
    X_train, X_test, meta = _get_feature_set(cfg, spec.feature_set)
    cat_idx = meta.get("cat_feature_indices", [])

    n_classes = len(cfg["data"]["classes"])
    n_folds = int(folds.max()) + 1

    oof = np.zeros((X_train.shape[0], n_classes), dtype=np.float32)
    test_pred = np.zeros((X_test.shape[0], n_classes), dtype=np.float32)
    fold_scores = []

    for fold in range(n_folds):
        train_idx = np.where(folds != fold)[0]
        val_idx = np.where(folds == fold)[0]

        if spec.subsample_frac:
            train_idx = _subsample_train_idx(train_idx, spec.subsample_frac, seed=spec.seed * 1000 + fold)

        X_tr, y_tr = X_train[train_idx], y[train_idx]
        X_val, y_val = X_train[val_idx], y[val_idx]

        estimator = build_estimator(spec, cfg, n_jobs, cat_feature_indices=cat_idx)
        fit_estimator(estimator, spec, X_tr, y_tr, cat_idx, X_val=X_val, y_val=y_val)

        fold_proba = predict_proba_ordered(estimator, X_val, n_classes, family=spec.family, cat_feature_indices=cat_idx)
        oof[val_idx] = fold_proba
        test_pred += predict_proba_ordered(estimator, X_test, n_classes, family=spec.family, cat_feature_indices=cat_idx) / n_folds

        fold_scores.append(float(balanced_accuracy_score(y_val, fold_proba.argmax(axis=1))))

    cv_score = float(balanced_accuracy_score(y, oof.argmax(axis=1)))
    return oof, test_pred, cv_score, fold_scores


def deterministic_shuffle(items, seed):
    items = list(items)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def _run_one(spec, cfg, y, folds, n_jobs):
    """Train one spec end-to-end (fit + save artifacts) and return its
    manifest row. Shared by the sequential loop and the parallel worker."""
    t0 = time.time()
    row = {
        "id": spec.id, "family": spec.family, "feature_set": spec.feature_set,
        "status": "", "balanced_accuracy_oof": "", "fold_scores": "",
        "n_folds": int(folds.max()) + 1, "seconds": "", "error": "",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _mark_running(cfg, spec)
    try:
        oof, test_pred, cv_score, fold_scores = train_one_model(spec, cfg, y, folds, n_jobs)
        _atomic_save(os.path.join(cfg["paths"]["oof_dir"], f"{spec.id}.npy"), oof)
        _atomic_save(os.path.join(cfg["paths"]["test_pred_dir"], f"{spec.id}.npy"), test_pred)
        row.update(status="success", balanced_accuracy_oof=f"{cv_score:.6f}",
                   fold_scores=";".join(f"{s:.6f}" for s in fold_scores), seconds=f"{time.time() - t0:.1f}")
    except Exception as e:  # noqa: BLE001 - a single bad model must never kill the run
        row.update(status="failed", error=f"{type(e).__name__}: {e}"[:500], seconds=f"{time.time() - t0:.1f}")
        log_path = os.path.join(cfg["paths"]["log_dir"], f"{spec.id}.err.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
    finally:
        _clear_running(cfg)
    return row


# --------------------------------------------------------------------------- #
# parallel worker pool -- each worker process trains one spec at a time, using
# a capped n_jobs so total threads across all workers stays <= cpu_count().
# Globals below are set once per worker process via ProcessPoolExecutor's
# `initializer`, not re-pickled per task.
# --------------------------------------------------------------------------- #
_POOL_CFG = None
_POOL_Y = None
_POOL_FOLDS = None
_POOL_N_JOBS = None


def _pool_worker_init(cfg, n_jobs_per_worker):
    global _POOL_CFG, _POOL_Y, _POOL_FOLDS, _POOL_N_JOBS, _MMAP_MODE
    _MMAP_MODE = "r"  # memory-map feature arrays: workers share OS page cache instead of duplicating ~1.7GB each
    _POOL_CFG = cfg
    _POOL_N_JOBS = n_jobs_per_worker
    train_df, _ = load_raw(cfg)
    _POOL_Y = encode_target(train_df, cfg)
    _POOL_FOLDS = get_or_create_folds(cfg)


def _pool_worker_run(spec):
    return _run_one(spec, _POOL_CFG, _POOL_Y, _POOL_FOLDS, _POOL_N_JOBS)


def _run_sequential(cfg, pending, y, folds, n_jobs, time_budget_min):
    run_start = time.time()
    for i, spec in enumerate(pending):
        if time_budget_min is not None and (time.time() - run_start) / 60.0 > time_budget_min:
            print(f"[train] time budget of {time_budget_min} min reached, stopping "
                  f"(started {i}/{len(pending)} of this run's queue)")
            break

        row = _run_one(spec, cfg, y, folds, n_jobs)
        if row["status"] == "success":
            print(f"[{i + 1}/{len(pending)}] {spec.id:70s} CV={float(row['balanced_accuracy_oof']):.5f} "
                  f"({row['seconds']}s)")
        else:
            print(f"[{i + 1}/{len(pending)}] {spec.id:70s} FAILED: {row['error']}")
        append_manifest_row(cfg, row)


def _drain_futures(cfg, futures, total):
    done = 0
    for future in as_completed(futures):
        spec = futures[future]
        done += 1
        try:
            row = future.result()
        except Exception as e:  # noqa: BLE001 - a worker crash must not kill the pool
            row = {
                "id": spec.id, "family": spec.family, "feature_set": spec.feature_set,
                "status": "failed", "balanced_accuracy_oof": "", "fold_scores": "",
                "n_folds": "", "seconds": "", "error": f"worker crash: {type(e).__name__}: {e}"[:500],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        if row["status"] == "success":
            print(f"[{done}/{total}] {spec.id:70s} CV={float(row['balanced_accuracy_oof']):.5f} "
                  f"({row['seconds']}s)")
        else:
            print(f"[{done}/{total}] {spec.id:70s} FAILED: {row['error']}")
        append_manifest_row(cfg, row)


# a single 450-col CatBoost fit on GPU was observed to use ~7GB of an 8GB
# card by itself under concurrent load -- GPU-capable families always get
# serialized to ONE worker, regardless of --max-parallel-workers, to avoid
# OOM-ing the GPU. CPU-only families still run with full worker parallelism
# concurrently in a separate pool, so GPU and CPU work overlap safely.
GPU_LANE_WORKERS = 1
GPU_LANE_RESERVED_THREADS = 4  # CPU-side data-prep threads for the GPU worker


def _run_parallel(cfg, pending, max_workers):
    from src.config import is_gpu
    from src.estimators import GPU_CAPABLE_FAMILIES

    if is_gpu(cfg):
        gpu_specs = [s for s in pending if s.family in GPU_CAPABLE_FAMILIES]
        cpu_specs = [s for s in pending if s.family not in GPU_CAPABLE_FAMILIES]
        cpu_n_jobs = max(1, ((os.cpu_count() or 4) - GPU_LANE_RESERVED_THREADS) // max(1, max_workers))
        print(f"[train] GPU mode: 1 GPU worker (serialized -- concurrent GPU fits risk VRAM OOM) "
              f"handling {len(gpu_specs)} GPU-capable models, "
              f"+ {max_workers} CPU workers x {cpu_n_jobs} threads handling {len(cpu_specs)} CPU-only models, "
              f"running concurrently (cpu_count={os.cpu_count()})")

        with ProcessPoolExecutor(
            max_workers=GPU_LANE_WORKERS, initializer=_pool_worker_init, initargs=(cfg, GPU_LANE_RESERVED_THREADS)
        ) as gpu_exec, ProcessPoolExecutor(
            max_workers=max_workers, initializer=_pool_worker_init, initargs=(cfg, cpu_n_jobs)
        ) as cpu_exec:
            futures = {gpu_exec.submit(_pool_worker_run, spec): spec for spec in gpu_specs}
            futures.update({cpu_exec.submit(_pool_worker_run, spec): spec for spec in cpu_specs})
            _drain_futures(cfg, futures, len(pending))
    else:
        n_jobs_per_worker = max(1, (os.cpu_count() or 4) // max_workers)
        print(f"[train] parallel mode: {max_workers} workers x {n_jobs_per_worker} threads/worker "
              f"(cpu_count={os.cpu_count()}); feature arrays memory-mapped to bound RAM use")

        with ProcessPoolExecutor(
            max_workers=max_workers, initializer=_pool_worker_init, initargs=(cfg, n_jobs_per_worker)
        ) as executor:
            futures = {executor.submit(_pool_worker_run, spec): spec for spec in pending}
            _drain_futures(cfg, futures, len(pending))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-models", type=int, default=None)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--max-parallel-workers", type=int, default=None,
                     help="train N models concurrently in separate processes (default: config.yaml run.max_parallel_workers, or 1 = sequential)")
    ap.add_argument("--only-family", type=str, default=None,
                     help="comma-separated family names -- restrict this run to just these families "
                          "(e.g. for splitting the registry across separate machines)")
    ap.add_argument("--ids-file", type=str, default=None,
                     help="path to a text file of explicit model ids (one per line) -- restricts this "
                          "run to exactly those ids. Use this (not --max-models) to split the registry "
                          "across independent machines with their own empty manifest: --max-models alone "
                          "would have every machine redundantly train the same leading slice of the "
                          "shuffled registry, since each has nothing marked completed/failed yet to "
                          "differentiate what's 'pending'. Composes with --only-family (both filters apply).")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    max_models = args.max_models if args.max_models is not None else cfg["run"].get("max_models")
    time_budget_min = args.time_budget_min if args.time_budget_min is not None else cfg["run"].get("time_budget_minutes")
    n_jobs_cfg = cfg["run"].get("n_jobs", -1)
    n_jobs = os.cpu_count() if n_jobs_cfg in (-1, None) else n_jobs_cfg
    max_parallel_workers = args.max_parallel_workers if args.max_parallel_workers is not None else cfg["run"].get("max_parallel_workers", 1)

    print(f"[train] device.mode = {cfg['device']['mode']}")
    build_all_feature_sets(cfg)
    folds = get_or_create_folds(cfg)
    train_df, _ = load_raw(cfg)
    y = encode_target(train_df, cfg)

    registry = build_registry(cfg)
    if cfg["run"].get("shuffle_order", True):
        registry = deterministic_shuffle(registry, seed=cfg["seed"])

    manifest = load_manifest(cfg)
    completed_ids = {i for i, r in manifest.items() if r["status"] == "success"}
    failed_ids = {i for i, r in manifest.items() if r["status"] == "failed"}

    pending = [s for s in registry if s.id not in completed_ids and (args.retry_failed or s.id not in failed_ids)]

    only_family = None
    if args.only_family:
        only_family = {f.strip() for f in args.only_family.split(",") if f.strip()}
        unknown = only_family - {s.family for s in registry}
        if unknown:
            raise SystemExit(f"--only-family: unknown family name(s): {sorted(unknown)}")
        pending = [s for s in pending if s.family in only_family]

    ids_filter = None
    if args.ids_file:
        with open(args.ids_file, encoding="utf-8") as f:
            ids_filter = {line.strip() for line in f if line.strip()}
        unknown = ids_filter - {s.id for s in registry}
        if unknown:
            raise SystemExit(f"--ids-file: {len(unknown)} id(s) not found in the registry, e.g. {sorted(unknown)[:5]}")
        pending = [s for s in pending if s.id in ids_filter]

    if max_models is not None:
        pending = pending[:max_models]

    print(f"[train] registry={len(registry)} completed={len(completed_ids)} "
          f"failed={len(failed_ids)} pending_this_run={len(pending)}"
          + (f" (--only-family {sorted(only_family)})" if only_family else "")
          + (f" (--ids-file: {len(ids_filter)} ids requested)" if ids_filter else ""))

    if args.dry_run:
        return

    if max_parallel_workers and max_parallel_workers > 1:
        if time_budget_min is not None:
            print("[train] note: --time-budget-min is not enforced in parallel mode "
                  "(all pending tasks up to --max-models are submitted up front)")
        _run_parallel(cfg, pending, max_parallel_workers)
    else:
        _run_sequential(cfg, pending, y, folds, n_jobs, time_budget_min)

    print("[train] run complete.")


if __name__ == "__main__":
    main()
