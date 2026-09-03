#!/usr/bin/env bash
# Pull any completed Kaggle kernel outputs into the paths the stack expects,
# then re-fit the stack weights over whatever is now available.
#
# Safe to run repeatedly: kernels that are still running are skipped, and
# stack_weights.py ignores streams whose files are absent or incomplete.
set -u
cd "$(dirname "$0")"

fetch() {  # slug  destination
  local slug="$1" dest="$2"
  local status
  status=$(kaggle kernels status "boltuzamaki/${slug}" 2>&1 | tail -1)
  case "$status" in
    *COMPLETE*) ;;
    *) echo "[refresh] ${slug}: ${status##*status }"; return 0 ;;
  esac
  mkdir -p "$dest"
  if kaggle kernels output "boltuzamaki/${slug}" -p "$dest" >/dev/null 2>&1; then
    echo "[refresh] ${slug}: downloaded to ${dest}"
  else
    echo "[refresh] ${slug}: COMPLETE but download failed (Kaggle API flaky, retry later)"
  fi
}

fetch s6e8-foldsafe-te-catboost-private   artifacts/foldsafe_te_cat
fetch s6e8-foldsafe-te-multismooth-private artifacts/foldsafe_te_multi

echo "[refresh] re-fitting stack weights"
OMP_NUM_THREADS=${STACK_THREADS:-6} OPENBLAS_NUM_THREADS=${STACK_THREADS:-6} \
  MKL_NUM_THREADS=${STACK_THREADS:-6} \
  .venv/bin/python -u ensemble_original/stack_weights.py
