#!/usr/bin/env bash
# Ten-fold Lookup Transformer V2 on the local GPU.
#
# Two facts point here. Moving XGBoost from five folds to ten was worth +0.00018,
# the largest single-model gain measured in this project. And the lookup
# transformer family carries the most stack weight while being the only family
# materially decorrelated from the trees. Combining the proven lever with the
# highest-value family is a better use of the card than a sixth five-fold seed.
#
# Roughly 78 minutes: ten folds at the ~39 minutes a five-fold run takes.
set -u
cd "$(dirname "$0")"

SEED=20260910
OUTDIR="artifacts/local_lookup_v2_10fold"
SRC="/tmp/lookup_v2_10fold.py"

if [ -f "${OUTDIR}/metrics.json" ]; then
  echo "[tenfold] already complete, nothing to do"
  exit 0
fi

sed -e "s/,20261037,5,24/,${SEED},10,24/" \
    -e "s#artifacts/local_lookup_v2_seed1037#${OUTDIR}#" \
    local_lookup_v2_seed1037/train.py > "${SRC}"

# Fail loudly rather than silently training a five-fold run under a ten-fold name.
grep -q ",${SEED},10,24" "${SRC}" || { echo "[tenfold] fold/seed substitution failed"; exit 1; }
grep -q "${OUTDIR}" "${SRC}" || { echo "[tenfold] output substitution failed"; exit 1; }

echo "[tenfold] training 10-fold lookup transformer, seed ${SEED} -> ${OUTDIR}"
.venv/bin/python -u "${SRC}" 2>&1 | grep -E "fold|oof_auc|Error|Traceback|CUDA|out of memory"
echo "[tenfold] finished with status ${PIPESTATUS[0]}"
