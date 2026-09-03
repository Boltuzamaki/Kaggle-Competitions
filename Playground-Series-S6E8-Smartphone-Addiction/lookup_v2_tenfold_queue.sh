#!/usr/bin/env bash
# Additional ten-fold Lookup Transformer V2 seeds on the local GPU.
#
# The single ten-fold run (seed 20260910) is the strongest stream in the whole
# library at 0.9685902, ahead of every five-fold seed of the same family and
# every boosted tree. The family is also the only one materially decorrelated
# from the trees. Ten folds applied to that family is therefore the highest-value
# thing the card can be doing, and a bag of ten-fold seeds should beat a bag of
# five-fold ones for the same reason a ten-fold run beats a five-fold one.
#
# Roughly 78 minutes per seed. Strictly sequential: only one job on the 8GB card.
set -u
cd "$(dirname "$0")"

SEEDS=${@:-"20260911 20260912"}

for SEED in $SEEDS; do
  OUTDIR="artifacts/local_lookup_v2_10fold_s${SEED}"
  if [ -f "${OUTDIR}/metrics.json" ]; then
    echo "[tenfold-queue] seed ${SEED} already complete, skipping"
    continue
  fi
  SRC="/tmp/lookup_v2_10fold_s${SEED}.py"
  sed -e "s/,20261037,5,24/,${SEED},10,24/" \
      -e "s#artifacts/local_lookup_v2_seed1037#${OUTDIR}#" \
      local_lookup_v2_seed1037/train.py > "${SRC}"
  # Fail loudly rather than silently training a five-fold run under a ten-fold name.
  grep -q ",${SEED},10,24" "${SRC}" || { echo "[tenfold-queue] fold/seed substitution failed for ${SEED}"; exit 1; }
  grep -q "${OUTDIR}" "${SRC}" || { echo "[tenfold-queue] output substitution failed for ${SEED}"; exit 1; }
  echo "[tenfold-queue] === training 10-fold seed ${SEED} -> ${OUTDIR}"
  .venv/bin/python -u "${SRC}" 2>&1 | grep -E "fold|oof_auc|Error|Traceback|CUDA|out of memory"
  echo "[tenfold-queue] seed ${SEED} finished with status ${PIPESTATUS[0]}"
done
echo "[tenfold-queue] ALL TENFOLD SEEDS DONE"
