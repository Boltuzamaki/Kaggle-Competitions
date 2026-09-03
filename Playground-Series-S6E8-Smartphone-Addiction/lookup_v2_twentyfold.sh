#!/usr/bin/env bash
# Twenty-fold Lookup Transformer V2 on the local GPU.
#
# More rows per model is the most reliable lever measured in this project, and it
# has only ever been pushed to ten:
#
#   XGBoost, 5 -> 10 folds          +0.00018   (E042b)
#   lookup transformer, 5 -> 10     +0.00025   (0.96838 avg -> 0.96859 avg)
#   DAE, 5 -> 10                    +0.00003   (E066)
#
# The lookup family has the steepest response and the highest stack weight, so it
# is the one worth taking to twenty. Each fold trains on 95% of the rows instead
# of 90%, which is a smaller step than 80% -> 90% was, so the honest expectation
# is roughly half the previous gain rather than another +0.00025.
#
# About 2.6 hours per seed: twenty folds at the ~7.8 minutes a ten-fold fold takes.
set -u
cd "$(dirname "$0")"

SEEDS=${@:-"20260920"}

for SEED in $SEEDS; do
  OUTDIR="artifacts/local_lookup_v2_20fold_s${SEED}"
  if [ -f "${OUTDIR}/metrics.json" ]; then
    echo "[twentyfold] seed ${SEED} already complete, skipping"
    continue
  fi
  SRC="/tmp/lookup_v2_20fold_s${SEED}.py"
  sed -e "s/,20261037,5,24/,${SEED},20,24/" \
      -e "s#artifacts/local_lookup_v2_seed1037#${OUTDIR}#" \
      local_lookup_v2_seed1037/train.py > "${SRC}"
  # Fail loudly rather than silently training a five-fold run under a 20-fold name.
  grep -q ",${SEED},20,24" "${SRC}" || { echo "[twentyfold] fold/seed substitution failed for ${SEED}"; exit 1; }
  grep -q "${OUTDIR}" "${SRC}" || { echo "[twentyfold] output substitution failed for ${SEED}"; exit 1; }
  echo "[twentyfold] === training 20-fold seed ${SEED} -> ${OUTDIR}"
  .venv/bin/python -u "${SRC}" 2>&1 | grep -E "fold|oof_auc|Error|Traceback|CUDA|out of memory"
  echo "[twentyfold] seed ${SEED} finished with status ${PIPESTATUS[0]}"
done
echo "[twentyfold] ALL TWENTYFOLD SEEDS DONE"
