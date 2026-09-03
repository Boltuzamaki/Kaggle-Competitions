#!/usr/bin/env bash
# Train additional fold-safe Lookup Transformer V2 seeds on the local GPU.
#
# The stream inventory is saturated on the tree side: every GBDT sits at 0.99+
# correlation with every other. The lookup transformer is the only family that
# is materially decorrelated from the trees (0.976), so widening its seed bag is
# the highest-value use of the local GPU. One seed is five folds in ~39 minutes.
#
# Runs strictly sequentially so only one job holds the 8GB card at a time.
set -u
cd "$(dirname "$0")"

# Seeds may be passed as arguments so the same queue can be re-run for a new
# batch; the original four stay the default so existing callers are unchanged.
SEEDS=${@:-"20260901 20260902 20260903 20260904"}

for SEED in $SEEDS; do
  OUTDIR="artifacts/local_lookup_v2_seed${SEED}"
  if [ -f "${OUTDIR}/metrics.json" ]; then
    echo "[queue] seed ${SEED} already complete, skipping"
    continue
  fi
  SRC="/tmp/lookup_v2_seed${SEED}.py"
  sed -e "s/,20261037,5,24/,${SEED},5,24/" \
      -e "s#artifacts/local_lookup_v2_seed1037#${OUTDIR}#" \
      local_lookup_v2_seed1037/train.py > "${SRC}"
  # Fail loudly rather than silently training the wrong seed.
  grep -q ",${SEED},5,24" "${SRC}" || { echo "[queue] seed substitution failed for ${SEED}"; exit 1; }
  grep -q "${OUTDIR}" "${SRC}" || { echo "[queue] output substitution failed for ${SEED}"; exit 1; }
  echo "[queue] === training seed ${SEED} -> ${OUTDIR}"
  .venv/bin/python -u "${SRC}" 2>&1 | grep -E "fold|oof_auc|Error|Traceback|CUDA|out of memory"
  echo "[queue] seed ${SEED} finished with status ${PIPESTATUS[0]}"
done
echo "[queue] ALL LOOKUP SEEDS DONE"
