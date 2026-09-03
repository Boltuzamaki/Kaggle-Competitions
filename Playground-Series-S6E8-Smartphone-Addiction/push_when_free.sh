#!/usr/bin/env bash
# Push kernel directories one at a time, waiting for a Kaggle batch slot.
#
# Kaggle caps concurrent batch sessions at five, and a push that arrives over the
# cap fails outright rather than queueing. So the queue lives here: retry each
# directory until the push is accepted, then move to the next.
#
#     ./push_when_free.sh cpu_kernel_fe_screen cpu_kernel_hpo_mlp_te
#
# Usage note: it retries indefinitely by design. A kernel that never gets a slot
# is a signal that something else is stuck, not something to paper over.
set -u
cd "$(dirname "$0")"

KAGGLE=/home/boltuzamaki/.local/bin/kaggle
INTERVAL=${INTERVAL:-300}

for DIR in "$@"; do
  if [ ! -f "${DIR}/kernel-metadata.json" ]; then
    echo "[push] ${DIR} has no kernel-metadata.json, skipping"
    continue
  fi
  echo "[push] waiting for a slot for ${DIR}"
  # The CLI prints "Kernel push error: Maximum batch CPU session count..." and
  # still exits 0, so the exit status cannot be the loop condition; the success
  # string is the only reliable signal.
  while true; do
    OUT=$("${KAGGLE}" kernels push -p "${DIR}" 2>&1)
    case "${OUT}" in
      *"successfully pushed"*) break ;;
    esac
    echo "[push] ${DIR}: ${OUT##*$'\n'}"
    sleep "${INTERVAL}"
  done
  echo "[push] ${DIR}: ${OUT##*$'\n'}"
  # A push that succeeds occupies the slot it just claimed, so give the session
  # a moment to register before testing whether another one is free.
  sleep 30
done
echo "[push] ALL PUSHED"
