#!/usr/bin/env bash
# Fetch a file list with long backoff, tolerating Kaggle 429 rate limits.
set -u
COMP=biohub-cell-tracking-during-development
DEST=${DEST:-data/comp}
LIST=$1
DELAY=${DELAY:-3}
for round in $(seq 1 40); do
  remaining=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    [ -f "$DEST/$f" ] && continue
    remaining=$((remaining+1))
    mkdir -p "$DEST/$(dirname "$f")"
    kaggle competitions download -c "$COMP" -f "$f" -p "$DEST/$(dirname "$f")" -q >/dev/null 2>&1
    sleep "$DELAY"
  done < "$LIST"
  echo "round $round: $remaining still missing at start of round"
  [ "$remaining" -eq 0 ] && break
  sleep 120
done
echo "patient_fetch done"
