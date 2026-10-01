#!/usr/bin/env bash
# Download a list of competition files (one relative path per line on stdin or $1)
# into ./data/comp/, preserving directory structure.
set -u
COMP=biohub-cell-tracking-during-development
DEST=${DEST:-data/comp}
LIST=${1:-/dev/stdin}
n=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  out="$DEST/$(dirname "$f")"
  [ -f "$DEST/$f" ] && continue
  mkdir -p "$out"
  kaggle competitions download -c "$COMP" -f "$f" -p "$out" -q >/dev/null 2>&1
  n=$((n+1))
  if [ $((n % 200)) -eq 0 ]; then echo "fetched $n"; fi
done < "$LIST"
echo "fetched $n files -> $DEST"
