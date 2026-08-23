#!/usr/bin/env bash
# Self-healing supervisor: keeps the work queue alive overnight without a human.
#
# Failure modes seen tonight that this covers:
#   * the runner dying leaves 8 cores idle and nothing says so;
#   * a job crashing early leaves the queue stalled behind it;
#   * a guard job waiting on a pattern that never clears (cost 19 idle minutes).
#
# Emits one line per event on stdout so a Monitor can surface it. Deliberately
# quiet while healthy -- silence plus a periodic HEARTBEAT means things are fine.
cd "$(dirname "$(readlink -f "$0")")/../.." || exit 1
S=scratchpad/scrape_20260804
BEAT=0
while true; do
  # 1. runner alive?
  if ! pgrep -f "python.*crn/runner\.py" >/dev/null 2>&1; then
    echo "WATCHDOG: runner is DOWN -- restarting"
    setsid nohup .venv/bin/python tools/crn/runner.py --slots 1 --poll 15 \
      >> $S/runner.log 2>&1 < /dev/null &
    sleep 20
    if pgrep -f "python.*crn/runner\.py" >/dev/null 2>&1; then
      echo "WATCHDOG: runner restarted OK"
    else
      echo "WATCHDOG: runner restart FAILED"
    fi
  fi

  # 2. any job actually burning CPU? loadavg under 2 with queue work left is a stall.
  # Stall check: a momentary load dip at job handover is NOT a stall, and firing
  # on every transition trains the reader to ignore the alarm. Require an actual
  # absence of experiment processes, confirmed twice 60s apart.
  PEND=$(grep -cvE '^#|^$' $S/queue.txt 2>/dev/null || echo 0)
  DONE=$(wc -l < $S/queue.done 2>/dev/null || echo 0)
  if [ "$DONE" -lt "$PEND" ]; then
    # A count of 1 is NOT idle: the evolution runs its promotion-confirmation
    # duel single-threaded in the parent, so the worker count legitimately drops
    # to one for minutes. Only zero processes means the queue has actually died.
    N1=$(pgrep -fc "python.*crn/[a-z_]*\.py" 2>/dev/null || echo 0)
    if [ "${N1:-0}" -eq 0 ]; then
      sleep 300
      N2=$(pgrep -fc "python.*crn/[a-z_]*\.py" 2>/dev/null || echo 0)
      if [ "${N2:-0}" -eq 0 ]; then
        echo "WATCHDOG: no experiment processes for 5min with ${DONE}/${PEND} done -- STALLED"
      fi
    fi
  fi

  # 3. surface any promotion or verdict as it happens
  for f in $S/logs/*.log; do
    [ -f "$f" ] || continue
    grep -hoE "PROMOTE \([^)]*\)|SHIPPABLE|DO NOT SHIP|BETTER  *p=[0-9.]+" "$f" 2>/dev/null \
      | tail -1 | sed "s|^|RESULT $(basename "$f"): |"
  done | sort -u

  BEAT=$((BEAT+1))
  if [ $((BEAT % 12)) -eq 0 ]; then
    echo "HEARTBEAT load=$(cut -d' ' -f1 /proc/loadavg) queue=${DONE}/${PEND} $(date +%H:%M)"
  fi
  sleep 300
done
