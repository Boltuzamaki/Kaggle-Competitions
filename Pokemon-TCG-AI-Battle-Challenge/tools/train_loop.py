"""
Auto-resuming training supervisor.

Runs tools/train_rl.py and, if it ever exits non-zero (crash, OOM, killed),
restarts it — train_rl.py picks up from its last per-iteration checkpoint
(model.pth.ckpt). Stops when training finishes cleanly (exit 0) or after
--max-restarts consecutive failures.

Usage:
    python tools/train_loop.py --iterations 30 --games 60 --search 16
    # weights -> agent/model.pth, checkpoint -> agent/model.pth.ckpt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN = os.path.join(_ROOT, "tools", "train_rl.py")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--search", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(_ROOT, "agent", "model.pth"))
    ap.add_argument("--max-restarts", type=int, default=50)
    args = ap.parse_args()

    cmd = [sys.executable, _TRAIN,
           "--iterations", str(args.iterations),
           "--games", str(args.games),
           "--search", str(args.search),
           "--out", args.out]

    fails = 0
    attempt = 0
    while True:
        attempt += 1
        print(f"\n===== supervisor: launch attempt {attempt} "
              f"(resumes from checkpoint if present) =====", flush=True)
        rc = subprocess.call(cmd)
        if rc == 0:
            print("===== supervisor: training finished cleanly. =====", flush=True)
            break
        fails += 1
        if fails >= args.max_restarts:
            print(f"===== supervisor: {fails} consecutive failures, giving up. =====", flush=True)
            sys.exit(1)
        wait = min(30, 5 * fails)
        print(f"===== supervisor: train exited {rc}; restarting in {wait}s "
              f"(fail {fails}/{args.max_restarts}) =====", flush=True)
        time.sleep(wait)


if __name__ == "__main__":
    main()
