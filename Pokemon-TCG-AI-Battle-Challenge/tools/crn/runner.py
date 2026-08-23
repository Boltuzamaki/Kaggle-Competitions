"""Work-queue daemon: keeps the box saturated so no core ever idles.

Problem it solves: every experiment so far has been launched by hand, so the
machine sat idle between one finishing and the next being started -- and twice
today an experiment I believed was running had never been launched at all.

Design:
  * `queue.txt` holds one shell command per line (blank lines and `#` comments
    ignored). It is re-read every poll, so new work can be appended WHILE the
    runner is live and it will be picked up without a restart.
  * At most `--slots` jobs run at once. When one exits the next queued line
    starts immediately.
  * Every job's stdout/stderr goes to `logs/<n>_<name>.log`; a one-line record
    (exit code, wall time) is appended to `runner_status.jsonl` on completion,
    so progress is auditable without attaching to the process.
  * Completed lines are recorded in `queue.done` by content hash, so a restart
    resumes rather than re-running finished work.

Deliberately dependency-free and crash-tolerant: a job that dies takes its slot
with it and nothing else, and the runner keeps going.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SCRATCH = os.path.join(ROOT, "scratchpad", "scrape_20260804")
QUEUE = os.path.join(SCRATCH, "queue.txt")
DONE = os.path.join(SCRATCH, "queue.done")
STATUS = os.path.join(SCRATCH, "runner_status.jsonl")
LOGDIR = os.path.join(SCRATCH, "logs")


def _key(line):
    return hashlib.sha1(line.strip().encode()).hexdigest()[:12]


def _load_done():
    if not os.path.exists(DONE):
        return set()
    return {l.strip() for l in open(DONE) if l.strip()}


def _read_queue():
    if not os.path.exists(QUEUE):
        return []
    out = []
    for raw in open(QUEUE):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _name_of(cmd):
    for tok in cmd.split():
        if tok.endswith(".py"):
            return os.path.basename(tok)[:-3]
    return "job"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=2,
                    help="concurrent jobs; each may itself use several cores")
    ap.add_argument("--poll", type=float, default=10.0)
    a = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)
    done = _load_done()
    running = []          # list of (proc, cmd, key, log, t0)
    seq = 0
    idle_since = None

    print(f"runner up: slots={a.slots} queue={QUEUE}", flush=True)
    while True:
        # reap finished jobs
        still = []
        for proc, cmd, key, log, t0 in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, cmd, key, log, t0))
                continue
            dt = time.time() - t0
            with open(DONE, "a") as f:
                f.write(key + "\n")
            done.add(key)
            rec = {"cmd": cmd, "exit": rc, "seconds": round(dt, 1), "log": log}
            with open(STATUS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[done rc={rc} {dt/60:.1f}m] {cmd[:90]}", flush=True)
        running = still

        # start whatever fits
        pending = [c for c in _read_queue() if _key(c) not in done
                   and _key(c) not in {k for _p, _c, k, _l, _t in running}]
        while pending and len(running) < a.slots:
            cmd = pending.pop(0)
            seq += 1
            log = os.path.join(LOGDIR, f"{seq:03d}_{_name_of(cmd)}.log")
            fh = open(log, "w")
            proc = subprocess.Popen(cmd, shell=True, cwd=ROOT,
                                    stdout=fh, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            running.append((proc, cmd, _key(cmd), log, time.time()))
            print(f"[start {seq}] {cmd[:90]}  -> {os.path.basename(log)}", flush=True)

        if not running and not pending:
            if idle_since is None:
                idle_since = time.time()
                print("[idle] queue empty -- append to queue.txt to add work",
                      flush=True)
        else:
            idle_since = None
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
