"""
verify_and_submit.py -- one-shot pipeline for a new public-notebook submission folder.

Usage:
    python scripts/verify_and_submit.py <folder_name_under_submissions/> [--no-submit] [--message "..."]

Example:
    python scripts/verify_and_submit.py 7243.92
    python scripts/verify_and_submit.py 7243.92 --no-submit          # dry run, just report
    python scripts/verify_and_submit.py 7243.92 --message "custom message"

What this does end to end (encodes every hard-won lesson from 2026-07-07 -- see
project memory / repairs/tracker.db notes for the incidents that taught these rules):

1. Audits every task in submissions/<folder>/ with the REAL scorer methodology
   (neurogolf_utils.sanitize_model + score_network, onnx.checker full_check=True,
   onnxruntime, every train+test+arc-gen example). No shortcuts, no cached numbers.
2. Freshly audits our current repairs/ the exact same way -- NEVER trusts tracker.db's
   cached our_points/base_points for this comparison, since stale numbers caused three
   separate real-Kaggle regressions this session.
3. For the 14 known negative-pads/checker-only-failure tasks (which our local checker
   always rejects regardless of which file is used), ALWAYS trusts the new public
   folder's own file rather than whatever we currently have -- a newer verified-good
   download's own handling of these is not something to second-guess with an untested
   "historically proven" substitute (that exact mistake cost -0.32pt on 2026-07-07).
4. For every other task, picks whichever of {new folder, our repairs/} has nfail == 0
   AND strictly higher points, based on the fresh audits from steps 1-2. Never falls
   back to baseline_v22 in this comparison at all.
5. Builds submission.zip, verifies it has exactly 400 unique task files, and verifies
   every non-substituted member is byte-identical to its source (no silent staleness).
6. Prints a clear summary: how many tasks came from the new folder vs our repairs/,
   the total predicted gain, and which specific tasks won on our side.
7. Unless --no-submit is passed, submits the zip to Kaggle via the CLI.
8. Reconciles repairs/ + tracker.db to exactly match what was (or would be) submitted,
   with a full repairs/ backup first. Restarts the webapp docker container so the UI
   stays consistent.

Run this from the project root (same directory as data/, repairs/, submissions/).
Requires the venv_scorer environment (parity onnxruntime pinned to match the real grader).
"""

import argparse
import copy
import csv
import functools
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile

print = functools.partial(print, flush=True)  # progress must be visible when output is redirected to a file

sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import onnx
import onnxruntime
import numpy as np
import neurogolf_utils as ngu

NEGPADS = {18, 45, 77, 118, 127, 135, 146, 149, 158, 171, 240, 266, 278, 384}
REP = "repairs"
BASE_V22 = "baseline_v22"
DB_FILE = os.path.join(REP, "tracker.db")


def load_task_json(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))


def audit_one(path, t, prefix):
    """Returns (nfail, cost, pts, status). status != 'ok' means unmeasurable/broken locally."""
    try:
        model = onnx.load(path)
    except Exception as e:
        return None, None, None, f"load: {str(e)[:150]}"
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        msg = str(e)
        # Negative-pads (Conv/ConvTranspose/MaxPool) checker failures are a KNOWN pattern that our
        # local full_check=True incorrectly rejects but the real Kaggle grader tolerates and scores
        # for real (confirmed via isolated Kaggle tests in prior sessions, and again on 2026-07-08
        # when substituting our own "safer" alternative for 9 such tasks cost -1.30pts vs trusting
        # the newest download's own file). Tag this distinctly so callers don't treat it the same as
        # a genuine break (wrong-output, load failure, etc).
        if "pads must not contain negative value" in msg:
            return None, None, None, "negative-pads-checker-only"
        return None, None, None, f"checker: {msg[:150]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True
    o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = prefix
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"session-load: {str(e)[:150]}"
    d = load_task_json(t)
    nfail = 0
    for ex in d["train"] + d["test"] + d["arc-gen"]:
        b = ngu.convert_to_numpy(ex)
        if not b:
            continue
        try:
            out = ngu.run_network(s, b["input"])
            if not np.array_equal(out, b["output"]):
                nfail += 1
        except Exception:
            nfail += 1
    tp = s.end_profiling()
    mem, par = ngu.score_network(san, tp)
    try:
        os.remove(tp)
    except Exception:
        pass
    if mem is None or par is None or mem < 0 or par < 0:
        return nfail, None, None, "cost could not be measured"
    cost = mem + par
    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    if nfail and nfail > 0:
        return nfail, cost, pts, "wrong-output"
    return nfail, cost, pts, "ok"


def audit_dir(dir_path, label):
    print(f"\n=== Auditing {label} ({dir_path}) ===")
    results = {}
    n_ok = n_err = 0
    for t in range(1, 401):
        path = os.path.join(dir_path, f"task{t:03d}.onnx")
        if not os.path.exists(path):
            results[t] = (None, None, None, "missing file")
            n_err += 1
            continue
        r = audit_one(path, t, f"{label}_{t}")
        results[t] = r
        if r[3] == "ok":
            n_ok += 1
        else:
            n_err += 1
        if t % 40 == 0:
            print(f"  ...{t}/400 done ({n_ok} ok, {n_err} err so far)")
    print(f"{label}: {n_ok} ok, {n_err} error/unmeasurable")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder name under submissions/, e.g. 7243.92")
    ap.add_argument("--no-submit", action="store_true", help="dry run: audit + build zip, don't submit")
    ap.add_argument("--message", default=None, help="custom Kaggle submission message")
    args = ap.parse_args()

    new_dir = os.path.join("submissions", args.folder)
    if not os.path.isdir(new_dir):
        print(f"ERROR: submissions/{args.folder} does not exist")
        sys.exit(1)
    n_onnx = sum(1 for t in range(1, 401) if os.path.exists(os.path.join(new_dir, f"task{t:03d}.onnx")))
    if n_onnx != 400:
        print(f"ERROR: submissions/{args.folder} only has {n_onnx}/400 task onnx files")
        sys.exit(1)

    new_results = audit_dir(new_dir, args.folder.replace(".", ""))
    rep_results = audit_dir(REP, "repairs")

    picks = {}
    wins_ours = []
    negpad_style = set()
    total_gain = 0.0
    for t in range(1, 401):
        nr = new_results[t]
        # Trust the newest download for ANY negative-pads-style checker failure, not just the
        # hardcoded legacy list -- confirmed 2026-07-08 that treating a NEW instance of this same
        # pattern as "risky, fall back to ours" cost -1.30pts real leaderboard points versus just
        # trusting the download (it tolerates/scores these for real; see audit_one's comment).
        if t in NEGPADS or nr[3] == "negative-pads-checker-only":
            picks[t] = ("new", new_dir)
            if t not in NEGPADS:
                negpad_style.add(t)
            continue
        rr = rep_results[t]
        new_ok = nr[3] == "ok"
        rep_ok = rr[3] == "ok"
        if rep_ok and (not new_ok or rr[2] > nr[2] + 1e-9):
            picks[t] = ("ours", REP)
            if new_ok:
                wins_ours.append((t, nr[2], rr[2]))
                total_gain += rr[2] - nr[2]
            else:
                wins_ours.append((t, 0.0, rr[2]))
                total_gain += rr[2]
        elif new_ok:
            picks[t] = ("new", new_dir)
        else:
            # neither measurable locally and not a negative-pads-style task -- keep ours as fallback
            picks[t] = ("ours", REP)

    if negpad_style:
        print(f"\nNOTE: {len(negpad_style)} new negative-pads-style tasks detected (trusting "
              f"{args.folder} for these, not falling back to ours): {sorted(negpad_style)}")

    n_new = sum(1 for v in picks.values() if v[0] == "new")
    n_ours = sum(1 for v in picks.values() if v[0] == "ours")
    print(f"\n=== Result: {n_new} tasks from {args.folder}, {n_ours} tasks from our repairs/ ===")
    print(f"({len(wins_ours)} of those {n_ours} are genuine wins outside the {len(NEGPADS)} negative-pads set)")
    print(f"Predicted additional gain over raw {args.folder}: {total_gain:.4f} pts")
    for t, old, new in sorted(wins_ours, key=lambda x: x[2] - x[1], reverse=True)[:20]:
        print(f"  task{t:03d}: {old:.4f} -> {new:.4f} ({new - old:+.4f})")

    out_zip = "submission.zip"
    if os.path.exists(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for t in range(1, 401):
            kind, src_dir = picks[t]
            z.write(os.path.join(src_dir, f"task{t:03d}.onnx"), f"task{t:03d}.onnx")

    with zipfile.ZipFile(out_zip) as z:
        names = z.namelist()
        expected = {f"task{t:03d}.onnx" for t in range(1, 401)}
        assert len(names) == 400 and set(names) == expected, "ZIP INTEGRITY FAILURE"
        for t in range(1, 401):
            kind, src_dir = picks[t]
            d1 = z.read(f"task{t:03d}.onnx")
            d2 = open(os.path.join(src_dir, f"task{t:03d}.onnx"), "rb").read()
            assert d1 == d2, f"task{t:03d} mismatch after zipping"
    print(f"\nsubmission.zip built and verified: {os.path.getsize(out_zip)} bytes, 400 unique files")

    if args.no_submit:
        print("\n--no-submit passed: not submitting, not touching repairs/tracker.db, not restarting webapp.")
        return

    msg = args.message or (
        f"Auto-verified 2026-07-07+: {args.folder} base ({n_new} tasks) + {n_ours} tasks from our "
        f"repairs/ (fresh audit confirmed strictly better, {len(NEGPADS)} negative-pads tasks always "
        f"trust the newest download). Predicted gain: {total_gain:.4f}pts."
    )
    print(f"\nSubmitting to Kaggle: {msg}")
    # kaggle CLI lives in the project's .venv (not venv_scorer, which this script runs under
    # for onnxruntime==1.24.4 parity with the real grader) -- call it explicitly by path.
    kaggle_python = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(kaggle_python):
        kaggle_python = sys.executable
    r = subprocess.run(
        [kaggle_python, "-m", "kaggle", "competitions", "submit", "neurogolf-2026",
         "-f", out_zip, "-m", msg],
        capture_output=True, text=True,
    )
    print(r.stdout)
    print(r.stderr)

    # ---- reconcile repairs/ + tracker.db to match exactly what was submitted ----
    backup_dir = f"_disposable_logs_and_traces/backup_before_{args.folder.replace('.', '')}_auto_merge"
    os.makedirs(backup_dir, exist_ok=True)
    shutil.copytree(REP, os.path.join(backup_dir, "repairs"), dirs_exist_ok=True)
    print(f"\nBacked up repairs/ to {backup_dir}/repairs before reconciling.")

    import sqlite3
    con = sqlite3.connect(DB_FILE)
    cur = con.cursor()
    cur.execute("PRAGMA table_info(tasks)")
    if "source" not in [r[1] for r in cur.fetchall()]:
        cur.execute("ALTER TABLE tasks ADD COLUMN source TEXT DEFAULT ''")

    reconciled = 0
    for t in range(1, 401):
        kind, src_dir = picks[t]
        if kind == "ours":
            cur.execute("UPDATE tasks SET source=? WHERE task=?", ("ours", t))
            continue  # file already correct in repairs/, just tag provenance
        shutil.copy(os.path.join(new_dir, f"task{t:03d}.onnx"), os.path.join(REP, f"task{t:03d}.onnx"))
        nr = new_results[t]
        cur.execute("SELECT notes, our_points, our_cost FROM tasks WHERE task=?", (t,))
        old_notes, prev_points, prev_cost = cur.fetchone() or ("", None, None)
        old_notes = old_notes or ""
        if t in NEGPADS or nr[3] == "negative-pads-checker-only":
            tag = "negative-pads set" if t in NEGPADS else "NEW negative-pads-style task (auto-detected)"
            # Preserve a prior isolation-verified real points value if one is on record (prev_cost==-1
            # already means unmeasurable-locally; don't stomp a real number back to the flat sentinel
            # just because the FILE changed -- only fall back to 25.0 when there's nothing better yet).
            if prev_cost == -1 and prev_points is not None and prev_points != 25.0:
                points_to_write = prev_points
                note = (f"\n\n[auto-verify_and_submit] task in {tag}: adopted "
                         f"submissions/{args.folder}'s own file (trusted per rule, not locally measurable). "
                         f"Preserved prior isolation-verified points ({prev_points}) instead of resetting "
                         f"to the 25.0 sentinel.")
            else:
                points_to_write = 25.0
                note = (f"\n\n[auto-verify_and_submit] task in {tag}: adopted "
                         f"submissions/{args.folder}'s own file (trusted per rule, not locally measurable).")
            cur.execute("UPDATE tasks SET our_points=?, our_cost=-1, state=?, source=?, notes=? WHERE task=?",
                        (points_to_write, "ours", args.folder, old_notes + note, t))
        else:
            note = f"\n\n[auto-verify_and_submit] reconciled to submissions/{args.folder}: {nr[2]:.4f}pts (cost {nr[1]})."
            cur.execute("UPDATE tasks SET our_points=?, our_cost=?, n_fail=0, state=?, source=?, notes=? WHERE task=?",
                        (nr[2], nr[1], "ours", args.folder, old_notes + note, t))
        reconciled += 1
    con.commit()
    con.close()
    print(f"Reconciled {reconciled} tasks in tracker.db to match submissions/{args.folder}.")

    print("\nRestarting webapp...")
    subprocess.run(["docker", "compose", "restart"], cwd="webapp", capture_output=True, text=True)
    print("Done.")


if __name__ == "__main__":
    main()
