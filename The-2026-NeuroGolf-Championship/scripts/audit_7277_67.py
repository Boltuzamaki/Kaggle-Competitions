"""
Full 400-task audit comparing our current repairs/ against submissions/7277.67/extracted/,
using the exact scorer methodology. Never trusts the folder's implied aggregate score --
only a task-by-task real comparison decides anything.
"""
import copy
import functools
import json
import math
import os
import sqlite3
import sys

import numpy as np
import onnx
import onnxruntime

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import neurogolf_utils as ngu

REP = "repairs"
OTHER = os.path.join("submissions", "7277.67", "extracted")
OUT_FILE = os.path.join("scratch_onnx", "audit_7277_67_results.json")


def load_task_json(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))


def audit_one(path, t, prefix):
    if not os.path.exists(path):
        return None, None, None, "missing"
    try:
        model = onnx.load(path)
    except Exception as e:
        return None, None, None, f"load-fail: {str(e)[:150]}"
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        msg = str(e)
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
        return None, None, None, f"session: {str(e)[:150]}"
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
        return nfail, None, None, "cost unmeasurable (negative pads pattern)"
    cost = mem + par
    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    if nfail:
        return nfail, cost, pts, "wrong-output"
    return nfail, cost, pts, "ok"


con = sqlite3.connect(os.path.join(REP, "tracker.db"))
cur = con.cursor()
cur.execute("SELECT task, our_cost, our_points FROM tasks")
tracker = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

results = {}
clean_wins = []
negpad_matches = []

for t in range(1, 401):
    our_cost_tracked, our_pts_tracked = tracker.get(t, (None, None))
    our_pts = our_pts_tracked if our_pts_tracked is not None else 1.0
    other_path = os.path.join(OTHER, f"task{t:03d}.onnx")

    other_r = audit_one(other_path, t, f"a72776_{t}")
    status = other_r[3]

    if status == "ok":
        their_pts = other_r[2]
        delta = round(their_pts - our_pts, 4)
        results[t] = {"our_pts": round(our_pts, 4), "their_pts": round(their_pts, 4),
                      "their_status": status, "delta": delta, "their_cost": other_r[2] and other_r[1]}
        if delta > 0.0005:
            clean_wins.append((t, our_pts, their_pts, delta, other_r[1]))
    elif status == "negative-pads-checker-only":
        results[t] = {"our_pts": round(our_pts, 4), "their_pts": None,
                      "their_status": status, "delta": None}
        negpad_matches.append(t)
    else:
        results[t] = {"our_pts": round(our_pts, 4), "their_pts": None,
                      "their_status": status, "delta": None}

    if t % 40 == 0:
        print(f"...{t}/400 done, {len(clean_wins)} clean wins, {len(negpad_matches)} negpad-pattern so far")

print(f"\n=== DONE: {len(clean_wins)} clean genuine wins, {len(negpad_matches)} negative-pads-pattern tasks ===")
clean_wins.sort(key=lambda x: -x[3])
for t, our_pts, their_pts, delta, cost in clean_wins:
    print(f"  task{t:03d}: {our_pts:.4f} -> {their_pts:.4f} ({delta:+.4f}) cost={cost}")
print(f"\nnegative-pads-pattern tasks: {negpad_matches}")

json.dump(results, open(OUT_FILE, "w"), indent=1)
print(f"\nsaved to {OUT_FILE}")
