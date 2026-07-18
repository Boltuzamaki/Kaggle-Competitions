"""
Audits the (unverified, possibly rule-violating) 'neurogolf7300' public dataset against our
current repairs/ on a per-task basis, using the exact same scorer methodology as everywhere
else in this project (sanitize_model + score_network, onnx.checker full_check, all
train+test+arc-gen examples).

This is READ-ONLY analysis. Nothing here merges, submits, or otherwise uses that dataset in
our actual submission -- see the forum warning in project notes about it possibly being a
poison pill (shared/public solution data, which most competitions of this kind disallow and
detect via structural/weight fingerprinting regardless of cosmetic renaming). The output feeds
a reference-only comparison page in the webapp so we can see where it claims to be ahead,
without touching repairs/ or submission.zip.

Usage: python scripts/audit_neurogolf7300.py
Writes results to repairs/neurogolf7300_compare.json
"""
import copy
import functools
import json
import math
import os
import sys

import numpy as np
import onnx
import onnxruntime

print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import neurogolf_utils as ngu

REP = "repairs"
OTHER = os.path.join("submissions", "neurogolf7300")
OUT_FILE = os.path.join(REP, "neurogolf7300_compare.json")


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
        return nfail, None, None, "cost unmeasurable"
    cost = mem + par
    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    if nfail:
        return nfail, cost, pts, "wrong-output"
    return nfail, cost, pts, "ok"


import sqlite3

con = sqlite3.connect(os.path.join(REP, "tracker.db"))
cur = con.cursor()
cur.execute("SELECT task, our_cost, our_points FROM tasks")
tracker = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

results = {}
higher_count = 0
for t in range(1, 401):
    our_cost_tracked, our_pts_tracked = tracker.get(t, (None, None))
    ours_path = os.path.join(REP, f"task{t:03d}.onnx")
    other_path = os.path.join(OTHER, f"task{t:03d}.onnx")

    our_is_sentinel = our_cost_tracked == -1
    our_pts = our_pts_tracked if our_pts_tracked is not None else 1.0

    other_r = audit_one(other_path, t, f"a7300_{t}")
    if other_r[3] == "missing":
        results[t] = {"our_pts": our_pts, "their_pts": None, "their_status": "missing", "delta": None}
        continue

    other_ok = other_r[3] == "ok" and (other_r[0] or 0) == 0
    their_pts = other_r[2] if other_ok else None
    delta = round(their_pts - our_pts, 4) if their_pts is not None else None
    if delta is not None and delta > 0:
        higher_count += 1
    results[t] = {
        "our_pts": round(our_pts, 4),
        "their_pts": round(their_pts, 4) if their_pts is not None else None,
        "their_status": other_r[3],
        "delta": delta,
    }

    if t % 40 == 0:
        print(f"...{t}/400 done, {higher_count} tasks where the dataset claims higher so far")

print(f"\n=== DONE: {higher_count} tasks where neurogolf7300 scores higher than our current best ===")
json.dump(results, open(OUT_FILE, "w"), indent=1)
print(f"saved to {OUT_FILE}")
