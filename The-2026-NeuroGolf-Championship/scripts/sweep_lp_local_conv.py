"""
sweep_lp_local_conv.py -- generalizes the task192 discovery: for a task whose rule is a
DETERMINISTIC function of a small local neighborhood (KxK, checked at K=1 and K=3), solve a
per-output-channel linear-separation problem via LP (scipy linprog) to find a single Conv
(weight [10,10,K,K] + bias [10]) that reproduces the exact rule for every real example in ONE node.
Since a single input->output node has zero memory cost (both exempt), cost = pure param count
(10*10*K*K + 10), which beats most existing multi-node implementations when the rule really is local.

Two-stage cheap-to-expensive check per task:
  1. K=1 (pure per-cell color remap, params=110) -- tried first, cheapest if it works.
  2. K=3 (params=910) -- tried only if K=1's patches aren't deterministic or not separable.
Larger K (5x5 = 2510 params) is NOT attempted here: at that size params often exceed what a
multi-node structural implementation already achieves, so it's rarely a net win -- skip for now,
revisit manually per-task if a promising K=3 near-miss is found.

Determinism check (same patch always maps to the same target) proves/disproves locality outright.
LP infeasibility (linprog fails) means the rule may be local in RECEPTIVE FIELD but not LINEARLY
separable per-channel -- also skipped (no attempt to add a hidden layer / kernelize here).

Run:  venv_scorer/Scripts/python.exe scripts/sweep_lp_local_conv.py
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
from onnx import TensorProto, helper, numpy_helper
from scipy.optimize import linprog

sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import neurogolf_utils as ngu

print = functools.partial(print, flush=True)

REP = "repairs"
OUT_DIR = "scratch_onnx/lp_local_conv_sweep"
os.makedirs(OUT_DIR, exist_ok=True)


def load_task_json(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))


def collect_unique_patches(data, k):
    pad = k // 2
    samples = {}
    for example in data["train"] + data["test"] + data["arc-gen"]:
        inp, out = example["input"], example["output"]
        h, w = len(inp), len(inp[0]) if inp else 0
        if h == 0 or w == 0 or h > 30 or w > 30:
            continue
        x = np.zeros((10, 30 + 2 * pad, 30 + 2 * pad), dtype=np.uint8)
        y = np.zeros((10, 30, 30), dtype=np.uint8)
        for row, values in enumerate(inp):
            for col, color in enumerate(values):
                x[color, row + pad, col + pad] = 1
        for row, values in enumerate(out):
            for col, color in enumerate(values):
                y[color, row, col] = 1
        for row in range(30):
            for col in range(30):
                patch = tuple(x[:, row:row + k, col:col + k].reshape(-1))
                target = tuple(y[:, row, col])
                prev = samples.setdefault(patch, target)
                if prev != target:
                    return None, None  # not deterministic at this K
    if not samples:
        return None, None
    patches = np.asarray(list(samples), dtype=np.float64)
    targets = np.asarray(list(samples.values()), dtype=np.float64)
    return patches, targets


def solve_separator(patches, targets, k):
    n_feat = patches.shape[1]
    n_var = 2 * (n_feat + 1)
    augmented = np.column_stack((patches, np.ones(len(patches))))
    separators = []
    for channel in range(10):
        signs = targets[:, channel] * 2.0 - 1.0
        if np.all(signs > 0) or np.all(signs < 0):
            # degenerate: constant channel, no separation needed
            coeff = np.zeros(n_feat + 1)
            coeff[-1] = 1.0 if np.all(signs > 0) else -1.0
            separators.append(coeff)
            continue
        constraints = np.column_stack((-signs[:, None] * augmented, signs[:, None] * augmented))
        objective = np.ones(n_var)
        objective[n_feat] = objective[n_var - 1] = 0.05
        try:
            result = linprog(objective, A_ub=constraints, b_ub=-np.ones(len(patches)),
                              bounds=[(0, None)] * n_var, method="highs")
        except Exception:
            return None
        if not result.success:
            return None
        coeff = result.x[:n_feat + 1] - result.x[n_feat + 1:]
        coeff[np.abs(coeff) < 1e-7] = 0.0
        separators.append(coeff)
    return np.asarray(separators, dtype=np.float32)


def build_model(separators, k):
    n_feat = 10 * k * k
    weights = separators[:, :n_feat].reshape(10, 10, k, k)
    bias = separators[:, -1]
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    pad = k // 2
    node = helper.make_node("Conv", ["input", "weights", "bias"], ["output"], pads=[pad, pad, pad, pad])
    graph = helper.make_graph(
        [node], f"lp_conv_k{k}", [x], [y],
        [numpy_helper.from_array(weights, "weights"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model, full_check=True)
    return model


def audit_one(model, t):
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:150]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True
    o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = f"lpc_{t}"
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


def try_task(t, cur_cost):
    data = load_task_json(t)
    for k in (1, 3):
        try:
            patches, targets = collect_unique_patches(data, k)
        except Exception:
            continue
        if patches is None:
            continue
        n_params = 10 * 10 * k * k + 10
        if n_params >= cur_cost:
            continue  # can't possibly win, skip the expensive LP solve
        try:
            separators = solve_separator(patches, targets, k)
        except Exception:
            continue
        if separators is None:
            continue
        try:
            model = build_model(separators, k)
        except Exception:
            continue
        nfail, cost, pts, status = audit_one(model, t)
        if status == "ok" and nfail == 0 and cost < cur_cost:
            return k, model, cost, pts
    return None, None, None, None


con_path = os.path.join(REP, "tracker.db")
import sqlite3
con = sqlite3.connect(con_path)
cur = con.cursor()
cur.execute("SELECT task, our_cost, our_points FROM tasks")
tracker = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

wins = []
for t in range(1, 401):
    cur_cost, cur_pts = tracker.get(t, (None, None))
    if cur_cost is None or cur_cost <= 0:
        continue  # sentinel/unmeasurable tasks -- skip, can't fairly compare
    try:
        k, model, cost, pts = try_task(t, cur_cost)
    except Exception as e:
        print(f"task{t:03d}: sweep error {str(e)[:150]}")
        continue
    if model is not None:
        fn = os.path.join(OUT_DIR, f"task{t:03d}.onnx")
        onnx.save(model, fn)
        gain = pts - cur_pts
        wins.append({"task": t, "k": k, "old_cost": cur_cost, "new_cost": cost,
                     "old_pts": cur_pts, "new_pts": pts, "gain": gain})
        print(f"WIN task{t:03d} (K={k}): cost {cur_cost} -> {cost} ({gain:+.4f} pts)")
    if t % 40 == 0:
        print(f"...{t}/400 scanned, {len(wins)} wins so far")

json.dump(wins, open(os.path.join(OUT_DIR, "wins.json"), "w"), indent=1)
total = sum(w["gain"] for w in wins)
print(f"\nDONE: {len(wins)} verified strict wins, total gain {total:+.4f} pts")
