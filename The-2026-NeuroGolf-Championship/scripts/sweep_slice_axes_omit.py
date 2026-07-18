"""
sweep_slice_axes_omit.py -- generalizes the task190 finding: any Slice node whose `axes` input
skips ONLY axis 0 (the batch axis, provably always size 1 in this competition's fixed [1,...] input
format) can have that axis made explicit (starts=0, ends=1 prepended) and the axes tensor dropped
entirely, using int32 instead of int64. Pure structural rewrite, not data-dependent -- safe to merge
directly on a clean full audit, no isolation testing needed (same class as the ConvInteger
zero-point removal).

Only handles the SAFE subset (axes.min() == 1, i.e. batch axis skipped and nothing else) -- tasks
skipping additional leading axes (e.g. channel axis too) are left alone since their "full range"
bound isn't a universal constant and would need real shape inference to get right.

Run:  venv_scorer/Scripts/python.exe scripts/sweep_slice_axes_omit.py
"""
import copy
import functools
import json
import math
import os

import numpy as np
import onnx
import onnxruntime
from onnx import helper, numpy_helper
import sys

sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import neurogolf_utils as ngu

print = functools.partial(print, flush=True)

REP = "repairs"
OUT_DIR = "scratch_onnx/slice_axes_sweep"
os.makedirs(OUT_DIR, exist_ok=True)


def load_task_json(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))


def audit_model(model, t, prefix):
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


def try_rewrite_safe(model):
    """Cleaner rewrite avoiding self-referential input bug above."""
    g = model.graph
    init_map = {i.name: i for i in g.initializer}
    changed = 0

    for n in g.node:
        if n.op_type != "Slice" or len(n.input) != 4:
            continue
        data_name = n.input[0]
        starts_name, ends_name, axes_name = n.input[1], n.input[2], n.input[3]
        if axes_name not in init_map or starts_name not in init_map or ends_name not in init_map:
            continue
        axes = numpy_helper.to_array(init_map[axes_name])
        if axes.size == 0 or axes.min() != 1:
            continue
        starts = numpy_helper.to_array(init_map[starts_name])
        ends = numpy_helper.to_array(init_map[ends_name])

        new_starts = np.concatenate([[0], starts]).astype(np.int32)
        new_ends = np.concatenate([[1], ends]).astype(np.int32)

        new_starts_name = n.output[0] + "_starts4"
        new_ends_name = n.output[0] + "_ends4"
        g.initializer.append(numpy_helper.from_array(new_starts, name=new_starts_name))
        g.initializer.append(numpy_helper.from_array(new_ends, name=new_ends_name))
        del n.input[:]
        n.input.extend([data_name, new_starts_name, new_ends_name])
        changed += 1

    if changed:
        used = set()
        for n in g.node:
            used.update(x for x in n.input if x)
        keep = [init for init in g.initializer if init.name in used]
        del g.initializer[:]
        g.initializer.extend(keep)
    return changed


def main():
    candidates = []
    for t in range(1, 401):
        path = os.path.join(REP, f"task{t:03d}.onnx")
        if not os.path.exists(path):
            continue
        model = onnx.load(path)
        changed = try_rewrite_safe(model)
        if changed == 0:
            continue
        try:
            onnx.checker.check_model(model, full_check=True)
        except Exception as e:
            print(f"task{t:03d}: {changed} rewrite(s) but checker FAILED: {str(e)[:150]}")
            continue
        candidates.append((t, model, changed))

    print(f"Found {len(candidates)} tasks with a valid rewrite (checker passed)")

    wins = []
    for idx, (t, model, changed) in enumerate(candidates):
        base = onnx.load(os.path.join(REP, f"task{t:03d}.onnx"))
        base_r = audit_model(base, t, f"slax_base_{t}")
        if base_r[3] != "ok":
            continue
        new_r = audit_model(model, t, f"slax_new_{t}")
        if new_r[3] != "ok" or new_r[1] >= base_r[1]:
            continue
        gain = new_r[2] - base_r[2]
        fn = os.path.join(OUT_DIR, f"task{t:03d}.onnx")
        onnx.save(model, fn)
        wins.append({"task": t, "old_cost": base_r[1], "new_cost": new_r[1],
                     "old_pts": base_r[2], "new_pts": new_r[2], "gain": gain, "rewrites": changed})
        print(f"WIN task{t:03d} ({changed} rewrites): cost {base_r[1]} -> {new_r[1]} ({gain:+.5f} pts)")
        if (idx + 1) % 20 == 0:
            print(f"...{idx+1}/{len(candidates)} candidates audited, {len(wins)} wins so far")

    json.dump(wins, open(os.path.join(OUT_DIR, "wins.json"), "w"), indent=1)
    total = sum(w["gain"] for w in wins)
    print(f"\nDONE: {len(wins)} verified strict wins out of {len(candidates)} candidates, total gain {total:+.5f} pts")


if __name__ == "__main__":
    main()
