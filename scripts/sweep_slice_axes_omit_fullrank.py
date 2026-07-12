"""
sweep_slice_axes_omit_fullrank.py -- broader generalization of sweep_slice_axes_omit.py: instead of
only handling Slice nodes that skip exactly the batch axis, this handles ANY Slice node with
constant starts/ends/axes (opset 10-style 4-input Slice with no steps) where the data tensor's rank
is statically known via shape inference. It rewrites starts/ends to explicit full-rank int32 arrays
(filling omitted axes with the tensor's own full range) and drops the axes input entirely -- but only
keeps the rewrite if it is BYTE-STRICTLY CHEAPER (old starts+ends+axes initializer bytes vs new
starts+ends bytes), since making omitted axes explicit costs extra elements that must be offset by
dropping the axes tensor and int64->int32 downcasting.

Pure structural rewrite (same slice semantics, no data dependency) -- verified via full local audit
(checker + all train/test/arc-gen exact-match), no isolation testing needed, same class as the
narrower sweep_slice_axes_omit.py.

Run:  venv_scorer/Scripts/python.exe scripts/sweep_slice_axes_omit_fullrank.py
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
from onnx import helper, numpy_helper, shape_inference

sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import neurogolf_utils as ngu

print = functools.partial(print, flush=True)

REP = "repairs"
OUT_DIR = "scratch_onnx/slice_axes_fullrank_sweep"
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


def shape_of(shape_map, name):
    return shape_map.get(name)


def get_shape_map(model):
    try:
        inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=False)
    except Exception:
        inferred = model
    shape_map = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = []
        ok = True
        for d in vi.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                ok = False
                break
        if ok:
            shape_map[vi.name] = dims
    return shape_map


def try_rewrite(model):
    g = model.graph
    init_map = {i.name: i for i in g.initializer}
    shape_map = get_shape_map(model)
    changed = 0

    for n in g.node:
        if n.op_type != "Slice" or len(n.input) != 4:
            continue
        data_name = n.input[0]
        starts_name, ends_name, axes_name = n.input[1], n.input[2], n.input[3]
        if not all(x in init_map for x in (starts_name, ends_name, axes_name)):
            continue
        starts = numpy_helper.to_array(init_map[starts_name]).astype(np.int64).reshape(-1)
        ends = numpy_helper.to_array(init_map[ends_name]).astype(np.int64).reshape(-1)
        axes = numpy_helper.to_array(init_map[axes_name]).astype(np.int64).reshape(-1)
        if len(starts) != len(ends) or len(starts) != len(axes):
            continue
        shp = shape_of(shape_map, data_name)
        if not shp:
            continue
        rank = len(shp)
        axes = np.array([ax + rank if ax < 0 else ax for ax in axes], dtype=np.int64)
        if len(set(axes.tolist())) != len(axes) or any(ax < 0 or ax >= rank for ax in axes):
            continue
        if len(axes) == rank:
            continue  # nothing omitted, no win possible

        int32_min, int32_max = np.iinfo(np.int32).min, np.iinfo(np.int32).max
        full_st = np.zeros(rank, dtype=np.int64)
        full_en = np.array([d for d in shp], dtype=np.int64)
        ok = True
        for s, e, ax in zip(starts, ends, axes):
            if not (int32_min <= s <= int32_max) or not (int32_min <= min(e, shp[ax]) <= int32_max):
                ok = False
                break
            full_st[ax] = s
            full_en[ax] = min(e, shp[ax])
        if not ok:
            continue

        old_bytes = (
            numpy_helper.to_array(init_map[starts_name]).nbytes
            + numpy_helper.to_array(init_map[ends_name]).nbytes
            + numpy_helper.to_array(init_map[axes_name]).nbytes
        )
        new_starts = full_st.astype(np.int32)
        new_ends = full_en.astype(np.int32)
        new_bytes = new_starts.nbytes + new_ends.nbytes
        if new_bytes >= old_bytes:
            continue

        new_starts_name = n.output[0] + "_fr_starts"
        new_ends_name = n.output[0] + "_fr_ends"
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
        try:
            model = onnx.load(path)
        except Exception:
            continue
        try:
            changed = try_rewrite(model)
        except Exception as e:
            print(f"task{t:03d}: rewrite attempt raised {str(e)[:150]}")
            continue
        if changed == 0:
            continue
        try:
            onnx.checker.check_model(model, full_check=True)
        except Exception as e:
            print(f"task{t:03d}: {changed} rewrite(s) but checker FAILED: {str(e)[:150]}")
            continue
        candidates.append((t, model, changed))

    print(f"Found {len(candidates)} tasks with a valid full-rank rewrite (checker passed)")

    wins = []
    for idx, (t, model, changed) in enumerate(candidates):
        base = onnx.load(os.path.join(REP, f"task{t:03d}.onnx"))
        base_r = audit_model(base, t, f"slaxfr_base_{t}")
        if base_r[3] != "ok":
            print(f"  SKIP task{t:03d}: baseline audit not ok ({base_r[3]})")
            continue
        new_r = audit_model(model, t, f"slaxfr_new_{t}")
        if new_r[3] != "ok":
            print(f"  SKIP task{t:03d} ({changed} rewrites): {new_r[3]}")
            continue
        if new_r[1] >= base_r[1]:
            print(f"  NOWIN task{t:03d} ({changed} rewrites): {base_r[1]} -> {new_r[1]}")
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
