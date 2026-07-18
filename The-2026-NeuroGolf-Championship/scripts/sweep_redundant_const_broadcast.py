"""
sweep_redundant_const_broadcast.py -- generalizes the task084 broadcast-offsets finding: any
initializer with an axis that is fully redundant (every slice along that axis is identical, size
N>1), feeding a numpy-style broadcasting op (Add/Mul/Sub/etc.), where a SIBLING input to that same
op already carries the full size N along that axis, can have the initializer shrunk to size 1 on
that axis with ZERO added cost -- the op's output shape is already fixed by the sibling, so nothing
downstream changes, we just stop paying for redundant params.

Only handles initializers used by exactly ONE consumer node, to keep the safety analysis simple.
Pure structural/data-identity rewrite (same values, same broadcast semantics) -- verified via full
local audit (checker + all train/test/arc-gen exact-match), no isolation testing needed.

Run:  venv_scorer/Scripts/python.exe scripts/sweep_redundant_const_broadcast.py
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
OUT_DIR = "scratch_onnx/redundant_const_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

BROADCAST_OPS = {
    "Add", "Sub", "Mul", "Div", "Pow", "Equal", "Less", "Greater",
    "LessOrEqual", "GreaterOrEqual", "And", "Or", "Xor", "Where", "Min", "Max", "Mod",
}


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


def try_shrink(model):
    g = model.graph
    init_map = {i.name: i for i in g.initializer}

    use_count = {}
    for n in g.node:
        for inp in n.input:
            if inp in init_map:
                use_count[inp] = use_count.get(inp, 0) + 1

    shape_map = get_shape_map(model)
    changed = 0

    for n in g.node:
        if n.op_type not in BROADCAST_OPS:
            continue
        for slot, inp_name in enumerate(n.input):
            if inp_name not in init_map:
                continue
            if use_count.get(inp_name, 0) != 1:
                continue
            init = init_map[inp_name]
            if init.data_type not in (
                onnx.TensorProto.FLOAT, onnx.TensorProto.INT32, onnx.TensorProto.INT64,
                onnx.TensorProto.UINT8, onnx.TensorProto.INT8, onnx.TensorProto.BOOL,
            ):
                continue
            arr = numpy_helper.to_array(init)
            if arr.ndim == 0:
                continue

            sibling_shapes = []
            for j, other in enumerate(n.input):
                if j == slot:
                    continue
                if other in init_map:
                    sibling_shapes.append(tuple(numpy_helper.to_array(init_map[other]).shape))
                elif other in shape_map:
                    sibling_shapes.append(tuple(shape_map[other]))
            if not sibling_shapes:
                continue

            new_arr = arr
            local_changed = False
            for axis in range(new_arr.ndim):
                dim_size = new_arr.shape[axis]
                if dim_size <= 1:
                    continue
                first = np.take(new_arr, 0, axis=axis)
                redundant = all(
                    np.array_equal(np.take(new_arr, k, axis=axis), first)
                    for k in range(1, dim_size)
                )
                if not redundant:
                    continue
                pos_from_end = new_arr.ndim - axis
                ok_sibling = False
                for sh in sibling_shapes:
                    if len(sh) >= pos_from_end:
                        sib_axis = len(sh) - pos_from_end
                        if sh[sib_axis] == dim_size:
                            ok_sibling = True
                            break
                if not ok_sibling:
                    continue
                new_arr = np.take(new_arr, [0], axis=axis)
                local_changed = True

            if local_changed and new_arr.size < arr.size:
                new_init = numpy_helper.from_array(new_arr.astype(arr.dtype), name=inp_name)
                for idx_i, ini in enumerate(g.initializer):
                    if ini.name == inp_name:
                        del g.initializer[idx_i]
                        g.initializer.insert(idx_i, new_init)
                        break
                changed += 1

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
            changed = try_shrink(model)
        except Exception as e:
            print(f"task{t:03d}: shrink attempt raised {str(e)[:150]}")
            continue
        if changed == 0:
            continue
        try:
            onnx.checker.check_model(model, full_check=True)
        except Exception as e:
            print(f"task{t:03d}: {changed} shrink(s) but checker FAILED: {str(e)[:150]}")
            continue
        candidates.append((t, model, changed))

    print(f"Found {len(candidates)} tasks with a valid redundant-const shrink (checker passed)")

    wins = []
    for idx, (t, model, changed) in enumerate(candidates):
        base = onnx.load(os.path.join(REP, f"task{t:03d}.onnx"))
        base_r = audit_model(base, t, f"rconst_base_{t}")
        if base_r[3] != "ok":
            print(f"  SKIP task{t:03d}: baseline audit not ok ({base_r[3]})")
            continue
        new_r = audit_model(model, t, f"rconst_new_{t}")
        if new_r[3] != "ok":
            print(f"  SKIP task{t:03d} ({changed} shrinks): {new_r[3]}")
            continue
        if new_r[1] >= base_r[1]:
            print(f"  NOWIN task{t:03d} ({changed} shrinks): {base_r[1]} -> {new_r[1]}")
            continue
        gain = new_r[2] - base_r[2]
        fn = os.path.join(OUT_DIR, f"task{t:03d}.onnx")
        onnx.save(model, fn)
        wins.append({"task": t, "old_cost": base_r[1], "new_cost": new_r[1],
                     "old_pts": base_r[2], "new_pts": new_r[2], "gain": gain, "shrinks": changed})
        print(f"WIN task{t:03d} ({changed} shrinks): cost {base_r[1]} -> {new_r[1]} ({gain:+.5f} pts)")
        if (idx + 1) % 20 == 0:
            print(f"...{idx+1}/{len(candidates)} candidates audited, {len(wins)} wins so far")

    json.dump(wins, open(os.path.join(OUT_DIR, "wins.json"), "w"), indent=1)
    total = sum(w["gain"] for w in wins)
    print(f"\nDONE: {len(wins)} verified strict wins out of {len(candidates)} candidates, total gain {total:+.5f} pts")


if __name__ == "__main__":
    main()
