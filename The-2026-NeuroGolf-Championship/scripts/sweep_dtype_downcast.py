"""
sweep_dtype_downcast.py -- mechanical dtype-shrink sweep across all 400 tasks.

Two independent passes, applied per task and kept only on a fresh verified strict win:

  PASS A (int64 -> int32): convert int64 initializers whose consumers are ALL in
  slots where the ONNX spec + ORT allow int32 (Gather/GatherElements/ScatterElements/
  OneHot indices, Slice starts/ends/axes/steps). Halves those tensors' memory bytes.
  Pure structural change -- spec-level, not data-dependent.

  PASS B (fp32 -> fp16): grid values are colors 0-9 and dims <= 30, so float
  intermediates are small integers exactly representable in fp16 (exact up to 2048).
  Converts all float32 initializers (with round-trip exactness check), retargets
  Cast(to=FLOAT) -> FLOAT16, inserts a single fp16 cast after each first-touch op
  (nodes reading the fixed-f32 `input` stay f32 and keep f32 weights), and appends a
  final fp16->f32 cast for `output` if the original output was float. Halves every
  float intermediate's bytes downstream of first touch.

Every variant (A, B, A+B) is checker-validated and then fully audited (train+test+
arc-gen, real scorer, pinned parity onnxruntime). The CURRENT repairs/ file is also
re-audited fresh in the same run -- no stale tracker numbers. Only strict wins
(nfail=0 AND lower cost than the fresh current audit) are saved as candidates to
scratch_onnx/dtype_sweep/. NOTHING is merged or submitted by this script -- it only
reports, so the wins can be reviewed and then combined/isolation-checked.

Run under venv_scorer:  venv_scorer/Scripts/python.exe scripts/sweep_dtype_downcast.py
"""

import copy
import functools
import json
import math
import os
import sys

sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper
import neurogolf_utils as ngu

print = functools.partial(print, flush=True)

NEGPADS = {18, 45, 77, 118, 127, 135, 146, 149, 158, 171, 240, 266, 278, 384}
OUT_DIR = "scratch_onnx/dtype_sweep"
os.makedirs(OUT_DIR, exist_ok=True)

# (op_type, input_slot) pairs where the spec allows int32 index/param tensors
INT32_OK_SLOTS = {
    ("Gather", 1), ("GatherElements", 1), ("ScatterElements", 1), ("OneHot", 0),
    ("Slice", 1), ("Slice", 2), ("Slice", 3), ("Slice", 4),
}
# never touch initializers consumed by quantized ops (scales/zero-points are pinned dtypes)
QUANT_OPS = {"QLinearConv", "QLinearMatMul", "QuantizeLinear", "DequantizeLinear",
             "ConvInteger", "MatMulInteger"}


def load_task_json(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))


def audit_model(model, t, prefix):
    """(nfail, cost, pts, status) with real scorer; status!='ok' => unusable."""
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:120]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True
    o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = prefix
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"session: {str(e)[:120]}"
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


def consumers_map(graph):
    cons = {}
    for n in graph.node:
        for slot, name in enumerate(n.input):
            if name:
                cons.setdefault(name, []).append((n, slot))
    return cons


def pass_int32(model):
    """Convert eligible int64 initializers to int32. Returns changed count."""
    g = model.graph
    cons = consumers_map(g)
    changed = 0
    for init in g.initializer:
        if init.data_type != TensorProto.INT64:
            continue
        users = cons.get(init.name, [])
        if not users:
            continue
        if not all((n.op_type, slot) in INT32_OK_SLOTS for n, slot in users):
            continue
        if any(n.op_type in QUANT_OPS for n, _ in users):
            continue
        arr = numpy_helper.to_array(init)
        if arr.size == 0 or arr.min() < np.iinfo(np.int32).min or arr.max() > np.iinfo(np.int32).max:
            continue
        init.CopyFrom(numpy_helper.from_array(arr.astype(np.int32), init.name))
        changed += 1
    return changed


def pass_fp16(model):
    """Whole-graph fp32->fp16 except first-touch ops. Returns changed count or -1 on abort."""
    g = model.graph
    cons = consumers_map(g)

    first_touch = [n for n in g.node if "input" in n.input]
    keep_f32_inits = set()
    for n in first_touch:
        for name in n.input:
            keep_f32_inits.add(name)
    for n in g.node:
        if n.op_type in QUANT_OPS or n.op_type == "Resize":
            for name in n.input:
                keep_f32_inits.add(name)

    changed = 0
    for init in g.initializer:
        if init.data_type != TensorProto.FLOAT or init.name in keep_f32_inits:
            continue
        arr = numpy_helper.to_array(init)
        arr16 = arr.astype(np.float16)
        if not np.array_equal(arr16.astype(np.float32), arr):
            return -1  # value not exactly representable -> abort whole pass for this task
        init.CopyFrom(numpy_helper.from_array(arr16, init.name))
        changed += 1

    for n in g.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT:
                    a.i = TensorProto.FLOAT16
                    changed += 1
        if n.op_type in ("Constant", "ConstantOfShape"):
            for a in n.attribute:
                if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(a.t)
                    arr16 = arr.astype(np.float16)
                    if not np.array_equal(arr16.astype(np.float32), arr):
                        return -1
                    a.t.CopyFrom(numpy_helper.from_array(arr16, a.t.name or ""))
                    changed += 1

    # insert fp16 casts after first-touch f32 outputs (so downstream runs fp16)
    new_nodes = []
    rename = {}
    for n in g.node:
        new_nodes.append(n)
        if "input" not in n.input:
            continue
        for out in n.output:
            if out == "output":
                continue
            users = cons.get(out, [])
            if not users:
                continue
            cast_out = out + "__f16"
            new_nodes.append(helper.make_node("Cast", [out], [cast_out],
                                               name=out + "__to_f16", to=TensorProto.FLOAT16))
            rename[out] = cast_out
            changed += 1
    if rename:
        for n in new_nodes:
            if n.op_type == "Cast" and n.name.endswith("__to_f16"):
                continue
            for i, name in enumerate(n.input):
                if name in rename:
                    n.input[i] = rename[name]
        del g.node[:]
        g.node.extend(new_nodes)

    # final output: if the graph output was FLOAT, re-cast the (now fp16) producer back to f32
    out_vi = next((o for o in g.output if o.name == "output"), None)
    if out_vi is not None and out_vi.type.tensor_type.elem_type == TensorProto.FLOAT:
        producer = next((n for n in g.node if "output" in n.output), None)
        if producer is not None and "input" not in producer.input:
            for i, o in enumerate(producer.output):
                if o == "output":
                    producer.output[i] = "output__f16"
            g.node.append(helper.make_node("Cast", ["output__f16"], ["output"],
                                            name="output__to_f32", to=TensorProto.FLOAT))
            changed += 1
    return changed


def main():
    wins = []
    for t in range(1, 401):
        if t in NEGPADS:
            continue
        path = os.path.join("repairs", f"task{t:03d}.onnx")
        if not os.path.exists(path):
            continue
        try:
            base_model = onnx.load(path)
        except Exception:
            continue

        base = audit_model(base_model, t, f"dsw_base_{t}")
        if base[3] != "ok":
            continue
        base_cost = base[1]

        variants = []
        m_int = onnx.load(path)
        if pass_int32(m_int) > 0:
            variants.append(("int32", m_int))
        m_fp = onnx.load(path)
        r_fp = pass_fp16(m_fp)
        if r_fp > 0:
            variants.append(("fp16", m_fp))
        if any(v[0] == "int32" for v in variants) and r_fp != -1:
            m_both = onnx.load(path)
            if pass_int32(m_both) > 0 and pass_fp16(m_both) > 0:
                variants.append(("int32+fp16", m_both))

        best = None
        for label, vm in variants:
            r = audit_model(vm, t, f"dsw_{label.replace('+','_')}_{t}")
            if r[3] != "ok" or r[1] >= base_cost:
                continue
            if best is None or r[1] < best[2]:
                best = (label, vm, r[1], r[2])

        if best:
            label, vm, cost, pts = best
            fn = os.path.join(OUT_DIR, f"task{t:03d}.onnx")
            onnx.save(vm, fn)
            gain = pts - base[2]
            wins.append({"task": t, "variant": label, "old_cost": base_cost,
                         "new_cost": cost, "old_pts": base[2], "new_pts": pts, "gain": gain})
            print(f"WIN task{t:03d} [{label}]: cost {int(base_cost)} -> {int(cost)} ({gain:+.4f} pts)")
        if t % 25 == 0:
            print(f"...{t}/400 scanned, {len(wins)} wins so far")

    json.dump(wins, open(os.path.join(OUT_DIR, "wins.json"), "w"), indent=1)
    total = sum(w["gain"] for w in wins)
    print(f"\nDONE: {len(wins)} verified strict wins, total predicted gain {total:+.4f} pts")
    print(f"candidates saved in {OUT_DIR}/, summary in {OUT_DIR}/wins.json")
    print("Nothing merged or submitted -- review, then merge + single combined submission check.")


if __name__ == "__main__":
    main()
