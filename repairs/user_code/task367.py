import json
import os as _os
import copy as _copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F = TensorProto.FLOAT
I64 = TensorProto.INT64
I32 = TensorProto.INT32

R_ITERS = 60
SENT = 100000


def _K(n, a, d):
    return numpy_helper.from_array(np.array(a, dtype=d), name=n)


def _set_out_shape(m, dims):
    tt = m.graph.output[0].type.tensor_type
    tt.elem_type = TensorProto.FLOAT
    del tt.shape.dim[:]
    for d in dims:
        tt.shape.dim.add().dim_value = d


def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p):
            return p
    raise FileNotFoundError("task%03d.json" % t)


def _reps(t, k=8):
    d = json.load(open(_resolve_task_json(t)))
    exs = sorted(d["train"] + d["test"] + d["arc-gen"], key=lambda e: (len(e["input"]), len(e["input"][0])))
    idx = set([0, len(exs) - 1]) | {int(j * (len(exs) - 1) / (k - 1)) for j in range(1, k - 1)}
    out = []
    for i in sorted(idx):
        g = exs[i]["input"]
        a = np.zeros((1, 10, 30, 30), np.float32)
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                a[0, v, r, c] = 1.0
        out.append(a)
    return out


def _bake(m, t):
    import onnxruntime as _ort

    inf = onnx.shape_inference.infer_shapes(_copy.deepcopy(m), strict_mode=True)

    def sym(vi):
        return any(dd.HasField("dim_param") or not dd.HasField("dim_value") for dd in vi.type.tensor_type.shape.dim)

    good = {vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi)}
    good |= {x.name for x in list(m.graph.input) + list(m.graph.output)}
    missing = []
    for nd in m.graph.node:
        for o in nd.output:
            if o and o != "output" and o not in good and o not in missing:
                missing.append(o)
    if not missing:
        return m
    tmp = _copy.deepcopy(m)
    for nm in missing:
        vi = onnx.ValueInfoProto()
        vi.name = nm
        tmp.graph.output.append(vi)
    so = _ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = _ort.InferenceSession(tmp.SerializeToString(), so)
    mx, dt = {}, {}
    for inp in _reps(t):
        for nm, arr in zip(missing, s.run(missing, {"input": inp})):
            sh = list(arr.shape)
            mx[nm] = [max(a, b) for a, b in zip(mx[nm], sh)] if nm in mx else sh
            dt[nm] = arr.dtype
    keep = [vi for vi in m.graph.value_info if vi.name not in missing]
    del m.graph.value_info[:]
    m.graph.value_info.extend(keep)
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("bool"): TensorProto.BOOL, np.dtype("int32"): TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m


def _make():
    inits, nodes = [], []

    def addK(n, a, d):
        inits.append(_K(n, a, d))

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))

    def shift4(src, prefix):
        outs = []
        for k, (di, dj) in enumerate([(-1, 0), (0, -1), (0, 1), (1, 0)]):
            addK(prefix + "_st%d" % k, [1 + di, 1 + dj], np.int64)
            addK(prefix + "_en%d" % k, [31 + di, 31 + dj], np.int64)
            nn("Slice", [src, prefix + "_st%d" % k, prefix + "_en%d" % k, "ax23"], [prefix + "_sl%d" % k])
            outs.append(prefix + "_sl%d" % k)
        return outs

    def guard(name, rname, cname):
        nn("GreaterOrEqual", [rname, "c0i"], [name + "_rge"])
        nn("Less", [rname, "c30i"], [name + "_rlt"])
        nn("And", [name + "_rge", name + "_rlt"], [name + "_rv"])
        nn("GreaterOrEqual", [cname, "c0i"], [name + "_cge"])
        nn("Less", [cname, "c30i"], [name + "_clt"])
        nn("And", [name + "_cge", name + "_clt"], [name + "_cv"])
        nn("And", [name + "_rv", name + "_cv"], [name + "_valid"])
        nn("Mul", [rname, "c30i"], [name + "_r30"])
        nn("Add", [name + "_r30", cname], [name + "_flat0"])
        nn("Where", [name + "_valid", name + "_flat0", "c0i"], [name + "_flat"])
        nn("Gather", ["is5_flat", name + "_flat"], [name + "_g"], axis=0)
        nn("And", [name + "_g", name + "_valid"], [name + "_hit"])
        return name + "_hit"

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    addK("c0i", [0], np.int64)
    addK("c4i", [4], np.int64)
    addK("c5i", [5], np.int64)
    addK("c30i", [30], np.int64)
    addK("c1i", [1], np.int64)
    addK("c2i", [2], np.int64)
    addK("m1i", [-1], np.int64)
    addK("p999i", [999], np.int64)
    addK("c0f", [0.0], np.float32)
    addK("pads4", [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK("sent_i32", [SENT], np.int32)
    addK("ax23", [2, 3], np.int64)
    addK("shape900", [900], np.int64)
    addK("shape900_1", [900, 1], np.int64)
    addK("shape1_900", [1, 900], np.int64)
    addK("shape113030", [1, 1, 30, 30], np.int64)
    addK("depth10", [10], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)
    row = np.repeat(np.arange(30), 30)
    col = np.tile(np.arange(30), 30)
    addK("row_idx", row.reshape(1, 1, 30, 30), np.int64)
    addK("col_idx", col.reshape(1, 1, 30, 30), np.int64)
    addK("row_flat", row, np.int64)
    addK("col_flat", col, np.int64)
    addK("init_label", (row * 30 + col).astype(np.int32).reshape(1, 1, 30, 30), np.int32)

    nn("ArgMax", ["input"], ["idx"], axis=1, keepdims=1)
    nn("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)
    nn("Greater", ["presence", "c0f"], ["present"])
    nn("Equal", ["idx", "c0i"], ["is0"])
    nn("And", ["is0", "present"], ["zero"])
    nn("Equal", ["idx", "c5i"], ["is5"])

    label = "init_label"
    nn("Cast", ["zero"], ["zero_f"], to=F)
    nn("Pad", ["zero_f", "pads4", "c0f"], ["zero_pad_f"], mode="constant")
    nn("Greater", ["zero_pad_f", "c0f"], ["zero_pad"])
    zero_sh = shift4("zero_pad", "z0")
    for it in range(R_ITERS):
        nn("Pad", [label, "pads4", "sent_i32"], ["lab_pad%d" % it], mode="constant")
        lab_sh = shift4("lab_pad%d" % it, "l%d" % it)
        cur = label
        for k in range(4):
            nn("And", ["zero", zero_sh[k]], ["same%d_%d" % (it, k)])
            nn("Where", ["same%d_%d" % (it, k), lab_sh[k], "sent_i32"], ["cand%d_%d" % (it, k)])
            nn("Min", [cur, "cand%d_%d" % (it, k)], ["min%d_%d" % (it, k)])
            cur = "min%d_%d" % (it, k)
        label = cur

    nn("Reshape", [label, "shape900"], ["lab_flat"])
    nn("Reshape", ["lab_flat", "shape900_1"], ["lab_col"])
    nn("Reshape", ["lab_flat", "shape1_900"], ["lab_row"])
    nn("Equal", ["lab_col", "lab_row"], ["same_label0"])
    nn("Reshape", ["zero", "shape900"], ["zero_flat"])
    nn("Reshape", ["zero_flat", "shape900_1"], ["zero_col"])
    nn("Reshape", ["zero_flat", "shape1_900"], ["zero_row"])
    nn("And", ["same_label0", "zero_col"], ["same_label1"])
    nn("And", ["same_label1", "zero_row"], ["same_label"])
    nn("Cast", ["same_label"], ["same_i"], to=I64)
    nn("ReduceSum", ["same_i"], ["comp_size"], axes=[1], keepdims=0)
    nn("Where", ["same_label", "row_flat", "p999i"], ["rows_min_mat"])
    nn("ReduceMin", ["rows_min_mat"], ["rmin"], axes=[1], keepdims=0)
    nn("Where", ["same_label", "row_flat", "m1i"], ["rows_max_mat"])
    nn("ReduceMax", ["rows_max_mat"], ["rmax"], axes=[1], keepdims=0)
    nn("Where", ["same_label", "col_flat", "p999i"], ["cols_min_mat"])
    nn("ReduceMin", ["cols_min_mat"], ["cmin"], axes=[1], keepdims=0)
    nn("Where", ["same_label", "col_flat", "m1i"], ["cols_max_mat"])
    nn("ReduceMax", ["cols_max_mat"], ["cmax"], axes=[1], keepdims=0)
    nn("Sub", ["rmax", "rmin"], ["hh0"])
    nn("Add", ["hh0", "c1i"], ["hh"])
    nn("Sub", ["cmax", "cmin"], ["ww0"])
    nn("Add", ["ww0", "c1i"], ["ww"])
    nn("Mul", ["hh", "ww"], ["area"])
    nn("Equal", ["comp_size", "area"], ["is_rect"])

    nn("Reshape", ["is5", "shape900"], ["is5_flat"])
    hits = []
    for i, (dr, dc) in enumerate([(-2, -1), (-1, -2), (-2, 1), (-1, 2), (2, -1), (1, -2), (2, 1), (1, 2)]):
        base_r = "rmin" if dr < 0 else "rmax"
        base_c = "cmin" if dc < 0 else "cmax"
        addK("dr%d" % i, [dr], np.int64)
        addK("dc%d" % i, [dc], np.int64)
        nn("Add", [base_r, "dr%d" % i], ["gr%d" % i])
        nn("Add", [base_c, "dc%d" % i], ["gc%d" % i])
        hits.append(guard("g%d" % i, "gr%d" % i, "gc%d" % i))
    cur = hits[0]
    for i, h in enumerate(hits[1:], 1):
        nn("Or", [cur, h], ["guard_hit%d" % i])
        cur = "guard_hit%d" % i
    nn("Not", [cur], ["no_guard5"])
    nn("And", ["is_rect", "no_guard5"], ["fill_flat0"])
    nn("And", ["fill_flat0", "zero_flat"], ["fill_flat"])
    nn("Reshape", ["fill_flat", "shape113030"], ["fill"])
    nn("Where", ["fill", "c4i", "idx"], ["out_idx"])
    nn("Squeeze", ["out_idx"], ["out_sq"], axes=[1])
    nn("OneHot", ["out_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)
    nn("Mul", ["oh_raw", "presence"], ["output"])
    graph = helper.make_graph(nodes, "task367", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


model = _bake(_make(), 367)
