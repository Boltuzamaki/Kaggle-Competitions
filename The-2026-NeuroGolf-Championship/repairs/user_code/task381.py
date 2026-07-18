import json
import os as _os
import copy as _copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F = TensorProto.FLOAT
I64 = TensorProto.INT64


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
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("bool"): TensorProto.BOOL}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m


def _make():
    inits, nodes = [], []

    def addK(n, a, d):
        inits.append(_K(n, a, d))

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    addK("c0i", [0], np.int64)
    addK("c2i", [2], np.int64)
    addK("c9i", [9], np.int64)
    addK("c1i", [1], np.int64)
    addK("c9lim", [9], np.int64)
    addK("p999", [999], np.int64)
    addK("m1", [-1], np.int64)
    addK("c0f", [0.0], np.float32)
    addK("col_idx", np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK("row_idx", np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK("depth10", [10], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)

    nn("ArgMax", ["input"], ["idx"], axis=1, keepdims=1)
    nn("Equal", ["idx", "c2i"], ["is2"])
    nn("Equal", ["idx", "c0i"], ["is0"])
    nn("Cast", ["is2"], ["m2"], to=F)
    nn("ReduceMax", ["m2"], ["row_has2_f"], axes=[3], keepdims=1)
    nn("Greater", ["row_has2_f", "c0f"], ["row_has2"])
    nn("Where", ["is2", "col_idx", "p999"], ["cols_min"])
    nn("ReduceMin", ["cols_min"], ["cmin"], axes=[3], keepdims=1)
    nn("Where", ["is2", "col_idx", "m1"], ["cols_max"])
    nn("ReduceMax", ["cols_max"], ["cmax"], axes=[3], keepdims=1)
    nn("Greater", ["col_idx", "cmin"], ["gt_min"])
    nn("Less", ["col_idx", "cmax"], ["lt_max"])
    nn("And", ["gt_min", "lt_max"], ["between"])
    nn("GreaterOrEqual", ["row_idx", "c1i"], ["r_ge1"])
    nn("Less", ["row_idx", "c9lim"], ["r_lt9"])
    nn("GreaterOrEqual", ["col_idx", "c1i"], ["c_ge1"])
    nn("Less", ["col_idx", "c9lim"], ["c_lt9"])
    nn("And", ["r_ge1", "r_lt9"], ["inner_r"])
    nn("And", ["c_ge1", "c_lt9"], ["inner_c"])
    nn("And", ["inner_r", "inner_c"], ["inner"])
    nn("And", ["between", "row_has2"], ["bridge1"])
    nn("And", ["bridge1", "inner"], ["bridge2"])
    nn("And", ["bridge2", "is0"], ["paint9"])
    nn("Where", ["paint9", "c9i", "idx"], ["out_idx"])
    nn("Squeeze", ["out_idx"], ["out_sq"], axes=[1])
    nn("OneHot", ["out_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)
    nn("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)
    nn("Mul", ["oh_raw", "presence"], ["output"])
    graph = helper.make_graph(nodes, "task381", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


model = _bake(_make(), 381)
