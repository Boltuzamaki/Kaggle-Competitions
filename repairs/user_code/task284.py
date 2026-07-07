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

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])
    row = np.repeat(np.arange(30), 30).astype(np.int64)
    col = np.tile(np.arange(30), 30).astype(np.int64)

    for c in range(10):
        addK("c%di" % c, [c], np.int64)
    addK("c30i", [30], np.int64)
    addK("c2i_scalar", [2], np.int64)
    addK("m1i", [-1], np.int64)
    addK("p999i", [999], np.int64)
    addK("c0f", [0.0], np.float32)
    addK("row_flat", row, np.int64)
    addK("col_flat", col, np.int64)
    addK("shape900", [900], np.int64)
    addK("shape113030", [1, 1, 30, 30], np.int64)
    addK("depth10", [10], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)

    nn("ArgMax", ["input"], ["idx"], axis=1, keepdims=1)
    nn("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)
    nn("Greater", ["presence", "c0f"], ["present"])
    nn("Reshape", ["idx", "shape900"], ["idx_flat"])
    nn("Reshape", ["present", "shape900"], ["present_flat"])
    nn("Greater", ["idx_flat", "c0i"], ["nonzero0"])
    nn("And", ["nonzero0", "present_flat"], ["nonzero"])
    nn("Where", ["nonzero", "row_flat", "p999i"], ["rmin_vals"])
    nn("ReduceMin", ["rmin_vals"], ["rmin"], axes=[0], keepdims=0)
    nn("Where", ["nonzero", "row_flat", "m1i"], ["rmax_vals"])
    nn("ReduceMax", ["rmax_vals"], ["rmax"], axes=[0], keepdims=0)
    nn("Where", ["nonzero", "col_flat", "p999i"], ["cmin_vals"])
    nn("ReduceMin", ["cmin_vals"], ["cmin"], axes=[0], keepdims=0)
    nn("Where", ["nonzero", "col_flat", "m1i"], ["cmax_vals"])
    nn("ReduceMax", ["cmax_vals"], ["cmax"], axes=[0], keepdims=0)
    nn("Sub", ["rmax", "rmin"], ["bbox_h0"])
    nn("Sub", ["cmax", "cmin"], ["bbox_w0"])
    nn("Greater", ["bbox_h0", "bbox_w0"], ["portrait"])
    nn("Not", ["portrait"], ["trans"])

    nn("Where", ["trans", "col_flat", "row_flat"], ["orow"])
    nn("Where", ["trans", "row_flat", "col_flat"], ["ocol"])
    nn("Where", ["nonzero", "orow", "p999i"], ["omin_vals"])
    nn("ReduceMin", ["omin_vals"], ["top_r"], axes=[0], keepdims=0)
    nn("Where", ["nonzero", "orow", "m1i"], ["omax_vals"])
    nn("ReduceMax", ["omax_vals"], ["bot_r"], axes=[0], keepdims=0)
    nn("Where", ["nonzero", "ocol", "p999i"], ["oc_vals"])
    nn("ReduceMin", ["oc_vals"], ["mid_c"], axes=[0], keepdims=0)
    nn("Add", ["top_r", "bot_r"], ["sum_tb"])
    nn("Div", ["sum_tb", "c2i_scalar"], ["mid_r"])

    nn("Equal", ["orow", "top_r"], ["at_top"])
    nn("And", ["at_top", "nonzero"], ["top_mask"])
    nn("Where", ["top_mask", "idx_flat", "c0i"], ["top_color_vals"])
    nn("ReduceMax", ["top_color_vals"], ["top_color"], axes=[0], keepdims=0)
    nn("Equal", ["orow", "bot_r"], ["at_bot"])
    nn("And", ["at_bot", "nonzero"], ["bot_mask"])
    nn("Where", ["bot_mask", "idx_flat", "c0i"], ["bot_color_vals"])
    nn("ReduceMax", ["bot_color_vals"], ["bot_color"], axes=[0], keepdims=0)

    nn("Sub", ["mid_r", "c1i"], ["top_bar_r"])
    nn("Add", ["mid_r", "c2i"], ["bot_bar_r"])
    nn("Sub", ["mid_c", "c2i"], ["left_c"])
    nn("Add", ["mid_c", "c2i"], ["right_c"])
    nn("LessOrEqual", ["top_r", "orow"], ["ge_top"])
    nn("Less", ["orow", "mid_r"], ["lt_mid"])
    nn("And", ["ge_top", "lt_mid"], ["top_vert_r"])
    nn("Equal", ["ocol", "mid_c"], ["at_mid_c"])
    nn("And", ["top_vert_r", "at_mid_c"], ["top_vert"])
    nn("Equal", ["orow", "top_bar_r"], ["at_top_bar"])
    nn("LessOrEqual", ["left_c", "ocol"], ["ge_left"])
    nn("LessOrEqual", ["ocol", "right_c"], ["le_right"])
    nn("And", ["ge_left", "le_right"], ["in_bar_c"])
    nn("And", ["at_top_bar", "in_bar_c"], ["top_bar"])
    nn("Equal", ["orow", "mid_r"], ["at_mid_r"])
    nn("Equal", ["ocol", "left_c"], ["at_left_c"])
    nn("Equal", ["ocol", "right_c"], ["at_right_c"])
    nn("Or", ["at_left_c", "at_right_c"], ["at_side_c"])
    nn("And", ["at_mid_r", "at_side_c"], ["top_side"])
    nn("Or", ["top_vert", "top_bar"], ["top_a"])
    nn("Or", ["top_a", "top_side"], ["top_draw"])

    nn("Add", ["mid_r", "c3i"], ["bot_vert_start"])
    nn("LessOrEqual", ["bot_vert_start", "orow"], ["ge_bot_start"])
    nn("LessOrEqual", ["orow", "bot_r"], ["le_bot"])
    nn("And", ["ge_bot_start", "le_bot"], ["bot_vert_r"])
    nn("And", ["bot_vert_r", "at_mid_c"], ["bot_vert"])
    nn("Equal", ["orow", "bot_bar_r"], ["at_bot_bar"])
    nn("And", ["at_bot_bar", "in_bar_c"], ["bot_bar"])
    nn("Add", ["mid_r", "c1i"], ["bot_side_r"])
    nn("Equal", ["orow", "bot_side_r"], ["at_bot_side_r"])
    nn("And", ["at_bot_side_r", "at_side_c"], ["bot_side"])
    nn("Or", ["bot_vert", "bot_bar"], ["bot_a"])
    nn("Or", ["bot_a", "bot_side"], ["bot_draw"])

    nn("Where", ["bot_draw", "bot_color", "c0i"], ["out_bot"])
    nn("Where", ["top_draw", "top_color", "out_bot"], ["out_flat"])
    nn("Reshape", ["out_flat", "shape113030"], ["out_idx"])
    nn("Squeeze", ["out_idx"], ["out_sq"], axes=[1])
    nn("OneHot", ["out_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)
    nn("Mul", ["oh_raw", "presence"], ["output"])

    graph = helper.make_graph(nodes, "task284_dsl", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


model = _bake(_make(), 284)


def save(path=None):
    if path is None:
        path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "task284.onnx"))
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    print(save())
