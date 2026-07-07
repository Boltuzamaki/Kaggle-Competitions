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
    init_label = (row * 30 + col).astype(np.int32).reshape(1, 1, 30, 30)

    for c in [0, 1, 2, 3, 6]:
        addK("c%di" % c, [c], np.int64)
    addK("c0f", [0.0], np.float32)
    addK("m1i", [-1], np.int64)
    addK("p999i", [999], np.int64)
    addK("sent_i32", [100000], np.int32)
    addK("pads4", [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK("ax23", [2, 3], np.int64)
    addK("row_flat", row, np.int64)
    addK("col_flat", col, np.int64)
    addK("init_label", init_label, np.int32)
    addK("shape900", [900], np.int64)
    addK("shape900_1", [900, 1], np.int64)
    addK("shape1_900", [1, 900], np.int64)
    addK("shape113030", [1, 1, 30, 30], np.int64)
    addK("depth10", [10], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)

    def shift4(src, prefix):
        outs = []
        for k, (di, dj) in enumerate([(-1, 0), (0, -1), (0, 1), (1, 0)]):
            addK(prefix + "_st%d" % k, [1 + di, 1 + dj], np.int64)
            addK(prefix + "_en%d" % k, [31 + di, 31 + dj], np.int64)
            nn("Slice", [src, prefix + "_st%d" % k, prefix + "_en%d" % k, "ax23"], [prefix + "_sl%d" % k])
            outs.append(prefix + "_sl%d" % k)
        return outs

    nn("ArgMax", ["input"], ["idx"], axis=1, keepdims=1)
    nn("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)
    nn("Reshape", ["idx", "shape900"], ["idx_flat"])
    nn("Equal", ["idx", "c3i"], ["is3"])
    nn("Cast", ["is3"], ["is3_f"], to=F)
    nn("Pad", ["is3_f", "pads4", "c0f"], ["is3_pad_f"], mode="constant")
    nn("Greater", ["is3_pad_f", "c0f"], ["is3_pad"])
    is3_sh = shift4("is3_pad", "z")
    label = "init_label"
    for it in range(60):
        nn("Pad", [label, "pads4", "sent_i32"], ["lab_pad%d" % it], mode="constant")
        lab_sh = shift4("lab_pad%d" % it, "l%d" % it)
        cur = label
        for k in range(4):
            nn("And", ["is3", is3_sh[k]], ["same%d_%d" % (it, k)])
            nn("Where", ["same%d_%d" % (it, k), lab_sh[k], "sent_i32"], ["cand%d_%d" % (it, k)])
            nn("Min", [cur, "cand%d_%d" % (it, k)], ["min%d_%d" % (it, k)])
            cur = "min%d_%d" % (it, k)
        label = cur

    nn("Reshape", [label, "shape900"], ["lab_flat"])
    nn("Reshape", ["lab_flat", "shape900_1"], ["lab_col"])
    nn("Reshape", ["lab_flat", "shape1_900"], ["lab_row"])
    nn("Equal", ["lab_col", "lab_row"], ["same_label0"])
    nn("Reshape", ["is3", "shape900"], ["is3_flat"])
    nn("Reshape", ["is3_flat", "shape900_1"], ["is3_col"])
    nn("Reshape", ["is3_flat", "shape1_900"], ["is3_row"])
    nn("And", ["same_label0", "is3_col"], ["same_label1"])
    nn("And", ["same_label1", "is3_row"], ["same_label"])
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
    nn("Add", ["hh", "ww"], ["hplusw"])
    nn("Sub", ["hplusw", "c1i"], ["line_size"])
    nn("Equal", ["comp_size", "line_size"], ["is_line"])
    nn("Greater", ["row_flat", "rmin"], ["gt_rmin"])
    nn("Less", ["row_flat", "rmax"], ["lt_rmax"])
    nn("And", ["gt_rmin", "lt_rmax"], ["inside_r"])
    nn("Greater", ["col_flat", "cmin"], ["gt_cmin"])
    nn("Less", ["col_flat", "cmax"], ["lt_cmax"])
    nn("And", ["gt_cmin", "lt_cmax"], ["inside_c"])
    nn("And", ["inside_r", "inside_c"], ["inside_cell"])
    nn("And", ["same_label", "inside_cell"], ["inside_mat"])
    nn("Cast", ["inside_mat"], ["inside_i"], to=I64)
    nn("ReduceMax", ["inside_i"], ["has_inside_i"], axes=[1], keepdims=0)
    nn("Greater", ["has_inside_i", "c0i"], ["has_inside"])
    nn("Where", ["is_line", "c1i", "c6i"], ["class0"])
    nn("Where", ["has_inside", "c2i", "class0"], ["class1"])
    nn("Where", ["is_line", "c1i", "class1"], ["class_final"])
    nn("Where", ["is3_flat", "class_final", "idx_flat"], ["out_flat"])
    nn("Reshape", ["out_flat", "shape113030"], ["out_idx"])
    nn("Squeeze", ["out_idx"], ["out_sq"], axes=[1])
    nn("OneHot", ["out_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)
    nn("Mul", ["oh_raw", "presence"], ["output"])

    graph = helper.make_graph(nodes, "task364_dsl", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


model = _bake(_make(), 364)


def save(path=None):
    if path is None:
        path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "task364.onnx"))
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    print(save())
