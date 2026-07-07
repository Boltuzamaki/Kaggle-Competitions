import json
import os as _os
import copy as _copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

F = TensorProto.FLOAT
I64 = TensorProto.INT64
I32 = TensorProto.INT32


def _K(n, a, d):
    return numpy_helper.from_array(np.array(a, dtype=d), name=n)


def _set_out_shape(m, dims):
    tt = m.graph.output[0].type.tensor_type
    tt.elem_type = TensorProto.FLOAT
    del tt.shape.dim[:]
    for d in dims:
        tt.shape.dim.add().dim_value = d


def _resolve_task_json(t):
    for base in [
        _os.environ.get("PROJECT_DIR", "/project"),
        r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship",
        ".",
    ]:
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
    conv = {
        np.dtype("float32"): TensorProto.FLOAT,
        np.dtype("int64"): TensorProto.INT64,
        np.dtype("bool"): TensorProto.BOOL,
        np.dtype("int32"): TensorProto.INT32,
    }
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m


R_ITERS = 30
SENT = 100000


def _make():
    inits = []
    nodes = []

    def addK(name, arr, dtype):
        inits.append(_K(name, arr, dtype))
        return name

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    addK("c0i", [0], np.int64)
    addK("c1i", [1], np.int64)
    addK("c10i", [10], np.int64)
    addK("c29i", [29], np.int64)
    addK("c30x2", [30, 30], np.int64)
    addK("m1i", [-1], np.int64)
    addK("p999i", [999], np.int64)
    addK("bigf", [10000.0], np.float32)
    addK("negf", [-1.0], np.float32)
    addK("c0f", [0.0], np.float32)
    addK("pv", [0.0], np.float32)
    addK("shape1", [-1], np.int64)
    addK("shape11", [1, 1], np.int64)
    addK("shape1d900", [900], np.int64)
    addK("shape9001", [900, 1], np.int64)
    addK("shape1900", [1, 900], np.int64)
    addK("row_idx", np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK("col_idx", np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK("flat_idx", np.arange(900, dtype=np.int32).reshape(1, 1, 30, 30), np.int32)
    addK("sent_i32", [SENT], np.int32)
    addK("pads_hw", [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK("ax1", [1], np.int64)
    addK("ax2", [2], np.int64)
    addK("ax3", [3], np.int64)
    addK("s1", [1], np.int64)
    addK("s2", [2], np.int64)
    addK("e4", [4], np.int64)
    addK("pfx6", [0, 0, 0, 0, 0, 0], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)
    addK("depth10", [10], np.int64)
    addK("colors19", np.arange(1, 10, dtype=np.int64), np.int64)
    addK("colors19f", np.arange(1, 10, dtype=np.float32), np.float32)
    addK("axes_hw", [2, 3], np.int64)
    addK("axes0", [0], np.int64)
    addK("axes1", [1], np.int64)

    # leastcolor(I), restricted to nonzero colors; zero is background and never least in this task.
    nn("Slice", ["input", "s1", "c10i", "ax1"], ["nonzero_oh"])
    nn("ReduceSum", ["nonzero_oh"], ["nz_counts"], axes=[0, 2, 3], keepdims=0)
    nn("Greater", ["nz_counts", "c0f"], ["present19"])
    nn("Where", ["present19", "nz_counts", "bigf"], ["adj_counts"])
    nn("ArgMin", ["adj_counts"], ["least_idx0"], axis=0, keepdims=0)
    nn("Add", ["least_idx0", "c1i"], ["least_color"])

    nn("ArgMax", ["input"], ["color_i64"], axis=1, keepdims=1)
    nn("Cast", ["color_i64"], ["color_i32"], to=I32)
    nn("Reshape", ["least_color", "shape11"], ["least11"])
    nn("Equal", ["color_i64", "least11"], ["is_least"])
    nn("Equal", ["color_i64", "c0i"], ["is_zero"])
    nn("Or", ["is_least", "is_zero"], ["bad_fg"])
    nn("Not", ["bad_fg"], ["valid_bool"])

    # Loop-free 4-connected same-color min-label propagation.
    nn("Pad", ["color_i32", "pads_hw", "sent_i32"], ["pad_color"], mode="constant")
    dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    same_dirs = []
    for k, (di, dj) in enumerate(dirs):
        addK(f"cs{k}", [1 + di, 1 + dj], np.int64)
        addK(f"ce{k}", [31 + di, 31 + dj], np.int64)
        nn("Slice", ["pad_color", f"cs{k}", f"ce{k}", "axes_hw"], [f"sh_color{k}"])
        nn("Equal", [f"sh_color{k}", "color_i32"], [f"same{k}"])
        same_dirs.append(f"same{k}")

    label = "flat_idx"
    for it in range(R_ITERS):
        nn("Pad", [label, "pads_hw", "sent_i32"], [f"pad_lab{it}"], mode="constant")
        running = label
        for k, (di, dj) in enumerate(dirs):
            addK(f"ls{it}_{k}", [1 + di, 1 + dj], np.int64)
            addK(f"le{it}_{k}", [31 + di, 31 + dj], np.int64)
            nn("Slice", [f"pad_lab{it}", f"ls{it}_{k}", f"le{it}_{k}", "axes_hw"], [f"sh_lab{it}_{k}"])
            nn("Where", [same_dirs[k], f"sh_lab{it}_{k}", "sent_i32"], [f"cand{it}_{k}"])
            nn("Min", [running, f"cand{it}_{k}"], [f"min{it}_{k}"])
            running = f"min{it}_{k}"
        label = running

    nn("Reshape", [label, "shape1d900"], ["lab_flat"])
    nn("Reshape", ["valid_bool", "shape1d900"], ["valid_flat_b"])
    nn("Reshape", ["lab_flat", "shape9001"], ["lab_col"])
    nn("Reshape", ["lab_flat", "shape1900"], ["lab_row"])
    nn("Equal", ["lab_col", "lab_row"], ["same_label"])
    nn("Reshape", ["valid_flat_b", "shape1900"], ["valid_row"])
    nn("And", ["same_label", "valid_row"], ["same_valid"])
    nn("Cast", ["same_valid"], ["same_valid_f"], to=F)
    nn("ReduceSum", ["same_valid_f"], ["comp_sizes"], axes=[1], keepdims=0)
    nn("Where", ["valid_flat_b", "comp_sizes", "negf"], ["scores"])
    nn("ArgMax", ["scores"], ["best_flat"], axis=0, keepdims=0)
    nn("Gather", ["lab_flat", "best_flat"], ["best_label"], axis=0)
    nn("Equal", [label, "best_label"], ["sel_label"])
    nn("And", ["sel_label", "valid_bool"], ["sel_bool"])
    nn("Cast", ["sel_bool"], ["sel_f"], to=F)

    # Bounding box of largest remaining object.
    nn("ReduceMax", ["sel_f"], ["row_any"], axes=[3], keepdims=1)
    nn("Greater", ["row_any", "c0f"], ["row_b"])
    nn("Where", ["row_b", "row_idx", "p999i"], ["r_pmin"])
    nn("ReduceMin", ["r_pmin"], ["rmin"], axes=[2], keepdims=1)
    nn("Where", ["row_b", "row_idx", "m1i"], ["r_pmax"])
    nn("ReduceMax", ["r_pmax"], ["rmax"], axes=[2], keepdims=1)
    nn("ReduceMax", ["sel_f"], ["col_any"], axes=[2], keepdims=1)
    nn("Greater", ["col_any", "c0f"], ["col_b"])
    nn("Where", ["col_b", "col_idx", "p999i"], ["c_pmin"])
    nn("ReduceMin", ["c_pmin"], ["cmin"], axes=[3], keepdims=1)
    nn("Where", ["col_b", "col_idx", "m1i"], ["c_pmax"])
    nn("ReduceMax", ["c_pmax"], ["cmax"], axes=[3], keepdims=1)

    nn("Reshape", ["rmin", "shape1"], ["r0"])
    nn("Add", ["rmax", "c1i"], ["rmax1"])
    nn("Reshape", ["rmax1", "shape1"], ["r1"])
    nn("Reshape", ["cmin", "shape1"], ["c0"])
    nn("Add", ["cmax", "c1i"], ["cmax1"])
    nn("Reshape", ["cmax1", "shape1"], ["c1"])
    nn("Slice", ["input", "r0", "r1", "ax2"], ["crop_y"])
    nn("Slice", ["crop_y", "c0", "c1", "ax3"], ["crop"])

    # replace(x7, frame_color, least_color). The frame color is the selected object's color.
    nn("Reshape", ["color_i64", "shape1d900"], ["color_flat"])
    nn("Gather", ["color_flat", "best_flat"], ["frame_color"], axis=0)
    nn("Reshape", ["frame_color", "shape11"], ["frame11"])
    nn("ArgMax", ["crop"], ["crop_idx"], axis=1, keepdims=1)
    nn("Equal", ["crop_idx", "frame11"], ["is_frame_crop"])
    nn("Where", ["is_frame_crop", "least11", "crop_idx"], ["out_idx"])
    nn("Squeeze", ["out_idx"], ["out_idx_sq"], axes=[1])
    nn("OneHot", ["out_idx_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)

    # Pad the dynamic crop back to the scorer's static [1,10,30,30] tensor.
    nn("Shape", ["oh_raw"], ["osh"])
    nn("Slice", ["osh", "s2", "e4", "axes0"], ["hw"])
    nn("Sub", ["c30x2", "hw"], ["padhw"])
    nn("Concat", ["pfx6", "padhw"], ["pads"], axis=0)
    nn("Pad", ["oh_raw", "pads", "pv"], ["output"], mode="constant")

    graph = helper.make_graph(nodes, "task396", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


model = _bake(_make(), 396)
