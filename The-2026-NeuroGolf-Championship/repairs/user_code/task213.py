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
    colors = [1, 2, 3, 4, 6, 7, 8, 9]

    addK("c0i", [0], np.int64)
    addK("c1i", [1], np.int64)
    addK("p999i", [999], np.int64)
    addK("m1i", [-1], np.int64)
    addK("c0f", [0.0], np.float32)
    addK("depth10", [10], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)
    addK("shape1", [1], np.int64)
    addK("shape113030", [1, 1, 30, 30], np.int64)
    addK("shape3030", [30, 30], np.int64)
    addK("row_grid", np.repeat(np.arange(30), 30).reshape(30, 30), np.int64)
    addK("col_grid", np.tile(np.arange(30), 30).reshape(30, 30), np.int64)
    addK("row4", np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK("col4", np.arange(30).reshape(1, 1, 1, 30), np.int64)
    for c in colors:
        addK("val%d" % c, [c], np.int64)

    nn("ArgMax", ["input"], ["idx"], axis=1, keepdims=1)
    nn("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)

    present_names, rmins, cmins, heights, widths = [], [], [], [], []
    for c in colors:
        nn("Equal", ["idx", "val%d" % c], ["m%d" % c])
        nn("Cast", ["m%d" % c], ["mf%d" % c], to=F)
        nn("ReduceMax", ["mf%d" % c], ["ranyf%d" % c], axes=[3], keepdims=1)
        nn("Greater", ["ranyf%d" % c, "c0f"], ["rany%d" % c])
        nn("ReduceMax", ["mf%d" % c], ["canyf%d" % c], axes=[2], keepdims=1)
        nn("Greater", ["canyf%d" % c, "c0f"], ["cany%d" % c])
        nn("Where", ["rany%d" % c, "row4", "p999i"], ["rvmin%d" % c])
        nn("ReduceMin", ["rvmin%d" % c], ["rmin4_%d" % c], axes=[2, 3], keepdims=0)
        nn("Where", ["rany%d" % c, "row4", "m1i"], ["rvmax%d" % c])
        nn("ReduceMax", ["rvmax%d" % c], ["rmax4_%d" % c], axes=[2, 3], keepdims=0)
        nn("Where", ["cany%d" % c, "col4", "p999i"], ["cvmin%d" % c])
        nn("ReduceMin", ["cvmin%d" % c], ["cmin4_%d" % c], axes=[2, 3], keepdims=0)
        nn("Where", ["cany%d" % c, "col4", "m1i"], ["cvmax%d" % c])
        nn("ReduceMax", ["cvmax%d" % c], ["cmax4_%d" % c], axes=[2, 3], keepdims=0)
        nn("Greater", ["rmax4_%d" % c, "m1i"], ["present%d" % c])
        nn("Sub", ["rmax4_%d" % c, "rmin4_%d" % c], ["h0_%d" % c])
        nn("Add", ["h0_%d" % c, "c1i"], ["h_%d" % c])
        nn("Sub", ["cmax4_%d" % c, "cmin4_%d" % c], ["w0_%d" % c])
        nn("Add", ["w0_%d" % c, "c1i"], ["w_%d" % c])
        present_names.append("present%d" % c)
        rmins.append("rmin4_%d" % c)
        cmins.append("cmin4_%d" % c)
        heights.append("h_%d" % c)
        widths.append("w_%d" % c)

    nn("Concat", present_names, ["present_vec"], axis=0)
    nn("Cast", ["present_vec"], ["present_i"], to=I64)
    nn("ReduceSum", ["present_i"], ["n_colors0"], axes=[0], keepdims=0)
    nn("Reshape", ["n_colors0", "shape1"], ["n_colors"])
    nn("Concat", rmins, ["rmin_vec"], axis=0)
    nn("Concat", cmins, ["cmin_vec"], axis=0)
    nn("Concat", heights, ["height_vec"], axis=0)
    nn("Concat", widths, ["width_vec"], axis=0)
    nn("Where", ["present_vec", "height_vec", "c0i"], ["height_live"])
    nn("Where", ["present_vec", "width_vec", "c0i"], ["width_live"])
    nn("ReduceSum", ["height_live"], ["sum_h"], axes=[0], keepdims=0)
    nn("ReduceSum", ["width_live"], ["sum_w"], axes=[0], keepdims=0)
    nn("Greater", ["sum_h", "sum_w"], ["vertical"])
    nn("Where", ["vertical", "cmin_vec", "rmin_vec"], ["key_vec"])

    # rank[color] = number of present colors with smaller key.
    rank_names = []
    for i, c in enumerate(colors):
        less_parts = []
        for j, d in enumerate(colors):
            nn("Gather", ["key_vec", _const_idx(addK, "ki_%d_%d" % (i, j), i)], ["key_i_%d_%d" % (i, j)], axis=0)
            nn("Gather", ["key_vec", _const_idx(addK, "kj_%d_%d" % (i, j), j)], ["key_j_%d_%d" % (i, j)], axis=0)
            nn("Less", ["key_j_%d_%d" % (i, j), "key_i_%d_%d" % (i, j)], ["lt_%d_%d" % (i, j)])
            nn("Gather", ["present_vec", "kj_%d_%d" % (i, j)], ["pres_j_%d_%d" % (i, j)], axis=0)
            nn("And", ["lt_%d_%d" % (i, j), "pres_j_%d_%d" % (i, j)], ["rankbit_%d_%d" % (i, j)])
            less_parts.append("rankbit_%d_%d" % (i, j))
        nn("Concat", less_parts, ["rankbits_%d" % c], axis=0)
        nn("Cast", ["rankbits_%d" % c], ["rankbits_i_%d" % c], to=I64)
        nn("ReduceSum", ["rankbits_i_%d" % c], ["rank_%d" % c], axes=[0], keepdims=0)
        rank_names.append("rank_%d" % c)

    nn("Reshape", ["row_grid", "shape3030"], ["rr"])
    nn("Reshape", ["col_grid", "shape3030"], ["cc"])
    nn("Less", ["rr", "n_colors"], ["row_in"])
    nn("Less", ["cc", "n_colors"], ["col_in"])
    nn("And", ["row_in", "col_in"], ["inside"])
    nn("Where", ["vertical", "cc", "rr"], ["want_rank"])
    out_cur = "zero_grid"
    addK("zero_grid", np.zeros((30, 30), dtype=np.int64), np.int64)
    for c in colors:
        nn("Equal", ["want_rank", "rank_%d" % c], ["sel_rank_%d" % c])
        nn("Gather", ["present_vec", _const_idx(addK, "pidx_%d" % c, colors.index(c))], ["pres_c_%d" % c], axis=0)
        nn("And", ["sel_rank_%d" % c, "pres_c_%d" % c], ["sel_live_%d" % c])
        nn("And", ["sel_live_%d" % c, "inside"], ["sel_%d" % c])
        nn("Where", ["sel_%d" % c, "val%d" % c, out_cur], ["out_grid_%d" % c])
        out_cur = "out_grid_%d" % c
    nn("Reshape", [out_cur, "shape113030"], ["out_idx"])
    nn("Reshape", ["inside", "shape113030"], ["out_presence"])
    nn("Squeeze", ["out_idx"], ["out_sq"], axes=[1])
    nn("OneHot", ["out_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)
    nn("Cast", ["out_presence"], ["out_presence_f"], to=F)
    nn("Mul", ["oh_raw", "out_presence_f"], ["output"])

    graph = helper.make_graph(nodes, "task213_dsl", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


def _const_idx(addK, name, value):
    addK(name, [value], np.int64)
    return name


model = _bake(_make(), 213)


def save(path=None):
    if path is None:
        path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "task213.onnx"))
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    print(save())
