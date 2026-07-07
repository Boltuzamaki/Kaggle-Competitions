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
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("bool"): TensorProto.BOOL}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m


def _make():
    inits, nodes = [], []

    def addK(name, arr, dtype):
        inits.append(_K(name, arr, dtype))

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0]

    x = helper.make_tensor_value_info("input", F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", F, [1, 10, 30, 30])

    addK("c0i", [0], np.int64)
    addK("c1i", [1], np.int64)
    addK("c2i", [2], np.int64)
    addK("c5i", [5], np.int64)
    addK("c29i", [29], np.int64)
    addK("m1i", [-1], np.int64)
    addK("p999i", [999], np.int64)
    addK("c0f", [0.0], np.float32)
    addK("shape1x1x30x30", [1, 1, 30, 30], np.int64)
    addK("row4", np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK("col4", np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK("row2", np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK("col2", np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK("zero_grid", np.zeros((1, 1, 30, 30), dtype=np.int64), np.int64)
    addK("ax1", [1], np.int64)
    addK("s5", [5], np.int64)
    addK("e6", [6], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)
    addK("depth10", [10], np.int64)

    nn("ArgMax", ["input"], ["idx"], axis=1, keepdims=1)
    nn("Equal", ["idx", "c2i"], ["is2"])
    nn("Cast", ["is2"], ["m2"], to=F)
    nn("Equal", ["idx", "c5i"], ["is5"])

    nn("ReduceMax", ["m2"], ["row_any"], axes=[3], keepdims=1)
    nn("Greater", ["row_any", "c0f"], ["row_b"])
    nn("Where", ["row_b", "row4", "p999i"], ["r_pmin"])
    nn("ReduceMin", ["r_pmin"], ["rmin"], axes=[2], keepdims=1)
    nn("Where", ["row_b", "row4", "m1i"], ["r_pmax"])
    nn("ReduceMax", ["r_pmax"], ["rmax"], axes=[2], keepdims=1)
    nn("ReduceMax", ["m2"], ["col_any"], axes=[2], keepdims=1)
    nn("Greater", ["col_any", "c0f"], ["col_b"])
    nn("Where", ["col_b", "col4", "p999i"], ["c_pmin"])
    nn("ReduceMin", ["c_pmin"], ["cmin"], axes=[3], keepdims=1)
    nn("Where", ["col_b", "col4", "m1i"], ["c_pmax"])
    nn("ReduceMax", ["c_pmax"], ["cmax"], axes=[3], keepdims=1)

    nn("Sub", ["rmax", "rmin"], ["hm1"])
    nn("Add", ["hm1", "c1i"], ["H"])
    nn("Sub", ["cmax", "cmin"], ["wm1"])
    nn("Add", ["wm1", "c1i"], ["W"])
    nn("Sub", ["H", "c2i"], ["ih"])
    nn("Sub", ["W", "c2i"], ["iw"])
    nn("Div", ["ih", "c2i"], ["hh"])
    nn("Div", ["iw", "c2i"], ["hw"])
    nn("Mod", ["H", "c2i"], ["hmod"])
    nn("Mod", ["W", "c2i"], ["wmod"])
    nn("Add", ["hmod", "c1i"], ["pe_v"])
    nn("Add", ["wmod", "c1i"], ["pe_h"])
    nn("GreaterOrEqual", ["W", "H"], ["landscape"])

    nn("Slice", ["input", "s5", "e6", "ax1"], ["ch5"])

    def gather_mask(prefix, rr, cc, valid):
        nn("Max", [rr, "c0i"], [prefix + "_r0"])
        nn("Min", [prefix + "_r0", "c29i"], [prefix + "_rs"])
        nn("Max", [cc, "c0i"], [prefix + "_c0"])
        nn("Min", [prefix + "_c0", "c29i"], [prefix + "_cs"])
        nn("Expand", [prefix + "_rs", "shape1x1x30x30"], [prefix + "_rse"])
        nn("Expand", [prefix + "_cs", "shape1x1x30x30"], [prefix + "_cse"])
        nn("Unsqueeze", ["zero_grid"], [prefix + "_b"], axes=[4])
        nn("Unsqueeze", ["zero_grid"], [prefix + "_ch"], axes=[4])
        nn("Unsqueeze", [prefix + "_rse"], [prefix + "_ru"], axes=[4])
        nn("Unsqueeze", [prefix + "_cse"], [prefix + "_cu"], axes=[4])
        nn("Concat", [prefix + "_b", prefix + "_ch", prefix + "_ru", prefix + "_cu"], [prefix + "_ix"], axis=4)
        nn("GatherND", ["ch5", prefix + "_ix"], [prefix + "_g"])
        nn("Greater", [prefix + "_g", "c0f"], [prefix + "_gb"])
        nn("And", [valid, prefix + "_gb"], [prefix + "_hit"])
        return prefix + "_hit"

    nn("Add", ["cmin", "c1i"], ["c_anchor"])
    nn("Add", ["rmin", "c1i"], ["r_anchor"])

    # W >= H: vertical mirror, split into left/right halves, paint them left/right of the frame.
    nn("Add", ["hw", "wmod"], ["h_start"])
    nn("Add", ["hw", "c1i"], ["h_neg_dist"])
    nn("Sub", ["c_anchor", "h_neg_dist"], ["h_neg_zero"])
    nn("Sub", ["col2", "h_neg_zero"], ["h_k"])
    nn("GreaterOrEqual", ["h_k", "c0i"], ["h_k_ge"])
    nn("Less", ["h_k", "hw"], ["h_k_lt"])
    nn("And", ["h_k_ge", "h_k_lt"], ["h_neg_col_ok"])
    nn("Sub", ["row2", "r_anchor"], ["h_ir"])
    nn("GreaterOrEqual", ["h_ir", "c0i"], ["h_ir_ge"])
    nn("Less", ["h_ir", "ih"], ["h_ir_lt"])
    nn("And", ["h_ir_ge", "h_ir_lt"], ["h_row_ok"])
    nn("And", ["h_neg_col_ok", "h_row_ok"], ["h_neg_ok"])
    nn("Add", ["cmin", "iw"], ["h_src_base"])
    nn("Sub", ["h_src_base", "h_start"], ["h_src_neg_base"])
    nn("Sub", ["h_src_neg_base", "h_k"], ["h_src_c_neg"])
    nn("Add", ["r_anchor", "h_ir"], ["h_src_r"])
    gather_mask("hneg", "h_src_r", "h_src_c_neg", "h_neg_ok")

    nn("Mul", ["hw", "c2i"], ["h_2hw"])
    nn("Add", ["h_2hw", "pe_h"], ["h_pos_dist"])
    nn("Add", ["c_anchor", "h_pos_dist"], ["h_pos_zero"])
    nn("Sub", ["col2", "h_pos_zero"], ["h_j"])
    nn("GreaterOrEqual", ["h_j", "c0i"], ["h_j_ge"])
    nn("Less", ["h_j", "hw"], ["h_j_lt"])
    nn("And", ["h_j_ge", "h_j_lt"], ["h_pos_col_ok"])
    nn("And", ["h_pos_col_ok", "h_row_ok"], ["h_pos_ok"])
    nn("Sub", ["h_src_base", "h_j"], ["h_src_c_pos"])
    gather_mask("hpos", "h_src_r", "h_src_c_pos", "h_pos_ok")
    nn("Or", ["hneg_hit", "hpos_hit"], ["hit_h_2d"])

    # H > W: horizontal mirror, split into top/bottom halves, paint them above/below the frame.
    nn("Add", ["hh", "hmod"], ["v_start"])
    nn("Add", ["hh", "c1i"], ["v_neg_dist"])
    nn("Sub", ["r_anchor", "v_neg_dist"], ["v_neg_zero"])
    nn("Sub", ["row2", "v_neg_zero"], ["v_k"])
    nn("GreaterOrEqual", ["v_k", "c0i"], ["v_k_ge"])
    nn("Less", ["v_k", "hh"], ["v_k_lt"])
    nn("And", ["v_k_ge", "v_k_lt"], ["v_neg_row_ok"])
    nn("Sub", ["col2", "c_anchor"], ["v_ic"])
    nn("GreaterOrEqual", ["v_ic", "c0i"], ["v_ic_ge"])
    nn("Less", ["v_ic", "iw"], ["v_ic_lt"])
    nn("And", ["v_ic_ge", "v_ic_lt"], ["v_col_ok"])
    nn("And", ["v_neg_row_ok", "v_col_ok"], ["v_neg_ok"])
    nn("Add", ["rmin", "ih"], ["v_src_base"])
    nn("Sub", ["v_src_base", "v_start"], ["v_src_neg_base"])
    nn("Sub", ["v_src_neg_base", "v_k"], ["v_src_r_neg"])
    nn("Add", ["c_anchor", "v_ic"], ["v_src_c"])
    gather_mask("vneg", "v_src_r_neg", "v_src_c", "v_neg_ok")

    nn("Mul", ["hh", "c2i"], ["v_2hh"])
    nn("Add", ["v_2hh", "pe_v"], ["v_pos_dist"])
    nn("Add", ["r_anchor", "v_pos_dist"], ["v_pos_zero"])
    nn("Sub", ["row2", "v_pos_zero"], ["v_j"])
    nn("GreaterOrEqual", ["v_j", "c0i"], ["v_j_ge"])
    nn("Less", ["v_j", "hh"], ["v_j_lt"])
    nn("And", ["v_j_ge", "v_j_lt"], ["v_pos_row_ok"])
    nn("And", ["v_pos_row_ok", "v_col_ok"], ["v_pos_ok"])
    nn("Sub", ["v_src_base", "v_j"], ["v_src_r_pos"])
    gather_mask("vpos", "v_src_r_pos", "v_src_c", "v_pos_ok")
    nn("Or", ["vneg_hit", "vpos_hit"], ["hit_v_2d"])

    nn("And", ["landscape", "hit_h_2d"], ["hit_h_sel"])
    nn("Not", ["landscape"], ["portrait"])
    nn("And", ["portrait", "hit_v_2d"], ["hit_v_sel"])
    nn("Or", ["hit_h_sel", "hit_v_sel"], ["hit"])
    nn("Where", ["is5", "c0i", "idx"], ["cleared"])
    nn("Where", ["hit", "c5i", "cleared"], ["out_idx"])
    nn("Squeeze", ["out_idx"], ["out_sq"], axes=[1])
    nn("OneHot", ["out_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)
    nn("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)
    nn("Mul", ["oh_raw", "presence"], ["output"])

    graph = helper.make_graph(nodes, "task390", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


model = _bake(_make(), 390)
