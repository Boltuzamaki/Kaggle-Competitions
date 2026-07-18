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
    exs = [e for e in d["train"] + d["test"] + d["arc-gen"] if max(len(e["input"]), len(e["input"][0])) <= 30]
    exs = sorted(exs, key=lambda e: (len(e["input"]), len(e["input"][0])))
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

    cand_s, cand_r, cand_c = [], [], []
    for s in [2, 3, 4]:
        for r in range(30):
            for c in range(30):
                cand_s.append(s)
                cand_r.append(r)
                cand_c.append(c)
    cand_s = np.array(cand_s, dtype=np.int64)
    cand_r = np.array(cand_r, dtype=np.int64)
    cand_c = np.array(cand_c, dtype=np.int64)
    n_cand = len(cand_s)

    obj_pos = np.zeros((n_cand, 16), dtype=np.int64)
    obj_valid = np.zeros((n_cand, 16), dtype=np.bool_)
    bnd_pos = np.zeros((n_cand, 32), dtype=np.int64)
    bnd_valid = np.zeros((n_cand, 32), dtype=np.bool_)
    crop_pos = np.zeros((n_cand, 196), dtype=np.int64)
    crop_valid = np.zeros((n_cand, 196), dtype=np.bool_)
    crop_need = np.zeros(n_cand, dtype=np.int64)
    for i, (s, r, c) in enumerate(zip(cand_s, cand_r, cand_c)):
        k = 0
        for dr in range(4):
            for dc in range(4):
                rr, cc = r + dr, c + dc
                if dr < s and dc < s and 0 <= rr < 30 and 0 <= cc < 30:
                    obj_pos[i, k] = rr * 30 + cc
                    obj_valid[i, k] = True
                k += 1
        b = []
        for dc in range(s):
            b.append((r - 1, c + dc))
            b.append((r + s, c + dc))
        for dr in range(s):
            b.append((r + dr, c - 1))
            b.append((r + dr, c + s))
        for k, (rr, cc) in enumerate(b):
            if 0 <= rr < 30 and 0 <= cc < 30:
                bnd_pos[i, k] = rr * 30 + cc
                bnd_valid[i, k] = True
        h = 3 * s + 2
        st_r, st_c = r - s - 1, c - s - 1
        crop_need[i] = h * h
        k = 0
        for dr in range(14):
            for dc in range(14):
                rr, cc = st_r + dr, st_c + dc
                if dr < h and dc < h and 0 <= rr < 30 and 0 <= cc < 30:
                    crop_pos[i, k] = rr * 30 + cc
                    crop_valid[i, k] = True
                k += 1

    row = np.repeat(np.arange(30), 30).astype(np.int64)
    col = np.tile(np.arange(30), 30).astype(np.int64)
    row4 = np.arange(30, dtype=np.int64).reshape(1, 1, 30, 1)

    for c in range(10):
        addK("c%di" % c, [c], np.int64)
    addK("c30i", [30], np.int64)
    addK("c100i", [100], np.int64)
    addK("c899i", [899], np.int64)
    addK("c999999i", [999999], np.int64)
    addK("m1i", [-1], np.int64)
    addK("p999i", [999], np.int64)
    addK("m999999i", [-999999], np.int64)
    addK("c0f", [0.0], np.float32)
    addK("ax1", [1], np.int64)
    addK("shape900", [900], np.int64)
    addK("shape1", [1], np.int64)
    addK("shape1_1_30_30", [1, 1, 30, 30], np.int64)
    addK("shape1_2700", [1, n_cand], np.int64)
    addK("shape900_1", [900, 1], np.int64)
    addK("depth10", [10], np.int64)
    addK("oh_vals", [0.0, 1.0], np.float32)
    addK("row_flat", row, np.int64)
    addK("row4", row4, np.int64)
    addK("col_flat", col, np.int64)
    addK("out_r_col", row.reshape(900, 1), np.int64)
    addK("out_c_col", col.reshape(900, 1), np.int64)
    addK("cand_s", cand_s, np.int64)
    addK("cand_r", cand_r, np.int64)
    addK("cand_c", cand_c, np.int64)
    addK("cand_s_row", cand_s.reshape(1, n_cand), np.int64)
    addK("cand_r_row", cand_r.reshape(1, n_cand), np.int64)
    addK("cand_c_row", cand_c.reshape(1, n_cand), np.int64)
    addK("obj_pos", obj_pos, np.int64)
    addK("obj_valid", obj_valid, np.bool_)
    addK("bnd_pos", bnd_pos, np.int64)
    addK("bnd_valid", bnd_valid, np.bool_)
    addK("crop_pos", crop_pos, np.int64)
    addK("crop_valid", crop_valid, np.bool_)
    addK("crop_need", crop_need, np.int64)

    nn("ArgMax", ["input"], ["idx"], axis=1, keepdims=1)
    nn("ReduceMax", ["input"], ["presence"], axes=[1], keepdims=1)
    nn("Reshape", ["idx", "shape900"], ["idx_flat"])
    nn("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=0)
    nn("ArgMax", ["counts"], ["bg"], axis=1, keepdims=0)

    color_scores = []
    for c in range(1, 10):
        addK("colidx%d" % c, [c], np.int64)
        nn("Gather", ["counts", "colidx%d" % c], ["cnt%d_2d" % c], axis=1)
        nn("Reshape", ["cnt%d_2d" % c, "shape1"], ["cnt%d" % c])
        nn("Cast", ["cnt%d" % c], ["cnti%d" % c], to=I64)
        nn("Equal", ["idx", "c%di" % c], ["mask4_%d" % c])
        nn("Cast", ["mask4_%d" % c], ["maskf_%d" % c], to=F)
        nn("ReduceMax", ["maskf_%d" % c], ["row_anyf_%d" % c], axes=[3], keepdims=1)
        nn("Greater", ["row_anyf_%d" % c, "c0f"], ["row_any_%d" % c])
        nn("Where", ["row_any_%d" % c, "row4", "m1i"], ["rmax_vals_%d" % c])
        nn("ReduceMax", ["rmax_vals_%d" % c], ["rmax_%d" % c], axes=[0, 1, 2, 3], keepdims=0)
        nn("Where", ["row_any_%d" % c, "row4", "p999i"], ["rmin_vals_%d" % c])
        nn("ReduceMin", ["rmin_vals_%d" % c], ["rmin_%d" % c], axes=[0, 1, 2, 3], keepdims=0)
        nn("Sub", ["rmax_%d" % c, "rmin_%d" % c], ["height0_%d" % c])
        nn("Add", ["height0_%d" % c, "c1i"], ["height_%d" % c])
        nn("Mul", ["cnti%d" % c, "c100i"], ["score0_%d" % c])
        nn("Sub", ["score0_%d" % c, "height_%d" % c], ["score1_%d" % c])
        nn("Equal", ["cnti%d" % c, "c0i"], ["absent_%d" % c])
        nn("Equal", ["bg", "c%di" % c], ["is_bg_%d" % c])
        nn("Or", ["absent_%d" % c, "is_bg_%d" % c], ["bad_color_%d" % c])
        nn("Where", ["bad_color_%d" % c, "c999999i", "score1_%d" % c], ["score_%d" % c])
        color_scores.append("score_%d" % c)
    nn("Concat", color_scores, ["color_score_vec"], axis=0)
    nn("ArgMin", ["color_score_vec"], ["chosen0"], axis=0, keepdims=0)
    nn("Add", ["chosen0", "c1i"], ["chosen_color"])

    nn("Gather", ["idx_flat", "obj_pos"], ["obj_vals"], axis=0)
    nn("Equal", ["obj_vals", "chosen_color"], ["obj_eq"])
    nn("And", ["obj_eq", "obj_valid"], ["obj_good"])
    nn("Cast", ["obj_good"], ["obj_good_i"], to=I64)
    nn("ReduceSum", ["obj_good_i"], ["obj_count"], axes=[1], keepdims=0)
    nn("Mul", ["cand_s", "cand_s"], ["cand_area"])
    nn("Equal", ["obj_count", "cand_area"], ["block_ok"])
    nn("Gather", ["idx_flat", "bnd_pos"], ["bnd_vals"], axis=0)
    nn("Equal", ["bnd_vals", "chosen_color"], ["bnd_eq0"])
    nn("And", ["bnd_eq0", "bnd_valid"], ["bnd_eq"])
    nn("Cast", ["bnd_eq"], ["bnd_i"], to=I64)
    nn("ReduceSum", ["bnd_i"], ["bnd_count"], axes=[1], keepdims=0)
    nn("Equal", ["bnd_count", "c0i"], ["bnd_ok"])
    nn("And", ["block_ok", "bnd_ok"], ["obj_top_left"])

    nn("Gather", ["idx_flat", "crop_pos"], ["crop_vals"], axis=0)
    nn("Cast", ["crop_vals"], ["crop_vals_i"], to=I64)
    nn("Cast", ["crop_valid"], ["crop_valid_i"], to=I64)
    nn("Mul", ["crop_vals_i", "crop_valid_i"], ["crop_vals_masked"])
    nn("ReduceSum", ["crop_vals_masked"], ["crop_sum"], axes=[1], keepdims=0)
    nn("ReduceSum", ["crop_valid_i"], ["crop_valid_count"], axes=[1], keepdims=0)
    nn("Equal", ["crop_valid_count", "crop_need"], ["crop_ok"])
    nn("And", ["obj_top_left", "crop_ok"], ["source_ok"])
    nn("Where", ["source_ok", "crop_sum", "m999999i"], ["source_score"])
    nn("ArgMax", ["source_score"], ["source_idx"], axis=0, keepdims=0)
    nn("Gather", ["cand_s", "source_idx"], ["source_s"], axis=0)
    nn("Gather", ["cand_r", "source_idx"], ["source_r"], axis=0)
    nn("Gather", ["cand_c", "source_idx"], ["source_c"], axis=0)

    nn("Equal", ["cand_s_row", "source_s"], ["same_s"])
    nn("Reshape", ["obj_top_left", "shape1_2700"], ["target_ok"])
    nn("And", ["same_s", "target_ok"], ["target_same"])
    nn("Sub", ["cand_r_row", "cand_s_row"], ["win_r0a"])
    nn("Sub", ["win_r0a", "c1i"], ["win_r0"])
    nn("Mul", ["cand_s_row", "c2i"], ["s2row"])
    nn("Add", ["cand_r_row", "s2row"], ["win_r1"])
    nn("Sub", ["cand_c_row", "cand_s_row"], ["win_c0a"])
    nn("Sub", ["win_c0a", "c1i"], ["win_c0"])
    nn("Add", ["cand_c_row", "s2row"], ["win_c1"])
    nn("GreaterOrEqual", ["out_r_col", "win_r0"], ["paint_r_ge"])
    nn("LessOrEqual", ["out_r_col", "win_r1"], ["paint_r_le"])
    nn("And", ["paint_r_ge", "paint_r_le"], ["paint_r_ok"])
    nn("GreaterOrEqual", ["out_c_col", "win_c0"], ["paint_c_ge"])
    nn("LessOrEqual", ["out_c_col", "win_c1"], ["paint_c_le"])
    nn("And", ["paint_c_ge", "paint_c_le"], ["paint_c_ok"])
    nn("And", ["paint_r_ok", "paint_c_ok"], ["inside_win"])
    nn("And", ["inside_win", "target_same"], ["paint_mask"])
    nn("Sub", ["out_r_col", "cand_r_row"], ["rel_r"])
    nn("Add", ["rel_r", "source_r"], ["src_r"])
    nn("Sub", ["out_c_col", "cand_c_row"], ["rel_c"])
    nn("Add", ["rel_c", "source_c"], ["src_c"])
    nn("Mul", ["src_r", "c30i"], ["src_r30"])
    nn("Add", ["src_r30", "src_c"], ["src_flat"])
    nn("GreaterOrEqual", ["src_flat", "c0i"], ["src_ge0"])
    nn("LessOrEqual", ["src_flat", "c899i"], ["src_le899"])
    nn("And", ["src_ge0", "src_le899"], ["src_in_bounds"])
    nn("And", ["paint_mask", "src_in_bounds"], ["paint_mask_safe"])
    nn("Where", ["src_in_bounds", "src_flat", "c0i"], ["src_flat_safe"])
    nn("Gather", ["idx_flat", "src_flat_safe"], ["paint_vals"], axis=0)
    nn("Where", ["paint_mask_safe", "paint_vals", "c0i"], ["paint_vals_masked"])
    nn("ReduceMax", ["paint_vals_masked"], ["paint_val"], axes=[1], keepdims=0)
    nn("Cast", ["paint_mask_safe"], ["paint_mask_i"], to=I64)
    nn("ReduceMax", ["paint_mask_i"], ["hit_i"], axes=[1], keepdims=0)
    nn("Greater", ["hit_i", "c0i"], ["hit"])
    nn("Where", ["hit", "paint_val", "idx_flat"], ["out_flat"])
    nn("Reshape", ["out_flat", "shape1_1_30_30"], ["out_idx"])
    nn("Squeeze", ["out_idx"], ["out_sq"], axes=[1])
    nn("OneHot", ["out_sq", "depth10", "oh_vals"], ["oh_raw"], axis=1)
    nn("Mul", ["oh_raw", "presence"], ["output"])

    graph = helper.make_graph(nodes, "task080_dsl", [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid("", 12)])
    _set_out_shape(m, [1, 10, 30, 30])
    return m


model = _bake(_make(), 80)


def save(path=None):
    if path is None:
        path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "task080.onnx"))
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    print(save())
