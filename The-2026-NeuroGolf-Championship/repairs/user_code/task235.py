import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64

# ---------------------------------------------------------------------------
# Task 235 rule (arc-dsl solve_995c5fa3), fully traced from arc_dsl_ref/solvers.py:
#
#   x1 = hsplit(I, THREE)                      # 3 vertical strips (same height)
#   for each strip:
#     x3 = ofcolor(strip, ZERO)                # cells with value 0
#     x4 = ulcorner(x3)                        # (min row, min col) of those cells
#     x5 = size(x3)
#     x6 = (x5 == 0)              -> value 2   # strip has NO zero cells at all
#     x7 = (x4 == (1,1))          -> value 8   # UNITY
#     x8 = (x4 == (1,0))          -> value 3   # DOWN
#     x9 = (x4 == (2,1))          -> value 4   # (TWO, ONE)
#     x18 = 2*x6 + 8*x7 + 3*x8 + 4*x9          # (mutually exclusive by construction)
#   O = 3x3 grid, row i filled with the per-strip value, upscaled x3 horizontally
#
# All 69 train+test+arc-gen inputs in data/task235.json are the SAME 4x14 shape,
# so hsplit(I,3) with w=14//3=4, offset=1 (since 14%3!=0) always gives:
#   strip0 = cols[0:4], strip1 = cols[5:9], strip2 = cols[10:14]   (rows 0:4)
# Verified exactly (0 failures) against all 69 examples via pure-numpy replica
# before writing any ONNX.
# ---------------------------------------------------------------------------

STRIP_COLS = [(0, 4), (5, 9), (10, 14)]


def _K(name, arr, dtype=np.int64):
    return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)


def build_235():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 3, 3])

    inits = [
        _K('row_idx4', np.arange(4).reshape(1, 1, 4, 1), np.int64),
        _K('col_idx4', np.arange(4).reshape(1, 1, 1, 4), np.int64),
        _K('p999', [999], np.int64),
        _K('half', [0.5], np.float32),
        _K('zero_f', [0.0], np.float32),
        _K('zero_i', [0], np.int64),
        _K('one_i', [1], np.int64),
        _K('two_i', [2], np.int64),
        _K('three_i', [3], np.int64),
        _K('four_i', [4], np.int64),
        _K('eight_i', [8], np.int64),
        _K('shape_111', [1, 1, 1], np.int64),
        _K('shape_row3', [1, 1, 3], np.int64),
        _K('depth10', [10], np.int64),
        _K('oh_vals', [0.0, 1.0], np.float32),
        _K('axes123', [1, 2, 3], np.int64),
        _K('ax_row', [1], np.int64),  # unused placeholder removed below
    ]
    # drop the unused placeholder (kept list construction simple above)
    inits = [k for k in inits if k.name != 'ax_row']

    nodes = []
    row_names = []

    for i, (cs, ce) in enumerate(STRIP_COLS):
        starts_name = f'starts_{i}'
        ends_name = f'ends_{i}'
        inits.append(_K(starts_name, [0, 0, cs], np.int64))
        inits.append(_K(ends_name, [1, 4, ce], np.int64))

        mask = f'zmask_{i}'
        nodes.append(helper.make_node('Slice', ['input', starts_name, ends_name, 'axes123'], [mask]))

        # row-min of zero cells (sentinel 999 if none in that row)
        row_any = f'row_any_{i}'
        row_b = f'row_b_{i}'
        row_sel = f'row_sel_{i}'
        r_min = f'r_min_{i}'
        nodes.append(helper.make_node('ReduceMax', [mask], [row_any], axes=[3], keepdims=1))
        nodes.append(helper.make_node('Greater', [row_any, 'half'], [row_b]))
        nodes.append(helper.make_node('Where', [row_b, 'row_idx4', 'p999'], [row_sel]))
        nodes.append(helper.make_node('ReduceMin', [row_sel], [r_min], axes=[2], keepdims=1))

        # col-min of zero cells (sentinel 999 if none in that col)
        col_any = f'col_any_{i}'
        col_b = f'col_b_{i}'
        col_sel = f'col_sel_{i}'
        c_min = f'c_min_{i}'
        nodes.append(helper.make_node('ReduceMax', [mask], [col_any], axes=[2], keepdims=1))
        nodes.append(helper.make_node('Greater', [col_any, 'half'], [col_b]))
        nodes.append(helper.make_node('Where', [col_b, 'col_idx4', 'p999'], [col_sel]))
        nodes.append(helper.make_node('ReduceMin', [col_sel], [c_min], axes=[3], keepdims=1))

        # count of zero cells in the strip -> is the strip "full" (no zero cells)?
        cnt = f'cnt_{i}'
        is_full = f'is_full_{i}'
        nodes.append(helper.make_node('ReduceSum', [mask, 'axes123'], [cnt], keepdims=1))
        nodes.append(helper.make_node('Equal', [cnt, 'zero_f'], [is_full]))

        eq_r1 = f'eq_r1_{i}'
        eq_r2 = f'eq_r2_{i}'
        eq_c0 = f'eq_c0_{i}'
        eq_c1 = f'eq_c1_{i}'
        nodes.append(helper.make_node('Equal', [r_min, 'one_i'], [eq_r1]))
        nodes.append(helper.make_node('Equal', [r_min, 'two_i'], [eq_r2]))
        nodes.append(helper.make_node('Equal', [c_min, 'zero_i'], [eq_c0]))
        nodes.append(helper.make_node('Equal', [c_min, 'one_i'], [eq_c1]))

        m_unity = f'm_unity_{i}'  # ulcorner == (1,1) -> 8
        m_down = f'm_down_{i}'    # ulcorner == (1,0) -> 3
        m_21 = f'm_21_{i}'        # ulcorner == (2,1) -> 4
        nodes.append(helper.make_node('And', [eq_r1, eq_c1], [m_unity]))
        nodes.append(helper.make_node('And', [eq_r1, eq_c0], [m_down]))
        nodes.append(helper.make_node('And', [eq_r2, eq_c1], [m_21]))

        va = f'va_{i}'
        vb = f'vb_{i}'
        vc = f'vc_{i}'
        vd = f'vd_{i}'
        nodes.append(helper.make_node('Where', [m_21, 'four_i', 'zero_i'], [va]))
        nodes.append(helper.make_node('Where', [m_down, 'three_i', va], [vb]))
        nodes.append(helper.make_node('Where', [m_unity, 'eight_i', vb], [vc]))
        nodes.append(helper.make_node('Where', [is_full, 'two_i', vc], [vd]))

        vr = f'vr_{i}'
        row = f'row_{i}'
        nodes.append(helper.make_node('Reshape', [vd, 'shape_111'], [vr]))
        nodes.append(helper.make_node('Expand', [vr, 'shape_row3'], [row]))
        row_names.append(row)

    nodes.append(helper.make_node('Concat', row_names, ['raw_grid'], axis=1))
    nodes.append(helper.make_node('OneHot', ['raw_grid', 'depth10', 'oh_vals'], ['output'], axis=1))

    graph = helper.make_graph(nodes, 'task235', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _rename_output(m, new):
    for nd in m.graph.node:
        for i, o in enumerate(nd.output):
            if o == "output":
                nd.output[i] = new
                return
def _set_out_shape(m, dims):
    tt = m.graph.output[0].type.tensor_type
    tt.elem_type = TensorProto.FLOAT
    del tt.shape.dim[:]
    for d in dims:
        tt.shape.dim.add().dim_value = d
def _crop_pad(m):
    """OneHot 'output' is a static [1,10,3,3] result; Pad to static 30x30
    using h,w read from the tensor's own Shape (keeps padding all-zero)."""
    _rename_output(m, "oh_raw")
    m.graph.initializer.extend([_K("__s2", [2], np.int64), _K("__e4", [4], np.int64), _K("__a0", [0], np.int64),
        _K("__30x2", [30, 30], np.int64), _K("__pfx6", [0, 0, 0, 0, 0, 0], np.int64), _K("__pv", [0.0], np.float32)])
    m.graph.node.extend([
        helper.make_node("Shape", ["oh_raw"], ["__osh"]),
        helper.make_node("Slice", ["__osh", "__s2", "__e4", "__a0"], ["__hw"]),
        helper.make_node("Sub", ["__30x2", "__hw"], ["__padhw"]),
        helper.make_node("Concat", ["__pfx6", "__padhw"], ["__pads"], axis=0),
        helper.make_node("Pad", ["oh_raw", "__pads", "__pv"], ["output"], mode="constant")])
    _set_out_shape(m, [1, 10, 30, 30])
    return m
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p):
            return p
    raise FileNotFoundError("task%03d.json" % t)
def _reps(t, k=8):
    import onnxruntime as _ort  # noqa
    d = json.load(open(_resolve_task_json(t)))
    exs = sorted(d["train"] + d["test"] + d["arc-gen"], key=lambda e: (len(e["input"]), len(e["input"][0])))
    idx = set([0, len(exs) - 1]) | {int(j * (len(exs) - 1) / (k - 1)) for j in range(1, k - 1)}
    out = []
    for i in sorted(idx):
        g = exs[i]["input"]
        a = np.zeros((1, 10, 30, 30), np.float32)
        for r, row in enumerate(g):
            for c, v in enumerate(row):
                a[0][v][r][c] = 1.0
        out.append(a)
    return out
def _bake(m, t):
    import onnxruntime as _ort
    inf = onnx.shape_inference.infer_shapes(_copy.deepcopy(m), strict_mode=True)
    def sym(vi): return any(dd.HasField("dim_param") or not dd.HasField("dim_value") for dd in vi.type.tensor_type.shape.dim)
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
    mx = {}
    dt = {}
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
    return _crop_pad(build_235())


model = _bake(_make(), 235)

if __name__ == "__main__":
    onnx.save(model, _os.path.join(_os.path.dirname(__file__), "..", "task235.onnx"))
