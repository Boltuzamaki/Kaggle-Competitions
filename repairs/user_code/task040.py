import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

# ===== task040: 2204b7a8 =====
# Grid is always exactly 10x10 (placed at top-left of the 30x30 canvas).
# It contains one solid border row (top+bottom) OR one solid border column
# (left+right), plus scattered color-3 noise cells elsewhere.
# Determine orientation from whether row0 is a single nonzero color (horizontal
# split) or col0 is a single nonzero color (vertical split) -- exactly one holds.
# For horizontal split: cells with r<5 that are color 3 get replaced by the
# top-border color (= value at (0,0)); cells with r>=5 get the bottom-border
# color (= value at (9,0)).
# For vertical split: cells with c<5 get replaced by the left-border color
# (= value at (0,0)); cells with c>=5 get the right-border color (= value at (0,9)).
# Verified n_fail==0 against every train+test+arc-gen example in data/task040.json.

def create_model():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    def K(name, arr, dtype):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    row_lt5 = np.zeros((1, 10, 1), dtype=np.int64)
    row_lt5[0, 0:5, 0] = 1
    col_lt5 = np.zeros((1, 1, 10), dtype=np.int64)
    col_lt5[0, 0, 0:5] = 1

    inits = [
        K('s00', [0, 0], np.int64), K('e1010', [10, 10], np.int64), K('ax23', [2, 3], np.int64),
        K('s00b', [0, 0], np.int64), K('e11', [1, 1], np.int64), K('ax12', [1, 2], np.int64),
        K('e110', [1, 10], np.int64), K('e101', [10, 1], np.int64),
        K('s90', [9, 0], np.int64), K('e101_bl', [10, 1], np.int64),
        K('s09', [0, 9], np.int64), K('e110_tr', [1, 10], np.int64),
        K('c0_f', [0.0], np.float32),
        K('c3', [3], np.int64),
        K('row_lt5', row_lt5, np.int64),
        K('col_lt5', col_lt5, np.int64),
        K('depth10', [10], np.int64), K('oh_vals', [0.0, 1.0], np.float32),
        K('pads', [0, 0, 0, 0, 0, 0, 20, 20], np.int64),
        K('pv', [0.0], np.float32),
    ]

    n = []
    # crop static 10x10 region
    n.append(helper.make_node('Slice', ['input', 's00', 'e1010', 'ax23'], ['grid10']))
    n.append(helper.make_node('ArgMax', ['grid10'], ['colidx'], axis=1, keepdims=0))  # [1,10,10] int64

    # corners / edges
    n.append(helper.make_node('Slice', ['colidx', 's00b', 'e11', 'ax12'], ['corner00']))       # [1,1,1] top-left
    n.append(helper.make_node('Slice', ['colidx', 's00b', 'e110', 'ax12'], ['row0']))           # [1,1,10]
    n.append(helper.make_node('Slice', ['colidx', 's00b', 'e101', 'ax12'], ['col0']))           # [1,10,1]
    n.append(helper.make_node('Slice', ['colidx', 's90', 'e101_bl', 'ax12'], ['cornerBL']))     # [1,1,1] bottom-left
    n.append(helper.make_node('Slice', ['colidx', 's09', 'e110_tr', 'ax12'], ['cornerTR']))     # [1,1,1] top-right

    # row0_const_bool = all(row0 == corner00) AND corner00 != 0
    n.append(helper.make_node('Equal', ['row0', 'corner00'], ['eq_row0']))
    n.append(helper.make_node('Cast', ['eq_row0'], ['eq_row0_i'], to=I64))
    n.append(helper.make_node('ReduceMin', ['eq_row0_i'], ['row0_all_eq'], axes=[2], keepdims=1))
    n.append(helper.make_node('Cast', ['row0_all_eq'], ['row0_all_eq_b'], to=BOOL))
    n.append(helper.make_node('Cast', ['corner00'], ['corner00_f'], to=F))
    n.append(helper.make_node('Greater', ['corner00_f', 'c0_f'], ['corner00_nz']))
    n.append(helper.make_node('And', ['row0_all_eq_b', 'corner00_nz'], ['row0_const']))  # [1,1,1] bool

    # color used for the "B" half: bottom-border color if horizontal, right-border color if vertical
    n.append(helper.make_node('Where', ['row0_const', 'cornerBL', 'cornerTR'], ['color_B']))

    # selector: True => use corner00 (A-half color), False => use color_B
    n.append(helper.make_node('Where', ['row0_const', 'row_lt5', 'col_lt5'], ['selectorA_i']))  # [1,10,10] int64
    n.append(helper.make_node('Cast', ['selectorA_i'], ['selectorA'], to=BOOL))
    n.append(helper.make_node('Where', ['selectorA', 'corner00', 'color_B'], ['color_choice']))  # [1,10,10]

    n.append(helper.make_node('Equal', ['colidx', 'c3'], ['is_three']))
    n.append(helper.make_node('Where', ['is_three', 'color_choice', 'colidx'], ['final_idx']))  # [1,10,10]

    n.append(helper.make_node('OneHot', ['final_idx', 'depth10', 'oh_vals'], ['oh_raw'], axis=1))  # [1,10,10,10]
    n.append(helper.make_node('Pad', ['oh_raw', 'pads', 'pv'], ['output'], mode='constant'))

    graph = helper.make_graph(n, 'task040', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n, a, d): return numpy_helper.from_array(np.array(a, dtype=d), name=n)
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json" % t)
def _reps(t, k=8):
    import onnxruntime as _ort  # noqa
    d = json.load(open(_resolve_task_json(t)))
    exs = sorted(d["train"] + d["test"] + d["arc-gen"], key=lambda e: (len(e["input"]), len(e["input"][0])))
    idx = set([0, len(exs) - 1]) | {int(j * (len(exs) - 1) / (k - 1)) for j in range(1, k - 1)}
    out = []
    for i in sorted(idx):
        g = exs[i]["input"]; a = np.zeros((1, 10, 30, 30), np.float32)
        for r, row in enumerate(g):
            for c, v in enumerate(row): a[0][v][r][c] = 1.0
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
            if o and o != "output" and o not in good and o not in missing: missing.append(o)
    if not missing: return m
    tmp = _copy.deepcopy(m)
    for nm in missing:
        vi = onnx.ValueInfoProto(); vi.name = nm; tmp.graph.output.append(vi)
    so = _ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = _ort.InferenceSession(tmp.SerializeToString(), so)
    mx = {}; dt = {}
    for inp in _reps(t):
        for nm, arr in zip(missing, s.run(missing, {"input": inp})):
            sh = list(arr.shape); mx[nm] = [max(a, b) for a, b in zip(mx[nm], sh)] if nm in mx else sh; dt[nm] = arr.dtype
    keep = [vi for vi in m.graph.value_info if vi.name not in missing]
    del m.graph.value_info[:]; m.graph.value_info.extend(keep)
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("bool"): TensorProto.BOOL, np.dtype("int32"): TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m

def _make():
    return create_model()

model = _bake(_make(), 40)
