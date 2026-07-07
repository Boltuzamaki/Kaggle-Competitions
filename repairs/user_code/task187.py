# --- task187: enclosed background regions -> 2, border-connected background -> 3.
# Non-background colours are left untouched (verified against all 266 train+test+arc-gen
# examples). Implementation: unrolled BFS-style dilation (plus-shaped Conv, 4-connectivity)
# seeded from the full 30x30 canvas border through "free" cells (background-or-padding).
# Padding cells never carry channel-0=1 in the one-hot encoding (only real grid cells do),
# so the flood correctly treats the zero-padded margin as part of the "outside" and the
# recolour deltas (built from real channel-0 presence only) are automatically zero outside
# the true grid extent -> no separate crop/mask step needed. Verified n_fail=0 (min required
# ITERS empirically 23; using 32 for safety margin).
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64
ITERS = 32


def _K(n, a, d=np.float32):
    return numpy_helper.from_array(np.array(a, dtype=d), name=n)


def build_187():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    border = np.zeros((1, 1, 30, 30), np.float32)
    border[:, :, 0, :] = 1; border[:, :, 29, :] = 1
    border[:, :, :, 0] = 1; border[:, :, :, 29] = 1

    inits = [
        _K('idx0', [0], np.int64),
        _K('c1s', [0, 1, 0, 0], np.int64), _K('c1e', [1, 10, 30, 30], np.int64), _K('axall', [0, 1, 2, 3], np.int64),
        _K('ax1', [1], np.int64),
        _K('one', [1.0], np.float32), _K('zero', [0.0], np.float32), _K('neg1', [-1.0], np.float32),
        _K('border', border, np.float32),
        _K('plus', np.array([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]], np.float32), np.float32),
        _K('pad0', [0, 0, 0, 0, 0, 9, 0, 0], np.int64),
        _K('pad2', [0, 2, 0, 0, 0, 7, 0, 0], np.int64),
        _K('pad3', [0, 3, 0, 0, 0, 6, 0, 0], np.int64),
    ]
    nodes = []

    # ch0 = real background-cell presence (only set for actual grid cells, never padding)
    nodes.append(helper.make_node('Gather', ['input', 'idx0'], ['ch0'], axis=1))
    # free = background-or-padding cells (anything with no colour 1..9 present)
    nodes.append(helper.make_node('Slice', ['input', 'c1s', 'c1e', 'axall'], ['nz']))
    nodes.append(helper.make_node('ReduceSum', ['nz', 'ax1'], ['nonbg'], keepdims=1))
    nodes.append(helper.make_node('Sub', ['one', 'nonbg'], ['free']))

    prev = 'seed'
    nodes.append(helper.make_node('Mul', ['free', 'border'], [prev]))
    for k in range(1, ITERS + 1):
        nodes.append(helper.make_node('Conv', [prev, 'plus'], [f'd{k}'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node('Clip', [f'd{k}', 'zero', 'one'], [f'dc{k}']))
        nodes.append(helper.make_node('Mul', [f'dc{k}', 'free'], [f'r{k}']))
        prev = f'r{k}'
    reached = prev

    nodes.append(helper.make_node('Sub', ['one', reached], ['notreached']))
    nodes.append(helper.make_node('Mul', ['ch0', 'notreached'], ['enc']))   # enclosed bg -> colour 2
    nodes.append(helper.make_node('Mul', ['ch0', reached], ['ext']))        # border-connected bg -> colour 3
    nodes.append(helper.make_node('Mul', ['ch0', 'neg1'], ['negch0']))

    nodes.append(helper.make_node('Pad', ['negch0', 'pad0'], ['delta0']))
    nodes.append(helper.make_node('Pad', ['enc', 'pad2'], ['delta2']))
    nodes.append(helper.make_node('Pad', ['ext', 'pad3'], ['delta3']))

    nodes.append(helper.make_node('Add', ['input', 'delta0'], ['t1']))
    nodes.append(helper.make_node('Add', ['t1', 'delta2'], ['t2']))
    nodes.append(helper.make_node('Add', ['t2', 'delta3'], ['output']))

    graph = helper.make_graph(nodes, 'task187', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json" % t)
def _reps(t, k=8):
    d = json.load(open(_resolve_task_json(t)))
    exs = sorted(d["train"] + d["test"] + d["arc-gen"], key=lambda e: (len(e["input"]), len(e["input"][0])))
    idx = set([0, len(exs) - 1]) | set(int(j * (len(exs) - 1) / (k - 1)) for j in range(1, k - 1))
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
    good = set(vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi))
    good |= set(x.name for x in list(m.graph.input) + list(m.graph.output))
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
    return build_187()


model = _bake(_make(), 187)
