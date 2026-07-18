import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT
I64 = TensorProto.INT64


def _K(n, a, d=np.float32):
    return numpy_helper.from_array(np.array(a, dtype=d), name=n)


def build_090():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    inits = [
        _K('idx0', [0], np.int64),
        _K('shape1', [1], np.int64),
        _K('shape1111', [1, 1, 1, 1], np.int64),
        _K('one_c', [1.0], np.float32),
        _K('onehot6', np.array([1.0 if c == 6 else 0.0 for c in range(10)]).reshape(1, 10, 1, 1), np.float32),
    ]
    nodes = []

    # M = channel-0 (background) presence mask, [1,1,30,30]. Padding cells (beyond the
    # true grid extent) have ALL channels 0, so this is 1 only at real background cells.
    nodes.append(helper.make_node('Gather', ['input', 'idx0'], ['M'], axis=1))

    scores = []
    contribs = []
    for h in range(2, 10):
        for w in range(2, 10):
            tag = f'{h}_{w}'
            area = float(h * w)
            inits.append(_K(f'kw_{tag}', np.ones((1, 1, h, w), dtype=np.float32), np.float32))
            inits.append(_K(f'thr_{tag}', [area - 0.5], np.float32))
            inits.append(_K(f'area_{tag}', [area], np.float32))

            # sum of M over every h x w window -> exact count of background cells in it
            nodes.append(helper.make_node('Conv', ['M', f'kw_{tag}'], [f'sum_{tag}'],
                                           kernel_shape=[h, w], strides=[1, 1], pads=[0, 0, 0, 0]))
            nodes.append(helper.make_node('Greater', [f'sum_{tag}', f'thr_{tag}'], [f'gt_{tag}']))
            nodes.append(helper.make_node('Cast', [f'gt_{tag}'], [f'matchf_{tag}'], to=F))
            nodes.append(helper.make_node('ReduceMax', [f'matchf_{tag}'], [f'any4_{tag}'],
                                           axes=[0, 1, 2, 3], keepdims=1))
            nodes.append(helper.make_node('Reshape', [f'any4_{tag}', 'shape1'], [f'any_{tag}']))
            nodes.append(helper.make_node('Mul', [f'any_{tag}', f'area_{tag}'], [f'score_{tag}']))
            scores.append(f'score_{tag}')

    nodes.append(helper.make_node('Concat', scores, ['scores_all'], axis=0))
    nodes.append(helper.make_node('ReduceMax', ['scores_all'], ['best_area'], axes=[0], keepdims=1))

    for h in range(2, 10):
        for w in range(2, 10):
            tag = f'{h}_{w}'
            nodes.append(helper.make_node('Equal', [f'score_{tag}', 'best_area'], [f'sel_b_{tag}']))
            nodes.append(helper.make_node('Cast', [f'sel_b_{tag}'], [f'sel_f_{tag}'], to=F))
            nodes.append(helper.make_node('Reshape', [f'sel_f_{tag}', 'shape1111'], [f'sel4_{tag}']))
            nodes.append(helper.make_node('Mul', [f'matchf_{tag}', f'sel4_{tag}'], [f'gated_{tag}']))
            nodes.append(helper.make_node('ConvTranspose', [f'gated_{tag}', f'kw_{tag}'], [f'contrib_{tag}'],
                                           kernel_shape=[h, w], strides=[1, 1], pads=[0, 0, 0, 0]))
            contribs.append(f'contrib_{tag}')

    # sum all (at most one is non-zero, by construction of the ARC rule verified vs data)
    acc = contribs[0]
    for i, nm in enumerate(contribs[1:]):
        nodes.append(helper.make_node('Add', [acc, nm], [f'acc_{i}']))
        acc = f'acc_{i}'
    inits.append(_K('half_c', [0.5], np.float32))
    nodes.append(helper.make_node('Greater', [acc, 'half_c'], ['mask_b']))
    nodes.append(helper.make_node('Cast', ['mask_b'], ['final_mask'], to=F))

    nodes.append(helper.make_node('Sub', ['one_c', 'final_mask'], ['inv_mask']))
    nodes.append(helper.make_node('Mul', ['input', 'inv_mask'], ['kept']))
    nodes.append(helper.make_node('Mul', ['final_mask', 'onehot6'], ['added']))
    nodes.append(helper.make_node('Add', ['kept', 'added'], ['output']))

    graph = helper.make_graph(nodes, 'task090', [x], [y], inits)
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
    return build_090()


model = _bake(_make(), 90)
