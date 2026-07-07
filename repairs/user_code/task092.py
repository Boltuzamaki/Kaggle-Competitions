import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT
I64 = TensorProto.INT64

# Rule (verified vs solve_40853293 / arc_dsl_ref, and vs train+test+arc-gen in pure numpy first):
#   x1 = partition(I)                      # one object per color value (incl. background), purely color-based
#   x2 = fork(recolor, color, backdrop)    # replace each object's cells with its own bbox rectangle, same color
#   x3 = apply(x2, x1)
#   x4 = mfilter(x3, hline)                # keep bbox-rectangles that are exactly 1 row tall -> paint first
#   x5 = mfilter(x3, vline)                # keep bbox-rectangles that are exactly 1 col wide  -> paint on top
#   O  = paint(paint(I, x4), x5)
# i.e. per color c: if its bbox (over the ORIGINAL grid) has height==1, fill that row-span [cmin,cmax] with c;
#      if its bbox has width==1, fill that col-span [rmin,rmax] with c (applied after, so wins ties/overlaps).
# A single isolated cell satisfies both hline and vline (trivially), but the painted value is identical either way.


def create_model():
    # cost = memory (sum of intermediate tensor byte-sizes) + params, so this is built to
    # keep every per-color intermediate as small/narrow as possible:
    #  - row/col presence is derived ONCE for all 10 colors via a single ReduceMax over the
    #    full 10-channel input (shape [1,10,30,1]/[1,10,1,30]), then cheaply Slice'd to size-30
    #    per color, instead of materializing a full 900-elem per-color channel slice.
    #  - the row/col-range AND is reordered so the two size-30 conditions combine first and
    #    only the final [1,1,30,30] mask tensor is full-size (one 900-elem bool per stage/color
    #    instead of two).
    #  - the sequential per-color grid repaint chain uses int32 (not ArgMax's native int64) to
    #    halve the cost of its 20 [1,1,30,30] intermediate copies.
    I32 = TensorProto.INT32
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    inits = [
        K('row_idx', np.arange(30).reshape(1, 1, 30, 1), np.int32),
        K('col_idx', np.arange(30).reshape(1, 1, 1, 30), np.int32),
        K('m1', [-1], np.int32), K('p999', [999], np.int32),
        K('ax1', [1]),
        K('c0_f', [0.0], np.float32),
        K('depth10', [10]),
        K('oh_vals', [0.0, 1.0], np.float32),
        K('shape_1_30_30', [1, 30, 30]),
    ]
    for c in range(10):
        inits.append(K(f'st{c}', [c]))
        inits.append(K(f'en{c}', [c + 1]))
        inits.append(K(f'cv{c}', [c], np.int32))

    n = []
    # base grid = argmax over channels of the (one-hot) input == the original color grid
    n.append(helper.make_node('ArgMax', ['input'], ['grid0_64'], axis=1, keepdims=1))
    n.append(helper.make_node('Cast', ['grid0_64'], ['grid0'], to=I32))

    # row/col "any color present" computed once for ALL 10 channels (shape [1,10,30,1] / [1,10,1,30])
    n.append(helper.make_node('ReduceMax', ['input'], ['row_any_all'], axes=[3], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['input'], ['col_any_all'], axes=[2], keepdims=1))

    hmasks = []
    vmasks = []
    for c in range(10):
        n.append(helper.make_node('Slice', ['row_any_all', f'st{c}', f'en{c}', 'ax1'], [f'rowany{c}']))
        n.append(helper.make_node('Greater', [f'rowany{c}', 'c0_f'], [f'rowb{c}']))
        n.append(helper.make_node('Where', [f'rowb{c}', 'row_idx', 'p999'], [f'rminsrc{c}']))
        n.append(helper.make_node('ReduceMin', [f'rminsrc{c}'], [f'rmin{c}'], axes=[2], keepdims=1))
        n.append(helper.make_node('Where', [f'rowb{c}', 'row_idx', 'm1'], [f'rmaxsrc{c}']))
        n.append(helper.make_node('ReduceMax', [f'rmaxsrc{c}'], [f'rmax{c}'], axes=[2], keepdims=1))

        n.append(helper.make_node('Slice', ['col_any_all', f'st{c}', f'en{c}', 'ax1'], [f'colany{c}']))
        n.append(helper.make_node('Greater', [f'colany{c}', 'c0_f'], [f'colb{c}']))
        n.append(helper.make_node('Where', [f'colb{c}', 'col_idx', 'p999'], [f'cminsrc{c}']))
        n.append(helper.make_node('ReduceMin', [f'cminsrc{c}'], [f'cmin{c}'], axes=[3], keepdims=1))
        n.append(helper.make_node('Where', [f'colb{c}', 'col_idx', 'm1'], [f'cmaxsrc{c}']))
        n.append(helper.make_node('ReduceMax', [f'cmaxsrc{c}'], [f'cmax{c}'], axes=[3], keepdims=1))

        n.append(helper.make_node('Equal', [f'rmax{c}', f'rmin{c}'], [f'is_h{c}']))
        n.append(helper.make_node('Equal', [f'cmax{c}', f'cmin{c}'], [f'is_v{c}']))

        # small (size-30) pieces combined first, so only ONE full [1,1,30,30] tensor per mask
        n.append(helper.make_node('Equal', ['row_idx', f'rmin{c}'], [f'rowmatch{c}']))
        n.append(helper.make_node('And', [f'rowmatch{c}', f'is_h{c}'], [f'rowmatch_h{c}']))
        n.append(helper.make_node('GreaterOrEqual', ['col_idx', f'cmin{c}'], [f'colge{c}']))
        n.append(helper.make_node('LessOrEqual', ['col_idx', f'cmax{c}'], [f'colle{c}']))
        n.append(helper.make_node('And', [f'colge{c}', f'colle{c}'], [f'colrange{c}']))
        n.append(helper.make_node('And', [f'rowmatch_h{c}', f'colrange{c}'], [f'hmask{c}']))
        hmasks.append((c, f'hmask{c}'))

        n.append(helper.make_node('Equal', ['col_idx', f'cmin{c}'], [f'colmatch{c}']))
        n.append(helper.make_node('And', [f'colmatch{c}', f'is_v{c}'], [f'colmatch_v{c}']))
        n.append(helper.make_node('GreaterOrEqual', ['row_idx', f'rmin{c}'], [f'rowge{c}']))
        n.append(helper.make_node('LessOrEqual', ['row_idx', f'rmax{c}'], [f'rowle{c}']))
        n.append(helper.make_node('And', [f'rowge{c}', f'rowle{c}'], [f'rowrange{c}']))
        n.append(helper.make_node('And', [f'colmatch_v{c}', f'rowrange{c}'], [f'vmask{c}']))
        vmasks.append((c, f'vmask{c}'))

    # paint hlines first (ascending color order), then vlines on top of the result
    prev = 'grid0'
    for c, mask in hmasks:
        cur = f'gridh{c}'
        n.append(helper.make_node('Where', [mask, f'cv{c}', prev], [cur]))
        prev = cur
    for c, mask in vmasks:
        cur = f'gridv{c}'
        n.append(helper.make_node('Where', [mask, f'cv{c}', prev], [cur]))
        prev = cur

    n.append(helper.make_node('Reshape', [prev, 'shape_1_30_30'], ['grid_final32']))
    n.append(helper.make_node('Cast', ['grid_final32'], ['grid_final'], to=I64))
    n.append(helper.make_node('OneHot', ['grid_final', 'depth10', 'oh_vals'], ['output'], axis=1))

    graph = helper.make_graph(n, 'task092', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n, a, d): return numpy_helper.from_array(np.array(a, dtype=d), name=n)
def _rename_output(m, new):
    for nd in m.graph.node:
        for i, o in enumerate(nd.output):
            if o == "output": nd.output[i] = new; return
def _mask(m):
    """Same-shape task: padded cells outside the true grid have ALL channels==0 in
    both input and expected output (convert_to_numpy leaves them fully zero, not
    channel-0=1) — zero the OneHot's guessed channel-0 there via an input-presence mask."""
    _rename_output(m, "oh_raw")
    m.graph.node.append(helper.make_node('ReduceMax', ['input'], ['presence_m'], axes=[1], keepdims=1))
    m.graph.node.append(helper.make_node('Mul', ['oh_raw', 'presence_m'], ['output']))
    return m
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json" % t)
def _reps(t, k=8):
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
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64,
            np.dtype("bool"): TensorProto.BOOL, np.dtype("int32"): TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m


def _make():
    return _mask(create_model())


model = _bake(_make(), 92)

if __name__ == "__main__":
    onnx.save(model, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "task092.onnx"))
    print("saved repairs/task092.onnx")
