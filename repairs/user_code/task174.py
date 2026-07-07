import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT
I64 = TensorProto.INT64

# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n, a, d): return numpy_helper.from_array(np.array(a, dtype=d), name=n)
def _rename_output(m, new):
    for nd in m.graph.node:
        for i, o in enumerate(nd.output):
            if o == "output": nd.output[i] = new; return
def _set_out_shape(m, dims):
    tt = m.graph.output[0].type.tensor_type; tt.elem_type = TensorProto.FLOAT; del tt.shape.dim[:]
    for d in dims: tt.shape.dim.add().dim_value = d
def _crop_pad(m):
    """OneHot/Slice 'output' is a dynamic [1,10,h,w] crop at top-left; Pad to static 30x30
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
    _set_out_shape(m, [1, 10, 30, 30]); return m
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


def build_174():
    """Rule (task174 / 72ca375d): among the 8-connected single-colored objects in
    the grid, find the one whose own subgrid equals its own horizontal (vmirror)
    reflection, and output that subgrid (including any background gaps inside its
    own bounding box). Verified: in every train+test+arc-gen example, each color
    forms exactly one 8-connected object, and exactly one color's subgrid is
    left-right symmetric (no ties) - n_fail==0 on the full numpy reference.

    All per-color symmetry checks below are done on FIXED [1,1,30,30] tensors
    (via a clipped GatherElements mirror-index lookup) so no intermediate has a
    genuinely input-dependent shape; only the single final crop (choosing the
    winning color's bbox) is dynamic, exactly like the established 031/049 style.
    """
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 'H_out', 'W_out'])

    I = [
        _K('ax1', [1], np.int64), _K('ax2', [2], np.int64), _K('ax3', [3], np.int64),
        _K('row_idx', np.arange(30).reshape(1, 1, 30, 1), np.int64),
        _K('col_idx', np.arange(30).reshape(1, 1, 1, 30), np.int64),
        _K('m1', [-1], np.int64), _K('p999', [999], np.int64),
        _K('c0f', [0.0], np.float32), _K('half', [0.5], np.float32), _K('onef', [1.0], np.float32),
        _K('c1', [1], np.int64), _K('shape1d', [-1], np.int64),
        _K('full_shape', [1, 1, 30, 30], np.int64),
        _K('idx_min', [0], np.int64), _K('idx_max', [29], np.int64),
    ]
    n = []

    weighted_r0, weighted_r1, weighted_c0, weighted_c1 = [], [], [], []

    for c in range(1, 10):
        I.append(_K(f'cc{c}', [c], np.int64))
        I.append(_K(f'cc1{c}', [c + 1], np.int64))

        n.append(helper.make_node('Slice', ['input', f'cc{c}', f'cc1{c}', 'ax1'], [f'chan{c}']))

        # presence of this color anywhere
        n.append(helper.make_node('ReduceMax', [f'chan{c}'], [f'anyc{c}'], axes=[0, 1, 2, 3], keepdims=1))
        n.append(helper.make_node('Greater', [f'anyc{c}', 'c0f'], [f'present{c}']))

        # row bbox
        n.append(helper.make_node('ReduceMax', [f'chan{c}'], [f'rowany{c}'], axes=[3], keepdims=1))
        n.append(helper.make_node('Greater', [f'rowany{c}', 'c0f'], [f'rowb{c}']))
        n.append(helper.make_node('Where', [f'rowb{c}', 'row_idx', 'm1'], [f'rowpmax{c}']))
        n.append(helper.make_node('ReduceMax', [f'rowpmax{c}'], [f'rmax{c}'], axes=[2], keepdims=1))
        n.append(helper.make_node('Where', [f'rowb{c}', 'row_idx', 'p999'], [f'rowpmin{c}']))
        n.append(helper.make_node('ReduceMin', [f'rowpmin{c}'], [f'rmin{c}'], axes=[2], keepdims=1))

        # col bbox
        n.append(helper.make_node('ReduceMax', [f'chan{c}'], [f'colany{c}'], axes=[2], keepdims=1))
        n.append(helper.make_node('Greater', [f'colany{c}', 'c0f'], [f'colb{c}']))
        n.append(helper.make_node('Where', [f'colb{c}', 'col_idx', 'm1'], [f'colpmax{c}']))
        n.append(helper.make_node('ReduceMax', [f'colpmax{c}'], [f'cmax{c}'], axes=[3], keepdims=1))
        n.append(helper.make_node('Where', [f'colb{c}', 'col_idx', 'p999'], [f'colpmin{c}']))
        n.append(helper.make_node('ReduceMin', [f'colpmin{c}'], [f'cmin{c}'], axes=[3], keepdims=1))

        n.append(helper.make_node('Add', [f'rmax{c}', 'c1'], [f'rmax1_{c}']))
        n.append(helper.make_node('Add', [f'cmax{c}', 'c1'], [f'cmax1_{c}']))

        # ---- static-shape [1,1,30,30] symmetry check via mirrored gather ----
        # in-bbox mask (row in [rmin,rmax] and col in [cmin,cmax])
        n.append(helper.make_node('GreaterOrEqual', ['row_idx', f'rmin{c}'], [f'rge{c}']))
        n.append(helper.make_node('LessOrEqual', ['row_idx', f'rmax{c}'], [f'rle{c}']))
        n.append(helper.make_node('And', [f'rge{c}', f'rle{c}'], [f'rin{c}']))
        n.append(helper.make_node('GreaterOrEqual', ['col_idx', f'cmin{c}'], [f'cge{c}']))
        n.append(helper.make_node('LessOrEqual', ['col_idx', f'cmax{c}'], [f'cle{c}']))
        n.append(helper.make_node('And', [f'cge{c}', f'cle{c}'], [f'cin{c}']))
        n.append(helper.make_node('And', [f'rin{c}', f'cin{c}'], [f'inbbox{c}']))

        # mirror index (reflect columns about bbox center), clipped to valid range
        n.append(helper.make_node('Add', [f'cmin{c}', f'cmax{c}'], [f'csum{c}']))
        n.append(helper.make_node('Sub', [f'csum{c}', 'col_idx'], [f'refl{c}']))
        n.append(helper.make_node('Expand', [f'refl{c}', 'full_shape'], [f'reflx{c}']))
        n.append(helper.make_node('Clip', [f'reflx{c}', 'idx_min', 'idx_max'], [f'reflc{c}']))

        n.append(helper.make_node('GatherElements', [f'chan{c}', f'reflc{c}'], [f'mirr{c}'], axis=3))
        n.append(helper.make_node('Equal', [f'chan{c}', f'mirr{c}'], [f'eq{c}']))
        n.append(helper.make_node('Cast', [f'eq{c}'], [f'eqf{c}'], to=F))
        n.append(helper.make_node('Where', [f'inbbox{c}', f'eqf{c}', 'onef'], [f'masked{c}']))
        n.append(helper.make_node('ReduceMin', [f'masked{c}'], [f'mineq{c}'], axes=[0, 1, 2, 3], keepdims=1))
        n.append(helper.make_node('Greater', [f'mineq{c}', 'half'], [f'sym{c}']))

        n.append(helper.make_node('And', [f'sym{c}', f'present{c}'], [f'valid{c}']))
        n.append(helper.make_node('Cast', [f'valid{c}'], [f'w{c}'], to=F))

        n.append(helper.make_node('Cast', [f'rmin{c}'], [f'rmin{c}_f'], to=F))
        n.append(helper.make_node('Cast', [f'rmax1_{c}'], [f'rmax1{c}_f'], to=F))
        n.append(helper.make_node('Cast', [f'cmin{c}'], [f'cmin{c}_f'], to=F))
        n.append(helper.make_node('Cast', [f'cmax1_{c}'], [f'cmax1{c}_f'], to=F))

        n.append(helper.make_node('Mul', [f'w{c}', f'rmin{c}_f'], [f'wr0_{c}']))
        n.append(helper.make_node('Mul', [f'w{c}', f'rmax1{c}_f'], [f'wr1_{c}']))
        n.append(helper.make_node('Mul', [f'w{c}', f'cmin{c}_f'], [f'wc0_{c}']))
        n.append(helper.make_node('Mul', [f'w{c}', f'cmax1{c}_f'], [f'wc1_{c}']))

        weighted_r0.append(f'wr0_{c}')
        weighted_r1.append(f'wr1_{c}')
        weighted_c0.append(f'wc0_{c}')
        weighted_c1.append(f'wc1_{c}')

    def sum_chain(names, out):
        cur = names[0]
        for i, nm in enumerate(names[1:]):
            new = out if i == len(names) - 2 else f'{out}_acc{i}'
            n.append(helper.make_node('Add', [cur, nm], [new]))
            cur = new
        return cur

    sum_chain(weighted_r0, 'final_r0_f')
    sum_chain(weighted_r1, 'final_r1_f')
    sum_chain(weighted_c0, 'final_c0_f')
    sum_chain(weighted_c1, 'final_c1_f')

    n.append(helper.make_node('Cast', ['final_r0_f'], ['final_r0_i'], to=I64))
    n.append(helper.make_node('Cast', ['final_r1_f'], ['final_r1_i'], to=I64))
    n.append(helper.make_node('Cast', ['final_c0_f'], ['final_c0_i'], to=I64))
    n.append(helper.make_node('Cast', ['final_c1_f'], ['final_c1_i'], to=I64))

    n.append(helper.make_node('Reshape', ['final_r0_i', 'shape1d'], ['fr0']))
    n.append(helper.make_node('Reshape', ['final_r1_i', 'shape1d'], ['fr1']))
    n.append(helper.make_node('Reshape', ['final_c0_i', 'shape1d'], ['fc0']))
    n.append(helper.make_node('Reshape', ['final_c1_i', 'shape1d'], ['fc1']))

    n.append(helper.make_node('Slice', ['input', 'fr0', 'fr1', 'ax2'], ['crop_y']))
    n.append(helper.make_node('Slice', ['crop_y', 'fc0', 'fc1', 'ax3'], ['output']))

    graph = helper.make_graph(n, 'task174', [x], [y], I)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


def _make():
    return _crop_pad(build_174())


model = _bake(_make(), 174)
