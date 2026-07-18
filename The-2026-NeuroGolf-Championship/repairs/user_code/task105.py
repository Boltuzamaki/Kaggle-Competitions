import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64
B = TensorProto.BOOL


def _K(n, a, d=np.int64):
    return numpy_helper.from_array(np.array(a, dtype=d), name=n)


def create_model():
    """task105 (4612dd53): find color-1 marker cells, draw the outline of their
    bounding box in color 2, then inside that box shoot full row- or full
    column-frontiers (whichever axis has FEWER total covered cells, per the
    literal DSL: branch(greater(size(vfrontier-union), size(hfrontier-union)),
    hfrontier-union, vfrontier-union)) through the remaining (non-border)
    color-1 cells, and underfill (only over background/0 cells) color 2 at the
    union of border + chosen-frontier back into the ORIGINAL grid.

    Verified byte-exact against every train/test/arc-gen example of task105
    via a literal step-by-step numpy translation of solve_4612dd53 AND this
    direct closed-form derivation (see scratchpad verify105_numpy.py) before
    writing this ONNX graph. No cropping/subgrid extraction is needed since
    output has the same H,W as input (underfill acts on the original grid) --
    everything below operates directly in the padded 30x30 canvas, and the
    computed candidate mask is provably confined to the real (non-padded)
    grid because it derives from the color-1 channel, which is all-zero in
    the padded region.
    """
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    # scatter indices: channel-0 slot always targets channel 0, channel-1 slot
    # always targets channel 2 -- a static [1,2,30,30] constant (cheap "params",
    # not "memory", since it's an initializer, not a computed node output).
    scatter_idx = np.zeros((1, 2, 30, 30), dtype=np.int64)
    scatter_idx[:, 1, :, :] = 2

    inits = [
        _K('c0', [0]), _K('c1', [1]), _K('c2', [2]), _K('c3', [3]), _K('c10', [10]),
        _K('ax1', [1]), _K('ax2', [2]), _K('ax3', [3]),
        _K('row_idx', np.arange(30).reshape(1, 1, 30, 1)),
        _K('col_idx', np.arange(30).reshape(1, 1, 1, 30)),
        _K('m1', [-1]), _K('p999', [999]),
        _K('half', [0.5], np.float32), _K('one_f', [1.0], np.float32),
        _K('scatter_idx', scatter_idx),
    ]

    n = []
    # ---- color-1 presence plane ----
    n.append(helper.make_node('Slice', ['input', 'c1', 'c2', 'ax1'], ['P_f']))
    n.append(helper.make_node('Greater', ['P_f', 'half'], ['P_b']))

    # ---- bounding box of color-1 cells (rmin,rmax,cmin,cmax) ----
    n.append(helper.make_node('ReduceMax', ['P_f'], ['row_any_f'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['row_any_f', 'half'], ['row_any_b']))
    n.append(helper.make_node('Where', ['row_any_b', 'row_idx', 'p999'], ['row_pmin']))
    n.append(helper.make_node('ReduceMin', ['row_pmin'], ['rmin'], axes=[2], keepdims=1))
    n.append(helper.make_node('Where', ['row_any_b', 'row_idx', 'm1'], ['row_pmax']))
    n.append(helper.make_node('ReduceMax', ['row_pmax'], ['rmax'], axes=[2], keepdims=1))

    n.append(helper.make_node('ReduceMax', ['P_f'], ['col_any_f'], axes=[2], keepdims=1))
    n.append(helper.make_node('Greater', ['col_any_f', 'half'], ['col_any_b']))
    n.append(helper.make_node('Where', ['col_any_b', 'col_idx', 'p999'], ['col_pmin']))
    n.append(helper.make_node('ReduceMin', ['col_pmin'], ['cmin'], axes=[3], keepdims=1))
    n.append(helper.make_node('Where', ['col_any_b', 'col_idx', 'm1'], ['col_pmax']))
    n.append(helper.make_node('ReduceMax', ['col_pmax'], ['cmax'], axes=[3], keepdims=1))

    # ---- row/col in-range & edge masks (border of the bbox) ----
    n.append(helper.make_node('GreaterOrEqual', ['row_idx', 'rmin'], ['row_ge']))
    n.append(helper.make_node('LessOrEqual', ['row_idx', 'rmax'], ['row_le']))
    n.append(helper.make_node('And', ['row_ge', 'row_le'], ['row_in_range']))
    n.append(helper.make_node('Equal', ['row_idx', 'rmin'], ['row_eq_min']))
    n.append(helper.make_node('Equal', ['row_idx', 'rmax'], ['row_eq_max']))
    n.append(helper.make_node('Or', ['row_eq_min', 'row_eq_max'], ['row_is_edge']))

    n.append(helper.make_node('GreaterOrEqual', ['col_idx', 'cmin'], ['col_ge']))
    n.append(helper.make_node('LessOrEqual', ['col_idx', 'cmax'], ['col_le']))
    n.append(helper.make_node('And', ['col_ge', 'col_le'], ['col_in_range']))
    n.append(helper.make_node('Equal', ['col_idx', 'cmin'], ['col_eq_min']))
    n.append(helper.make_node('Equal', ['col_idx', 'cmax'], ['col_eq_max']))
    n.append(helper.make_node('Or', ['col_eq_min', 'col_eq_max'], ['col_is_edge']))

    n.append(helper.make_node('And', ['row_is_edge', 'col_in_range'], ['border_h']))
    n.append(helper.make_node('And', ['col_is_edge', 'row_in_range'], ['border_v']))
    n.append(helper.make_node('Or', ['border_h', 'border_v'], ['border']))
    n.append(helper.make_node('Not', ['border'], ['not_border']))

    # ---- interior (non-border) color-1 cells -> which rows/cols carry them ----
    n.append(helper.make_node('And', ['P_b', 'not_border'], ['x5_b']))
    n.append(helper.make_node('Cast', ['x5_b'], ['x5_f'], to=F))
    n.append(helper.make_node('ReduceMax', ['x5_f'], ['col_has1_f'], axes=[2], keepdims=1))
    n.append(helper.make_node('Greater', ['col_has1_f', 'half'], ['col_has1_b']))
    n.append(helper.make_node('ReduceMax', ['x5_f'], ['row_has1_f'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['row_has1_f', 'half'], ['row_has1_b']))

    # ---- counts: x8 = n_cols*h4 (vertical total), x9 = n_rows*w4 (horizontal total) ----
    n.append(helper.make_node('Cast', ['col_has1_b'], ['col_has1_i'], to=I64))
    n.append(helper.make_node('ReduceSum', ['col_has1_i', 'ax3'], ['n_cols'], keepdims=1))
    n.append(helper.make_node('Cast', ['row_has1_b'], ['row_has1_i'], to=I64))
    n.append(helper.make_node('ReduceSum', ['row_has1_i', 'ax2'], ['n_rows'], keepdims=1))

    n.append(helper.make_node('Sub', ['rmax', 'rmin'], ['hspan']))
    n.append(helper.make_node('Add', ['hspan', 'c1'], ['h4']))
    n.append(helper.make_node('Sub', ['cmax', 'cmin'], ['wspan']))
    n.append(helper.make_node('Add', ['wspan', 'c1'], ['w4']))

    n.append(helper.make_node('Mul', ['n_cols', 'h4'], ['x8']))
    n.append(helper.make_node('Mul', ['n_rows', 'w4'], ['x9']))
    n.append(helper.make_node('Greater', ['x8', 'x9'], ['x10']))  # true -> pick horizontal (rows)

    # ---- frontier candidates (full column / full row within the bbox span) ----
    # NB: this onnxruntime build has no kernel for bool-typed Where (confirmed
    # experimentally), only for numeric types -- uint8 is the cheapest numeric
    # type with a working Where/Max kernel, so the x10 ? hfrontier : vfrontier
    # selection (and the border-OR-frontier merge) are done in uint8 (1 byte)
    # instead of float32 (4 bytes) to keep these [1,1,30,30] planes cheap.
    U8 = TensorProto.UINT8
    n.append(helper.make_node('And', ['col_has1_b', 'row_in_range'], ['vfrontier']))
    n.append(helper.make_node('And', ['row_has1_b', 'col_in_range'], ['hfrontier']))
    n.append(helper.make_node('Cast', ['vfrontier'], ['vfrontier_u8'], to=U8))
    n.append(helper.make_node('Cast', ['hfrontier'], ['hfrontier_u8'], to=U8))
    n.append(helper.make_node('Where', ['x10', 'hfrontier_u8', 'vfrontier_u8'], ['frontier_u8']))

    n.append(helper.make_node('Cast', ['border'], ['border_u8'], to=U8))
    n.append(helper.make_node('Max', ['border_u8', 'frontier_u8'], ['candidate_u8']))
    n.append(helper.make_node('Cast', ['candidate_u8'], ['cand_f'], to=F))

    # ---- underfill onto original grid: bg(=channel0) cells under `candidate` become color 2 ----
    # Only channels 0 and 2 ever change; channels 1,3..9 pass straight through.
    # Rather than materializing a large multi-channel pass-through slice, scatter
    # the two updated channels directly into a copy of `input` -- `output` is a
    # graph output and therefore exempt from the memory-cost accounting, so this
    # avoids ever paying for a full [1,10,30,30] (or [1,7,30,30]) intermediate.
    n.append(helper.make_node('Slice', ['input', 'c0', 'c1', 'ax1'], ['ch0']))
    n.append(helper.make_node('Slice', ['input', 'c2', 'c3', 'ax1'], ['ch2']))

    n.append(helper.make_node('Sub', ['one_f', 'cand_f'], ['one_minus_cand']))
    n.append(helper.make_node('Mul', ['ch0', 'one_minus_cand'], ['new_ch0']))
    n.append(helper.make_node('Mul', ['ch0', 'cand_f'], ['ch0_to_2']))
    n.append(helper.make_node('Add', ['ch2', 'ch0_to_2'], ['new_ch2']))

    n.append(helper.make_node('Concat', ['new_ch0', 'new_ch2'], ['updates'], axis=1))
    n.append(helper.make_node('ScatterElements', ['input', 'scatter_idx', 'updates'], ['output'], axis=1))

    graph = helper.make_graph(n, 'task105', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy


def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"),
                 r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship",
                 "."]:
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
                a[0][v][r][c] = 1.0
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
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64,
            np.dtype("bool"): TensorProto.BOOL, np.dtype("int32"): TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m


def _make():
    return create_model()


model = _bake(_make(), 105)
