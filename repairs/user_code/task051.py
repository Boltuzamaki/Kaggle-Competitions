import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def build_051():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    I = [
        K('ax_1', [1]), K('c1', [1]), K('depth10', [10]),
        K('row_indices', np.arange(30).reshape(1, 1, 30, 1), dtype=np.int64),
        K('col_indices', np.arange(30).reshape(1, 1, 1, 30), dtype=np.int64),
        K('m1', [-1]), K('p999', [999]),
        K('c0_f', [0.0], dtype=np.float32), K('c1_f', [1.0], dtype=np.float32),
        K('c0', [0]), K('c2', [2]), K('c42', [42]), K('c10000', [10000]),
        K('axall', [0, 1, 2, 3]),
        K('shape1', [-1]), K('shape1111', [1, 1, 1, 1]), K('shape111', [1, 1, 1]),
        K('oh_vals', [0.0, 1.0], dtype=np.float32),
    ]
    for c in range(1, 10):
        I.append(K(f'col{c}', [c]))

    n = []
    n.append(helper.make_node('ArgMax', ['input'], ['am'], axis=1, keepdims=1))

    adjs = []
    for c in range(1, 10):
        n.append(helper.make_node('Equal', ['am', f'col{c}'], [f'eqm{c}']))
        n.append(helper.make_node('Cast', [f'eqm{c}'], [f'eqmi{c}'], to=I64))
        n.append(helper.make_node('ReduceSum', [f'eqmi{c}', 'axall'], [f'cnt{c}'], keepdims=1))
        n.append(helper.make_node('Equal', [f'cnt{c}', 'c0'], [f'z{c}']))
        n.append(helper.make_node('Where', [f'z{c}', 'c10000', f'cnt{c}'], [f'adj{c}']))
        n.append(helper.make_node('Reshape', [f'adj{c}', 'shape1'], [f'adj1_{c}']))
        adjs.append(f'adj1_{c}')
    n.append(helper.make_node('Concat', adjs, ['stacked'], axis=0))
    n.append(helper.make_node('ArgMin', ['stacked'], ['idx'], axis=0, keepdims=1))
    n.append(helper.make_node('Add', ['idx', 'c1'], ['tgt']))
    n.append(helper.make_node('Reshape', ['tgt', 'shape1111'], ['tgt4']))
    n.append(helper.make_node('Equal', ['am', 'tgt4'], ['tmb']))
    n.append(helper.make_node('Cast', ['tmb'], ['tmf'], to=F))

    # bbox of rare-color mask -> x4 (row,col center)
    n.append(helper.make_node('ReduceMax', ['tmf'], ['row_any_f'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['row_any_f', 'c0_f'], ['row_any_b']))
    n.append(helper.make_node('Where', ['row_any_b', 'row_indices', 'p999'], ['row_pmin']))
    n.append(helper.make_node('ReduceMin', ['row_pmin'], ['r_up'], axes=[2], keepdims=1))
    n.append(helper.make_node('Where', ['row_any_b', 'row_indices', 'm1'], ['row_pmax']))
    n.append(helper.make_node('ReduceMax', ['row_pmax'], ['r_lo'], axes=[2], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['tmf'], ['col_any_f'], axes=[2], keepdims=1))
    n.append(helper.make_node('Greater', ['col_any_f', 'c0_f'], ['col_any_b']))
    n.append(helper.make_node('Where', ['col_any_b', 'col_indices', 'p999'], ['col_pmin']))
    n.append(helper.make_node('ReduceMin', ['col_pmin'], ['c_le'], axes=[3], keepdims=1))
    n.append(helper.make_node('Where', ['col_any_b', 'col_indices', 'm1'], ['col_pmax']))
    n.append(helper.make_node('ReduceMax', ['col_pmax'], ['c_ri'], axes=[3], keepdims=1))
    n.append(helper.make_node('Sub', ['r_lo', 'r_up'], ['r_hm1']))
    n.append(helper.make_node('Add', ['r_hm1', 'c1'], ['r_h']))
    n.append(helper.make_node('Sub', ['c_ri', 'c_le'], ['c_wm1']))
    n.append(helper.make_node('Add', ['c_wm1', 'c1'], ['c_w']))
    n.append(helper.make_node('Div', ['r_h', 'c2'], ['half_rh']))
    n.append(helper.make_node('Div', ['c_w', 'c2'], ['half_cw']))
    n.append(helper.make_node('Add', ['r_up', 'half_rh'], ['x4_row']))
    n.append(helper.make_node('Add', ['c_le', 'half_cw'], ['x4_col']))

    # bbox of all non-background (colors 1..9) -> x6 (row,col center)
    n.append(helper.make_node('Slice', ['input', 'c1', 'depth10', 'ax_1'], ['is_not_0']))
    n.append(helper.make_node('ReduceMax', ['is_not_0'], ['is_any_color'], axes=[1], keepdims=1))
    n.append(helper.make_node('Greater', ['is_any_color', 'c0_f'], ['is_any_bool']))
    n.append(helper.make_node('Cast', ['is_any_bool'], ['is_any_float'], to=F))
    n.append(helper.make_node('ReduceMax', ['is_any_float'], ['row_any_f2'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['row_any_f2', 'c0_f'], ['row_any_b2']))
    n.append(helper.make_node('Where', ['row_any_b2', 'row_indices', 'm1'], ['row_pmax2']))
    n.append(helper.make_node('ReduceMax', ['row_pmax2'], ['a_lo'], axes=[2], keepdims=1))
    n.append(helper.make_node('Where', ['row_any_b2', 'row_indices', 'p999'], ['row_pmin2']))
    n.append(helper.make_node('ReduceMin', ['row_pmin2'], ['a_up'], axes=[2], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['is_any_float'], ['col_any_f2'], axes=[2], keepdims=1))
    n.append(helper.make_node('Greater', ['col_any_f2', 'c0_f'], ['col_any_b2']))
    n.append(helper.make_node('Where', ['col_any_b2', 'col_indices', 'm1'], ['col_pmax2']))
    n.append(helper.make_node('ReduceMax', ['col_pmax2'], ['a_ri'], axes=[3], keepdims=1))
    n.append(helper.make_node('Where', ['col_any_b2', 'col_indices', 'p999'], ['col_pmin2']))
    n.append(helper.make_node('ReduceMin', ['col_pmin2'], ['a_le'], axes=[3], keepdims=1))
    n.append(helper.make_node('Sub', ['a_lo', 'a_up'], ['a_hm1']))
    n.append(helper.make_node('Add', ['a_hm1', 'c1'], ['a_h']))
    n.append(helper.make_node('Sub', ['a_ri', 'a_le'], ['a_wm1']))
    n.append(helper.make_node('Add', ['a_wm1', 'c1'], ['a_w']))
    n.append(helper.make_node('Div', ['a_h', 'c2'], ['half_ah']))
    n.append(helper.make_node('Div', ['a_w', 'c2'], ['half_aw']))
    n.append(helper.make_node('Add', ['a_up', 'half_ah'], ['x6_row']))
    n.append(helper.make_node('Add', ['a_le', 'half_aw'], ['x6_col']))

    # direction & ray endpoints
    n.append(helper.make_node('Sub', ['x6_row', 'x4_row'], ['d0']))
    n.append(helper.make_node('Sub', ['x6_col', 'x4_col'], ['d1']))
    n.append(helper.make_node('Identity', ['x4_row'], ['ai']))
    n.append(helper.make_node('Identity', ['x4_col'], ['aj']))
    n.append(helper.make_node('Mul', ['c42', 'd0'], ['step0']))
    n.append(helper.make_node('Mul', ['c42', 'd1'], ['step1']))
    n.append(helper.make_node('Add', ['ai', 'step0'], ['bi']))
    n.append(helper.make_node('Add', ['aj', 'step1'], ['bj']))
    n.append(helper.make_node('Min', ['ai', 'bi'], ['si']))
    n.append(helper.make_node('Max', ['ai', 'bi'], ['maxi']))
    n.append(helper.make_node('Add', ['maxi', 'c1'], ['ei']))
    n.append(helper.make_node('Min', ['aj', 'bj'], ['sj']))
    n.append(helper.make_node('Max', ['aj', 'bj'], ['maxj']))
    n.append(helper.make_node('Add', ['maxj', 'c1'], ['ej']))

    # branch selection (mirrors DSL connect(): horiz/vert/diag-pos/diag-neg)
    n.append(helper.make_node('Equal', ['d0', 'c0'], ['is_horiz']))
    n.append(helper.make_node('Equal', ['d1', 'c0'], ['d1_zero']))
    n.append(helper.make_node('Not', ['is_horiz'], ['not_horiz']))
    n.append(helper.make_node('And', ['not_horiz', 'd1_zero'], ['is_vert']))
    n.append(helper.make_node('Not', ['d1_zero'], ['not_d1zero']))
    n.append(helper.make_node('And', ['not_horiz', 'not_d1zero'], ['diag_ok']))
    n.append(helper.make_node('Equal', ['d0', 'd1'], ['d0_eq_d1']))
    n.append(helper.make_node('Neg', ['d1'], ['neg_d1']))
    n.append(helper.make_node('Equal', ['d0', 'neg_d1'], ['d0_eq_negd1']))
    n.append(helper.make_node('And', ['diag_ok', 'd0_eq_d1'], ['is_diagpos']))
    n.append(helper.make_node('And', ['diag_ok', 'd0_eq_negd1'], ['is_diagneg']))

    # per-cell line masks
    n.append(helper.make_node('Equal', ['row_indices', 'ai'], ['row_eq_ai']))
    n.append(helper.make_node('GreaterOrEqual', ['col_indices', 'sj'], ['col_ge_sj']))
    n.append(helper.make_node('Less', ['col_indices', 'ej'], ['col_lt_ej']))
    n.append(helper.make_node('And', ['col_ge_sj', 'col_lt_ej'], ['col_in_h']))
    n.append(helper.make_node('And', ['row_eq_ai', 'col_in_h'], ['line_horiz']))

    n.append(helper.make_node('Equal', ['col_indices', 'aj'], ['col_eq_aj']))
    n.append(helper.make_node('GreaterOrEqual', ['row_indices', 'si'], ['row_ge_si']))
    n.append(helper.make_node('Less', ['row_indices', 'ei'], ['row_lt_ei']))
    n.append(helper.make_node('And', ['row_ge_si', 'row_lt_ei'], ['row_in_range']))
    n.append(helper.make_node('And', ['col_eq_aj', 'row_in_range'], ['line_vert']))

    n.append(helper.make_node('Sub', ['row_indices', 'col_indices'], ['diff_rc']))
    n.append(helper.make_node('Sub', ['si', 'sj'], ['si_m_sj']))
    n.append(helper.make_node('Equal', ['diff_rc', 'si_m_sj'], ['diagpos_eq']))
    n.append(helper.make_node('And', ['diagpos_eq', 'row_in_range'], ['line_diagpos']))

    n.append(helper.make_node('Add', ['row_indices', 'col_indices'], ['sum_rc']))
    n.append(helper.make_node('Sub', ['ej', 'c1'], ['ejm1']))
    n.append(helper.make_node('Add', ['si', 'ejm1'], ['si_p_ejm1']))
    n.append(helper.make_node('Equal', ['sum_rc', 'si_p_ejm1'], ['diagneg_eq']))
    n.append(helper.make_node('And', ['diagneg_eq', 'row_in_range'], ['line_diagneg']))

    n.append(helper.make_node('And', ['is_horiz', 'line_horiz'], ['f_horiz']))
    n.append(helper.make_node('And', ['is_vert', 'line_vert'], ['f_vert']))
    n.append(helper.make_node('And', ['is_diagpos', 'line_diagpos'], ['f_diagpos']))
    n.append(helper.make_node('And', ['is_diagneg', 'line_diagneg'], ['f_diagneg']))
    n.append(helper.make_node('Or', ['f_horiz', 'f_vert'], ['or1']))
    n.append(helper.make_node('Or', ['f_diagpos', 'f_diagneg'], ['or2']))
    n.append(helper.make_node('Or', ['or1', 'or2'], ['final_mask_raw']))

    # true-grid bound (padding cells are all-zero across every channel)
    n.append(helper.make_node('ReduceMax', ['input'], ['presence_all'], axes=[1], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['presence_all'], ['row_any_rf'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['row_any_rf', 'c0_f'], ['row_any_rb']))
    n.append(helper.make_node('Where', ['row_any_rb', 'row_indices', 'm1'], ['row_pmax_r']))
    n.append(helper.make_node('ReduceMax', ['row_pmax_r'], ['r_max_real'], axes=[2], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['presence_all'], ['col_any_rf'], axes=[2], keepdims=1))
    n.append(helper.make_node('Greater', ['col_any_rf', 'c0_f'], ['col_any_rb']))
    n.append(helper.make_node('Where', ['col_any_rb', 'col_indices', 'm1'], ['col_pmax_r']))
    n.append(helper.make_node('ReduceMax', ['col_pmax_r'], ['c_max_real'], axes=[3], keepdims=1))
    n.append(helper.make_node('LessOrEqual', ['row_indices', 'r_max_real'], ['bound_row']))
    n.append(helper.make_node('LessOrEqual', ['col_indices', 'c_max_real'], ['bound_col']))
    n.append(helper.make_node('And', ['bound_row', 'bound_col'], ['bound_ok']))
    n.append(helper.make_node('And', ['final_mask_raw', 'bound_ok'], ['final_mask']))

    n.append(helper.make_node('Equal', ['am', 'c0'], ['bg_bool']))
    n.append(helper.make_node('And', ['final_mask', 'bg_bool'], ['fill_bool']))
    n.append(helper.make_node('Cast', ['fill_bool'], ['fill_float'], to=F))

    n.append(helper.make_node('Reshape', ['tgt', 'shape111'], ['tgt3']))
    n.append(helper.make_node('OneHot', ['tgt3', 'depth10', 'oh_vals'], ['rare_onehot'], axis=1))

    n.append(helper.make_node('Sub', ['c1_f', 'fill_float'], ['one_minus_fill']))
    n.append(helper.make_node('Mul', ['input', 'one_minus_fill'], ['term1']))
    n.append(helper.make_node('Mul', ['rare_onehot', 'fill_float'], ['term2']))
    n.append(helper.make_node('Add', ['term1', 'term2'], ['output']))

    graph = helper.make_graph(n, 'task051', [x], [y], I)
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
    return build_051()

model = _bake(_make(), 51)
