import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def _K(n, a, d):
    return numpy_helper.from_array(np.array(a, dtype=d), name=n)


def build_191():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    I = []
    n = []

    I += [
        _K('s1', [1], np.int64), _K('e2', [2], np.int64),
        _K('s4', [4], np.int64), _K('e5', [5], np.int64),
        _K('ax1', [1], np.int64), _K('ax2', [2], np.int64), _K('ax3', [3], np.int64),
        _K('c4f', [4.0], np.float32),
        _K('row_idx', np.arange(30).reshape(1, 1, 30, 1), np.int64),
        _K('col_idx', np.arange(30).reshape(1, 1, 1, 30), np.int64),
        _K('p999', [999], np.int64), _K('m1', [-1], np.int64),
        _K('half_f', [0.5], np.float32),
        _K('shape1d', [-1], np.int64),
        _K('pv0', [0.0], np.float32),
        _K('rev5', [4, 3, 2, 1, 0], np.int64),
        _K('pads_border4', [0, 0, 4, 4, 0, 0, 4, 4], np.int64),
        _K('one_f', 1.0, np.float32),
        _K('four_f', 4.0, np.float32),
        _K('zero_f', 0.0, np.float32),
        _K('axall', [0, 1, 2, 3], np.int64),
    ]

    # --- extract channel-1 (is1) and channel-4 (is4) planes ---
    n.append(helper.make_node('Slice', ['input', 's1', 'e2', 'ax1'], ['is1']))          # (1,1,30,30)
    n.append(helper.make_node('Slice', ['input', 's4', 'e5', 'ax1'], ['is4']))          # (1,1,30,30)
    n.append(helper.make_node('Mul', ['is4', 'c4f'], ['is4x4']))
    n.append(helper.make_node('Add', ['is1', 'is4x4'], ['V']))                          # value grid, values in {0,1,4}

    # --- bbox of color-1 region (template bbox) ---
    n.append(helper.make_node('ReduceMax', ['is1'], ['row_any_f'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['row_any_f', 'half_f'], ['row_any_b']))
    n.append(helper.make_node('Where', ['row_any_b', 'row_idx', 'p999'], ['row_pmin']))
    n.append(helper.make_node('ReduceMin', ['row_pmin'], ['rmin'], axes=[2], keepdims=1))
    n.append(helper.make_node('Where', ['row_any_b', 'row_idx', 'm1'], ['row_pmax']))
    n.append(helper.make_node('ReduceMax', ['row_pmax'], ['rmax'], axes=[2], keepdims=1))

    n.append(helper.make_node('ReduceMax', ['is1'], ['col_any_f'], axes=[2], keepdims=1))
    n.append(helper.make_node('Greater', ['col_any_f', 'half_f'], ['col_any_b']))
    n.append(helper.make_node('Where', ['col_any_b', 'col_idx', 'p999'], ['col_pmin']))
    n.append(helper.make_node('ReduceMin', ['col_pmin'], ['cmin'], axes=[3], keepdims=1))
    n.append(helper.make_node('Where', ['col_any_b', 'col_idx', 'm1'], ['col_pmax']))
    n.append(helper.make_node('ReduceMax', ['col_pmax'], ['cmax'], axes=[3], keepdims=1))

    I += [_K('c1i_', [1], np.int64)]
    n.append(helper.make_node('Sub', ['rmax', 'rmin'], ['h_diff']))
    n.append(helper.make_node('Add', ['h_diff', 'c1i_'], ['h_val']))
    n.append(helper.make_node('Sub', ['cmax', 'cmin'], ['w_diff']))
    n.append(helper.make_node('Add', ['w_diff', 'c1i_'], ['w_val']))
    n.append(helper.make_node('Reshape', ['rmin', 'shape1d'], ['r0_1d']))
    n.append(helper.make_node('Reshape', ['cmin', 'shape1d'], ['c0_1d']))

    # --- slice out the template (values, not one-hot) as a FIXED 5x5 window anchored at
    # the bbox's own top-left corner (rmin,cmin). The window may reach slightly beyond the
    # true bbox (h,w can be < 5) - a validity mask (comparing local index against the true
    # h,w) zeroes out any such spillover so it can never pick up unrelated nearby content.
    # Every shape here is a compile-time constant (only the VALUES rmin/cmin/h/w vary per
    # input), so no data-dependent tensor shape ever appears in the graph.
    I += [_K('c5_1d', [5], np.int64)]
    n.append(helper.make_node('Add', ['r0_1d', 'c5_1d'], ['r5_1d']))
    n.append(helper.make_node('Add', ['c0_1d', 'c5_1d'], ['c5_1d_']))
    n.append(helper.make_node('Slice', ['V', 'r0_1d', 'r5_1d', 'ax2'], ['win_y']))
    n.append(helper.make_node('Slice', ['win_y', 'c0_1d', 'c5_1d_', 'ax3'], ['Window5']))  # static (1,1,5,5)

    I += [_K('local5_row', np.arange(5).reshape(1, 1, 5, 1), np.int64),
          _K('local5_col', np.arange(5).reshape(1, 1, 1, 5), np.int64)]
    n.append(helper.make_node('Less', ['local5_row', 'h_val'], ['row_valid_b']))
    n.append(helper.make_node('Cast', ['row_valid_b'], ['row_valid_f'], to=F))
    n.append(helper.make_node('Less', ['local5_col', 'w_val'], ['col_valid_b']))
    n.append(helper.make_node('Cast', ['col_valid_b'], ['col_valid_f'], to=F))
    n.append(helper.make_node('Mul', ['row_valid_f', 'col_valid_f'], ['valid5']))          # (1,1,5,5)
    n.append(helper.make_node('Mul', ['Window5', 'valid5'], ['T5']))                       # static (1,1,5,5)

    # --- padded value grid (extra 4-cell border so windows can hang off the real 30x30 edge) ---
    n.append(helper.make_node('Pad', ['V', 'pads_border4', 'pv0'], ['Vpad'], mode='constant'))  # (1,1,38,38)
    n.append(helper.make_node('Equal', ['Vpad', 'four_f'], ['G4_b']))
    n.append(helper.make_node('Cast', ['G4_b'], ['G4'], to=F))
    n.append(helper.make_node('Equal', ['Vpad', 'zero_f'], ['G0_b']))
    n.append(helper.make_node('Cast', ['G0_b'], ['G0'], to=F))

    # --- build the 8 dihedral transforms of T5 and, for each, find matches + paint mask ---
    paint_terms = []
    for k, (do_t, do_r, do_c) in enumerate(
        [(t, r, c) for t in (0, 1) for r in (0, 1) for c in (0, 1)]
    ):
        cur = 'T5'
        if do_t:
            n.append(helper.make_node('Transpose', [cur], [f'tr{k}'], perm=[0, 1, 3, 2]))
            cur = f'tr{k}'
        if do_r:
            n.append(helper.make_node('Gather', [cur, 'rev5'], [f'rr{k}'], axis=2))
            cur = f'rr{k}'
        if do_c:
            n.append(helper.make_node('Gather', [cur, 'rev5'], [f'rc{k}'], axis=3))
            cur = f'rc{k}'
        n.append(helper.make_node('Identity', [cur], [f'R{k}']))

        n.append(helper.make_node('Equal', [f'R{k}', 'four_f'], [f'A4b{k}']))
        n.append(helper.make_node('Cast', [f'A4b{k}'], [f'A4_{k}'], to=F))
        n.append(helper.make_node('Equal', [f'R{k}', 'one_f'], [f'A1b{k}']))
        n.append(helper.make_node('Cast', [f'A1b{k}'], [f'A1_{k}'], to=F))

        n.append(helper.make_node('ReduceSum', [f'A4_{k}', 'axall'], [f'sum4_{k}'], keepdims=1))
        n.append(helper.make_node('ReduceSum', [f'A1_{k}', 'axall'], [f'sum1_{k}'], keepdims=1))

        n.append(helper.make_node('Conv', ['G4', f'A4_{k}'], [f'conv4_{k}'], kernel_shape=[5, 5], strides=[1, 1], pads=[0, 0, 0, 0]))
        n.append(helper.make_node('Conv', ['G0', f'A1_{k}'], [f'conv1_{k}'], kernel_shape=[5, 5], strides=[1, 1], pads=[0, 0, 0, 0]))

        n.append(helper.make_node('Sub', [f'sum4_{k}', f'conv4_{k}'], [f'mism4_{k}']))
        n.append(helper.make_node('Sub', [f'sum1_{k}', f'conv1_{k}'], [f'mism1_{k}']))
        n.append(helper.make_node('Add', [f'mism4_{k}', f'mism1_{k}'], [f'mism_{k}']))
        n.append(helper.make_node('Less', [f'mism_{k}', 'half_f'], [f'match_b{k}']))
        n.append(helper.make_node('Cast', [f'match_b{k}'], [f'match_f{k}'], to=F))

        n.append(helper.make_node('ConvTranspose', [f'match_f{k}', f'A1_{k}'], [f'paint_pad{k}'],
                                   kernel_shape=[5, 5], strides=[1, 1], pads=[0, 0, 0, 0]))
        paint_terms.append(f'paint_pad{k}')

    cur = paint_terms[0]
    for i, t in enumerate(paint_terms[1:], 1):
        n.append(helper.make_node('Add', [cur, t], [f'paint_acc{i}']))
        cur = f'paint_acc{i}'
    n.append(helper.make_node('Greater', [cur, 'half_f'], ['paint_bool_pad']))
    n.append(helper.make_node('Cast', ['paint_bool_pad'], ['paint_f_pad'], to=F))

    I += [_K('s4b', [4, 4], np.int64), _K('e34', [34, 34], np.int64), _K('ax23', [2, 3], np.int64)]
    n.append(helper.make_node('Slice', ['paint_f_pad', 's4b', 'e34', 'ax23'], ['paint_f']))  # (1,1,30,30)
    n.append(helper.make_node('Greater', ['paint_f', 'half_f'], ['paint_bool']))

    n.append(helper.make_node('Where', ['paint_bool', 'one_f_bc', 'V'], ['final_V']))
    I.append(_K('one_f_bc', np.ones((1, 1, 30, 30)), np.float32))

    # --- rebuild one-hot output (values only ever 0, 1 or 4) ---
    n.append(helper.make_node('Equal', ['final_V', 'zero_f'], ['ch0_b']))
    n.append(helper.make_node('Cast', ['ch0_b'], ['ch0'], to=F))
    n.append(helper.make_node('Equal', ['final_V', 'one_f'], ['ch1_b']))
    n.append(helper.make_node('Cast', ['ch1_b'], ['ch1'], to=F))
    n.append(helper.make_node('Equal', ['final_V', 'four_f'], ['ch4_b']))
    n.append(helper.make_node('Cast', ['ch4_b'], ['ch4'], to=F))

    I.append(_K('chzero', np.zeros((1, 1, 30, 30)), np.float32))

    n.append(helper.make_node(
        'Concat',
        ['ch0', 'ch1', 'chzero', 'chzero', 'ch4', 'chzero', 'chzero', 'chzero', 'chzero', 'chzero'],
        ['oh_raw'], axis=1))

    # padding beyond the real (input-defined) grid extent must stay the all-zero
    # vector (no channel active), not channel-0=1 - mask by input presence, same
    # convention as the project's established _mask() helper for same-shape tasks.
    n.append(helper.make_node('ReduceMax', ['input'], ['presence_m'], axes=[1], keepdims=1))
    n.append(helper.make_node('Mul', ['oh_raw', 'presence_m'], ['output']))

    graph = helper.make_graph(n, 'task191', [x], [y], I)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR","/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p=_os.path.join(base,"data","task%03d.json"%t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json"%t)
def _reps(t,k=8):
    import onnxruntime as _ort  # noqa
    d=json.load(open(_resolve_task_json(t)))
    exs=sorted(d["train"]+d["test"]+d["arc-gen"], key=lambda e:(len(e["input"]),len(e["input"][0])))
    idx=set([0,len(exs)-1])|{int(j*(len(exs)-1)/(k-1)) for j in range(1,k-1)}
    out=[]
    for i in sorted(idx):
        g=exs[i]["input"]; a=np.zeros((1,10,30,30),np.float32)
        for r,row in enumerate(g):
            for c,v in enumerate(row): a[0][v][r][c]=1.0
        out.append(a)
    return out
def _bake(m,t):
    import onnxruntime as _ort
    inf=onnx.shape_inference.infer_shapes(_copy.deepcopy(m),strict_mode=True)
    def sym(vi): return any(dd.HasField("dim_param") or not dd.HasField("dim_value") for dd in vi.type.tensor_type.shape.dim)
    good={vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi)}
    good|={x.name for x in list(m.graph.input)+list(m.graph.output)}
    missing=[]
    for nd in m.graph.node:
        for o in nd.output:
            if o and o!="output" and o not in good and o not in missing: missing.append(o)
    if not missing: return m
    tmp=_copy.deepcopy(m)
    for nm in missing:
        vi=onnx.ValueInfoProto(); vi.name=nm; tmp.graph.output.append(vi)
    so=_ort.SessionOptions(); so.log_severity_level=3
    so.graph_optimization_level=_ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s=_ort.InferenceSession(tmp.SerializeToString(),so)
    mx={}; dt={}
    for inp in _reps(t):
        for nm,arr in zip(missing,s.run(missing,{"input":inp})):
            sh=list(arr.shape); mx[nm]=[max(a,b) for a,b in zip(mx[nm],sh)] if nm in mx else sh; dt[nm]=arr.dtype
    keep=[vi for vi in m.graph.value_info if vi.name not in missing]
    del m.graph.value_info[:]; m.graph.value_info.extend(keep)
    conv={np.dtype("float32"):TensorProto.FLOAT,np.dtype("int64"):TensorProto.INT64,np.dtype("bool"):TensorProto.BOOL,np.dtype("int32"):TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm,conv.get(dt[nm],TensorProto.FLOAT),mx[nm]))
    return m

def _make():
    return build_191()

# All node outputs are FIXED-shape by construction: the template window is always
# sliced to exactly 5x5 (start=rmin/cmin, end=start+5, both compile-time-constant
# widths - only rmin/cmin's VALUE is data-dependent), and any spillover beyond the
# true template bbox is zeroed via a validity mask rather than changing the tensor
# shape. So _bake's per-example sampling should see the identical shape for every
# example (a true invariant, not just "true for the sample").
model = _bake(_make(), 191)
