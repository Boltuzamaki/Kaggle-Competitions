import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64
BOOL = TensorProto.BOOL

# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n,a,d): return numpy_helper.from_array(np.array(a,dtype=d),name=n)
def _rename_output(m,new):
    for nd in m.graph.node:
        for i,o in enumerate(nd.output):
            if o=="output": nd.output[i]=new; return
def _set_out_shape(m,dims):
    tt=m.graph.output[0].type.tensor_type; tt.elem_type=TensorProto.FLOAT; del tt.shape.dim[:]
    for d in dims: tt.shape.dim.add().dim_value=d
def _crop_pad(m):
    """OneHot 'output' is a dynamic [1,10,h,w] crop at top-left; Pad to static 30x30
    using h,w read from the tensor's own Shape (keeps padding all-zero)."""
    _rename_output(m,"oh_raw")
    m.graph.initializer.extend([_K("__s2",[2],np.int64),_K("__e4",[4],np.int64),_K("__a0",[0],np.int64),
        _K("__30x2",[30,30],np.int64),_K("__pfx6",[0,0,0,0,0,0],np.int64),_K("__pv",[0.0],np.float32)])
    m.graph.node.extend([
        helper.make_node("Shape",["oh_raw"],["__osh"]),
        helper.make_node("Slice",["__osh","__s2","__e4","__a0"],["__hw"]),
        helper.make_node("Sub",["__30x2","__hw"],["__padhw"]),
        helper.make_node("Concat",["__pfx6","__padhw"],["__pads"],axis=0),
        helper.make_node("Pad",["oh_raw","__pads","__pv"],["output"],mode="constant")])
    _set_out_shape(m,[1,10,30,30]); return m
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
# ===== end scaffolding =====

K_ITERS = 16  # flood-fill iterations; empirically max needed = 8 across all train+test+arc-gen


def build_138():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 'H_out', 'W_out'])

    I = []
    def addK(name, arr, dtype=np.int64):
        I.append(_K(name, arr, dtype))

    n = []
    def N(*args, **kw): n.append(helper.make_node(*args, **kw))

    addK('ax_0', [0]); addK('ax_1', [1]); addK('ax_2', [2]); addK('ax_3', [3])
    addK('row_indices', np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK('col_indices', np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK('row_indices_f', np.arange(30).reshape(1, 1, 30, 1), np.float32)
    addK('col_indices_f', np.arange(30).reshape(1, 1, 1, 30), np.float32)
    addK('m1', [-1]); addK('p999', [999])
    addK('m1_f', [-1.0], np.float32); addK('p999_f', [999.0], np.float32)
    addK('c0_f', [0.0], np.float32); addK('c1_f', [1.0], np.float32); addK('half_f', [0.5], np.float32)
    addK('c1', [1]); addK('c0', [0]); addK('c2', [2])
    addK('shape_1d', [-1])
    addK('depth10', [10]); addK('oh_vals', [0.0, 1.0], np.float32)
    addK('cross_kernel', np.array([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=np.float32), np.float32)
    addK('s2', [2]); addK('e4', [4])
    addK('big_end', [9223372036854775807])
    addK('one_one', [1, 1])
    addK('pads_edge', [0, 0, 1, 1, 0, 0, 1, 1])
    addK('pv0', [0.0], np.float32)

    # ---- bounding box of actual (non-padded) content, via any-nonzero-color presence ----
    N('Slice', ['input', 'c1', 'depth10', 'ax_1'], ['is_not_0'])
    N('ReduceMax', ['is_not_0'], ['is_any_color'], axes=[1], keepdims=1)
    N('Greater', ['is_any_color', 'c0_f'], ['is_any_bool'])
    N('Cast', ['is_any_bool'], ['is_any_float'], to=F)

    N('ReduceMax', ['is_any_float'], ['row_any_float'], axes=[3], keepdims=1)
    N('Greater', ['row_any_float', 'c0_f'], ['row_any_bool'])
    N('Where', ['row_any_bool', 'row_indices', 'm1'], ['row_present'])
    N('ReduceMax', ['row_present'], ['r_max'], axes=[2], keepdims=1)
    N('Where', ['row_any_bool', 'row_indices', 'p999'], ['row_present_min'])
    N('ReduceMin', ['row_present_min'], ['r_min'], axes=[2], keepdims=1)

    N('ReduceMax', ['is_any_float'], ['col_any_float'], axes=[2], keepdims=1)
    N('Greater', ['col_any_float', 'c0_f'], ['col_any_bool'])
    N('Where', ['col_any_bool', 'col_indices', 'm1'], ['col_present'])
    N('ReduceMax', ['col_present'], ['c_max'], axes=[3], keepdims=1)
    N('Where', ['col_any_bool', 'col_indices', 'p999'], ['col_present_min'])
    N('ReduceMin', ['col_present_min'], ['c_min'], axes=[3], keepdims=1)

    # ---- valid mask V (True inside real grid extent) ----
    N('GreaterOrEqual', ['row_indices', 'r_min'], ['row_ge'])
    N('LessOrEqual', ['row_indices', 'r_max'], ['row_le'])
    N('And', ['row_ge', 'row_le'], ['row_valid'])
    N('GreaterOrEqual', ['col_indices', 'c_min'], ['col_ge'])
    N('LessOrEqual', ['col_indices', 'c_max'], ['col_le'])
    N('And', ['col_ge', 'col_le'], ['col_valid'])
    N('And', ['row_valid', 'col_valid'], ['V_bool'])
    N('Cast', ['V_bool'], ['V_f'], to=F)

    # ---- background mask within valid region ----
    N('Slice', ['input', 'c0', 'c1', 'ax_1'], ['ch0'])
    N('Mul', ['ch0', 'V_f'], ['bg'])

    # ---- border ring of the real grid extent ----
    N('Equal', ['row_indices', 'r_min'], ['row_eq_min'])
    N('Equal', ['row_indices', 'r_max'], ['row_eq_max'])
    N('Or', ['row_eq_min', 'row_eq_max'], ['row_edge'])
    N('Equal', ['col_indices', 'c_min'], ['col_eq_min'])
    N('Equal', ['col_indices', 'c_max'], ['col_eq_max'])
    N('Or', ['col_eq_min', 'col_eq_max'], ['col_edge'])
    N('Or', ['row_edge', 'col_edge'], ['edge_bool'])
    N('And', ['edge_bool', 'V_bool'], ['border_ring_bool'])
    N('Cast', ['border_ring_bool'], ['border_ring_f'], to=F)

    N('Mul', ['bg', 'border_ring_f'], ['seed'])

    # ---- flood fill (reachable-from-border background), K_ITERS unrolled ----
    prev = 'seed'
    for it in range(K_ITERS):
        conv_o = f'nb_conv_{it}'
        any_o = f'nb_any_{it}'
        anyf_o = f'nb_anyf_{it}'
        gated_o = f'nb_gated_{it}'
        new_o = f'reach_{it}'
        N('Conv', [prev, 'cross_kernel'], [conv_o], pads=[1, 1, 1, 1])
        N('Greater', [conv_o, 'c0_f'], [any_o])
        N('Cast', [any_o], [anyf_o], to=F)
        N('Mul', ['bg', anyf_o], [gated_o])
        N('Max', [prev, gated_o], [new_o])
        prev = new_o
    reachable_final = prev

    N('Sub', ['c1_f', reachable_final], ['not_reachable'])
    N('Mul', ['bg', 'not_reachable'], ['enclosed'])

    N('Conv', ['enclosed', 'cross_kernel'], ['enc_nb_conv'], pads=[1, 1, 1, 1])
    N('Greater', ['enc_nb_conv', 'c0_f'], ['enc_nb_any'])
    N('Cast', ['enc_nb_any'], ['enc_nb_anyf'], to=F)
    N('Mul', ['enclosed', 'enc_nb_anyf'], ['target_hole'])

    # ---- bbox of target_hole ----
    N('ReduceMax', ['target_hole'], ['hole_row_any_f'], axes=[3], keepdims=1)
    N('Greater', ['hole_row_any_f', 'half_f'], ['hole_row_any_bool'])
    N('Where', ['hole_row_any_bool', 'row_indices', 'm1'], ['hole_row_present'])
    N('ReduceMax', ['hole_row_present'], ['h_r_max'], axes=[2], keepdims=1)
    N('Where', ['hole_row_any_bool', 'row_indices', 'p999'], ['hole_row_present_min'])
    N('ReduceMin', ['hole_row_present_min'], ['h_r_min'], axes=[2], keepdims=1)

    N('ReduceMax', ['target_hole'], ['hole_col_any_f'], axes=[2], keepdims=1)
    N('Greater', ['hole_col_any_f', 'half_f'], ['hole_col_any_bool'])
    N('Where', ['hole_col_any_bool', 'col_indices', 'm1'], ['hole_col_present'])
    N('ReduceMax', ['hole_col_present'], ['h_c_max'], axes=[3], keepdims=1)
    N('Where', ['hole_col_any_bool', 'col_indices', 'p999'], ['hole_col_present_min'])
    N('ReduceMin', ['hole_col_present_min'], ['h_c_min'], axes=[3], keepdims=1)

    # outbox = bbox expanded by 1 (exclusive end for Slice)
    N('Sub', ['h_r_min', 'c1'], ['R0'])
    N('Add', ['h_r_max', 'c2'], ['R1'])
    N('Sub', ['h_c_min', 'c1'], ['C0'])
    N('Add', ['h_c_max', 'c2'], ['C1'])

    N('Reshape', ['R0', 'shape_1d'], ['R0_1d'])
    N('Reshape', ['R1', 'shape_1d'], ['R1_1d'])
    N('Reshape', ['C0', 'shape_1d'], ['C0_1d'])
    N('Reshape', ['C1', 'shape_1d'], ['C1_1d'])

    N('Slice', ['input', 'R0_1d', 'R1_1d', 'ax_2'], ['x7_y'])
    N('Slice', ['x7_y', 'C0_1d', 'C1_1d', 'ax_3'], ['x7'])  # (1,10,H7,W7) dynamic

    # ---- dynamic H7, W7 and not_edge mask (1 interior, 0 on crop's own border ring) ----
    N('Shape', ['x7'], ['x7_shape'])
    N('Slice', ['x7_shape', 's2', 'e4', 'ax_0'], ['hw7'])           # [H7,W7]
    N('Slice', ['hw7', 'ax_0', 'ax_1', 'ax_0'], ['H7_1d'])          # [H7]
    N('Slice', ['hw7', 'ax_1', 'c2', 'ax_0'], ['W7_1d'])            # [W7]

    N('Sub', ['hw7', 'one_one_2'], ['inner_hw'])
    addK('one_one_2', [2, 2])
    N('Concat', ['one_one', 'inner_hw'], ['inner_shape'], axis=0)
    N('ConstantOfShape', ['inner_shape'], ['inner_ones'],
      value=numpy_helper.from_array(np.array([1.0], dtype=np.float32)))
    N('Pad', ['inner_ones', 'pads_edge', 'pv0'], ['not_edge'], mode='constant')  # (1,1,H7,W7)

    N('Slice', ['col_indices_f', 'c0', 'W7_1d', 'ax_3'], ['col_idx7'])  # (1,1,1,W7)
    N('Slice', ['row_indices_f', 'c0', 'H7_1d', 'ax_2'], ['row_idx7'])  # (1,1,H7,1)

    fill_masks = {}
    for cc in range(1, 10):
        addK(f'cst_{cc}', [cc]); addK(f'cst_{cc}p1', [cc + 1])
        addK(f'cst_{cc}_f', [float(cc)], np.float32)
        N('Slice', ['x7', f'cst_{cc}', f'cst_{cc}p1', 'ax_1'], [f'mask_{cc}'])  # (1,1,H7,W7)

        N('ReduceMax', [f'mask_{cc}'], [f'row_present_{cc}'], axes=[3], keepdims=1)  # (1,1,H7,1)
        N('ReduceMax', [f'mask_{cc}'], [f'col_present_{cc}'], axes=[2], keepdims=1)  # (1,1,1,W7)
        N('ReduceSum', [f'row_present_{cc}', 'ax_2'], [f'n_rows_{cc}'], keepdims=1)
        N('ReduceSum', [f'col_present_{cc}', 'ax_3'], [f'n_cols_{cc}'], keepdims=1)
        N('Greater', [f'n_rows_{cc}', 'c1_f'], [f'rows_ge2_{cc}'])
        N('Greater', [f'n_cols_{cc}', 'c1_f'], [f'cols_ge2_{cc}'])
        N('And', [f'rows_ge2_{cc}', f'cols_ge2_{cc}'], [f'irregular_bool_{cc}'])
        N('Cast', [f'irregular_bool_{cc}'], [f'irregular_{cc}'], to=F)

        N('Mul', [f'mask_{cc}', 'not_edge'], [f'is_outlier_f_{cc}'])
        N('Greater', [f'is_outlier_f_{cc}', 'half_f'], [f'is_outlier_bool_{cc}'])

        N('Slice', [f'mask_{cc}', 'c0', 'c1', 'ax_3'], [f'left_border_{cc}'])       # (1,1,H7,1)
        N('Slice', [f'mask_{cc}', 'm1', 'big_end', 'ax_3'], [f'right_border_{cc}'])  # (1,1,H7,1)
        N('Slice', [f'mask_{cc}', 'c0', 'c1', 'ax_2'], [f'top_border_{cc}'])        # (1,1,1,W7)
        N('Slice', [f'mask_{cc}', 'm1', 'big_end', 'ax_2'], [f'bottom_border_{cc}'])  # (1,1,1,W7)
        N('Greater', [f'left_border_{cc}', 'half_f'], [f'left_border_bool_{cc}'])
        N('Greater', [f'right_border_{cc}', 'half_f'], [f'right_border_bool_{cc}'])
        N('Greater', [f'top_border_{cc}', 'half_f'], [f'top_border_bool_{cc}'])
        N('Greater', [f'bottom_border_{cc}', 'half_f'], [f'bottom_border_bool_{cc}'])

        N('Where', [f'is_outlier_bool_{cc}', 'col_idx7', 'm1_f'], [f'colw_or_m1_{cc}'])
        N('ReduceMax', [f'colw_or_m1_{cc}'], [f'maxJ_row_{cc}'], axes=[3], keepdims=1)
        N('Where', [f'is_outlier_bool_{cc}', 'col_idx7', 'p999_f'], [f'colw_or_p_{cc}'])
        N('ReduceMin', [f'colw_or_p_{cc}'], [f'minJ_row_{cc}'], axes=[3], keepdims=1)
        N('Where', [f'is_outlier_bool_{cc}', 'row_idx7', 'm1_f'], [f'roww_or_m1_{cc}'])
        N('ReduceMax', [f'roww_or_m1_{cc}'], [f'maxI_col_{cc}'], axes=[2], keepdims=1)
        N('Where', [f'is_outlier_bool_{cc}', 'row_idx7', 'p999_f'], [f'roww_or_p_{cc}'])
        N('ReduceMin', [f'roww_or_p_{cc}'], [f'minI_col_{cc}'], axes=[2], keepdims=1)

        N('LessOrEqual', ['col_idx7', f'maxJ_row_{cc}'], [f'col_le_maxJ_{cc}'])
        N('And', [f'left_border_bool_{cc}', f'col_le_maxJ_{cc}'], [f'left_fill_{cc}'])
        N('GreaterOrEqual', ['col_idx7', f'minJ_row_{cc}'], [f'col_ge_minJ_{cc}'])
        N('And', [f'right_border_bool_{cc}', f'col_ge_minJ_{cc}'], [f'right_fill_{cc}'])
        N('LessOrEqual', ['row_idx7', f'maxI_col_{cc}'], [f'row_le_maxI_{cc}'])
        N('And', [f'top_border_bool_{cc}', f'row_le_maxI_{cc}'], [f'top_fill_{cc}'])
        N('GreaterOrEqual', ['row_idx7', f'minI_col_{cc}'], [f'row_ge_minI_{cc}'])
        N('And', [f'bottom_border_bool_{cc}', f'row_ge_minI_{cc}'], [f'bottom_fill_{cc}'])

        N('Or', [f'left_fill_{cc}', f'right_fill_{cc}'], [f'lr_fill_{cc}'])
        N('Or', [f'top_fill_{cc}', f'bottom_fill_{cc}'], [f'tb_fill_{cc}'])
        N('Or', [f'lr_fill_{cc}', f'tb_fill_{cc}'], [f'fill_bool_{cc}'])
        N('Cast', [f'fill_bool_{cc}', ], [f'fill_f_{cc}'], to=F)
        N('Mul', [f'fill_f_{cc}', f'irregular_{cc}'], [f'fill_mask_{cc}'])
        fill_masks[cc] = f'fill_mask_{cc}'

    # total_fill = OR over colors; color_layer = SUM(cc * fill_mask_cc)
    total_fill = fill_masks[1]
    color_layer = None
    for cc in range(1, 10):
        fm = fill_masks[cc]
        if cc > 1:
            N('Max', [total_fill, fm], [f'total_fill_acc_{cc}'])
            total_fill = f'total_fill_acc_{cc}'
        N('Mul', [fm, f'cst_{cc}_f'], [f'weighted_{cc}'])
        if color_layer is None:
            color_layer = f'weighted_{cc}'
        else:
            N('Add', [color_layer, f'weighted_{cc}'], [f'color_layer_acc_{cc}'])
            color_layer = f'color_layer_acc_{cc}'

    N('Sub', ['c1_f', total_fill], ['not_total_fill'])

    out_channels = []
    N('Slice', ['x7', 'c0', 'c1', 'ax_1'], ['orig_0'])
    N('Mul', ['orig_0', 'not_total_fill'], ['new_0'])
    out_channels.append('new_0')
    for cc in range(1, 10):
        N('Slice', ['x7', f'cst_{cc}', f'cst_{cc}p1', 'ax_1'], [f'orig_{cc}'])
        N('Mul', [f'orig_{cc}', 'not_total_fill'], [f'kept_{cc}'])
        N('Max', [fill_masks[cc], f'kept_{cc}'], [f'new_{cc}'])
        out_channels.append(f'new_{cc}')

    N('Concat', out_channels, ['output'], axis=1)

    return helper.make_model(helper.make_graph(n, 'task138', [x], [y], I),
                              ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


def _make():
    return _crop_pad(build_138())


model = _bake(_make(), 138)
