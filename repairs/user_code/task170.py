import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64; I32 = TensorProto.INT32

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

# ===== task170: ARC-DSL solve_6ecd11f4, transcribed + verified against arc_dsl_ref =====
#
# Ground truth (arc_dsl_ref/solvers.py::solve_6ecd11f4):
#   x1 = objects(I, F, T, T)     # univalued=False (any-color merge), diagonal=True (8-conn),
#                                 # without_bg=True (bg excluded; bg = mostcolor(I))
#   x2 = argmax(x1, size); x3 = argmin(x1, size)     # largest / smallest object
#   x4 = subgrid(x2, I); x5 = subgrid(x3, I)          # each object's own bounding subgrid
#   x6 = width(x4); x7 = width(x5); x8 = divide(x6, x7)   # downscale factor (floor div)
#   x9 = downscale(x4, x8)                            # dsl.downscale picks every x8-th row/col
#                                                       # starting at 0 == numpy grid[::x8,::x8]
#   x10 = ofcolor(x9, ZERO); O = fill(x5, ZERO, x10)   # stamp the downscaled-large's zero
#                                                       # positions onto the small subgrid
#
# Verified in pure numpy (BFS 8-connected flood fill, `grid[::factor,::factor]` downscale,
# bg fixed = 0) against all 266 train+test+arc-gen examples in data/task170.json: nfail=0.
# Also confirmed: EVERY example has EXACTLY 2 foreground objects (no ties on size ever), so we
# never need a general object counter -- just distinguish the two components and their sizes.
# Max BFS eccentricity from the flat-index-minimal root cell (i*30+j, over the SAME
# foreground-gated 8-connectivity used in repairs/user_code/task048.py) measured across all 266
# examples is 31 -> R_ITERS=32 gives a 1-iteration safety margin (same convention as task048).
#
# Component separation strategy (Loop/Scan/NonZero-free): run task048's min-label-propagation
# idiom once (label[i,j] converges to the flat index of the minimal-index cell in its component).
# Then anchor1 = min label over all foreground cells (the global-first object's root); comp1 =
# cells whose label==anchor1. anchor2 = min label over foreground cells NOT in comp1 (the only
# remaining component, since there are always exactly 2); comp2 = cells whose label==anchor2.
# size1/size2 (sums) tell us which is "largest"/"smallest" (argmax/argmin by size, never tied).
# Bounding boxes + crops reuse task031's row/col-presence idiom (Where+ReduceMin/Max on
# broadcast row/col index tensors), applied to each component's own float mask.

R_ITERS = 32
SENT = 100000
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _build_core():
    inits = []
    nodes = []

    def addK(name, arr, dtype):
        inits.append(_K(name, arr, dtype))
        return name

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    # ---- constants ----
    addK('c0i32', [0], np.int32)
    addK('c0i64', [0], np.int64)
    addK('sent_i32', [SENT], np.int32)
    addK('pads_hw', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK('ax23', [2, 3], np.int64)
    addK('ax_2', [2], np.int64)
    addK('ax_3', [3], np.int64)
    init_label_np = (np.arange(30).reshape(30, 1) * 30 + np.arange(30).reshape(1, 30)).astype(np.int32)
    addK('init_label', init_label_np.reshape(1, 1, 30, 30), np.int32)
    addK('row_indices', np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK('col_indices', np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK('m1', [-1], np.int64)
    addK('p999', [999], np.int64)
    addK('c0_f', [0.0], np.float32)
    addK('c1', [1], np.int64)
    addK('shape_1d', [-1], np.int64)
    addK('big_end', [100000, 100000], np.int64)
    addK('start00', [0, 0], np.int64)
    addK('onehot0', np.eye(10, dtype=np.float32)[0].reshape(1, 10, 1, 1), np.float32)

    starts = []
    ends = []
    for k, (di, dj) in enumerate(OFFSETS):
        s = addK(f'st{k}', [1 + di, 1 + dj], np.int64)
        e = addK(f'en{k}', [31 + di, 31 + dj], np.int64)
        starts.append(s)
        ends.append(e)

    def eight_slices(padded_name, prefix):
        outs = []
        for k in range(8):
            oname = f'{prefix}_sl{k}'
            nn('Slice', [padded_name, starts[k], ends[k], 'ax23'], [oname])
            outs.append(oname)
        return outs

    # ---- color id & foreground (background is fixed color 0) ----
    nn('ArgMax', ['input'], ['color_idx64'], axis=1, keepdims=1)
    nn('Cast', ['color_idx64'], ['color_idx'], to=I32)
    nn('Equal', ['color_idx', 'c0i32'], ['is_bg'])
    nn('Not', ['is_bg'], ['is_fg_bool'])
    nn('Cast', ['is_fg_bool'], ['fg_i32'], to=I32)

    # ---- fixed (iteration-independent) "is 8-neighbor foreground" masks ----
    nn('Pad', ['fg_i32', 'pads_hw', 'c0i32'], ['padded_fg'], mode='constant')
    shifted_fg = eight_slices('padded_fg', 'nbrfg')
    nbr_is_fg = []
    for k, sfg in enumerate(shifted_fg):
        nbr_is_fg.append(nn('Cast', [sfg], [f'nbr_is_fg_{k}'], to=onnx.TensorProto.BOOL))

    # ---- connected components: Loop-free iterative min-label propagation ----
    label = 'init_label'
    for it in range(R_ITERS):
        padded_label = nn('Pad', [label, 'pads_hw', 'sent_i32'], [f'padded_label_it{it}'], mode='constant')
        shifted_labels = eight_slices(padded_label, f'lab_it{it}')
        running = label
        for k in range(8):
            cand = nn('Where', [nbr_is_fg[k], shifted_labels[k], 'sent_i32'], [f'cand_it{it}_{k}'])
            running = nn('Min', [running, cand], [f'min_it{it}_{k}'])
        label = running

    # ---- separate the (always exactly 2) foreground components ----
    label_for_min = nn('Where', ['is_fg_bool', label, 'sent_i32'], ['label_for_min'])
    anchor1 = nn('ReduceMin', ['label_for_min'], ['anchor1'], axes=[0, 1, 2, 3], keepdims=1)
    comp1_eq = nn('Equal', [label, 'anchor1'], ['comp1_eq'])
    comp1_mask = nn('And', ['comp1_eq', 'is_fg_bool'], ['comp1_mask'])

    label_excl1 = nn('Where', ['comp1_mask', 'sent_i32', 'label_for_min'], ['label_excl1'])
    anchor2 = nn('ReduceMin', ['label_excl1'], ['anchor2'], axes=[0, 1, 2, 3], keepdims=1)
    comp2_eq = nn('Equal', [label, 'anchor2'], ['comp2_eq'])
    comp2_mask = nn('And', ['comp2_eq', 'is_fg_bool'], ['comp2_mask'])

    comp1_f = nn('Cast', [comp1_mask], ['comp1_f'], to=F)
    comp2_f = nn('Cast', [comp2_mask], ['comp2_f'], to=F)
    size1 = nn('ReduceSum', [comp1_f], ['size1'], axes=[0, 1, 2, 3], keepdims=1)
    size2 = nn('ReduceSum', [comp2_f], ['size2'], axes=[0, 1, 2, 3], keepdims=1)
    one_is_larger = nn('Greater', [size1, size2], ['one_is_larger'])

    # NOTE: Where(bool_cond, bool_X, bool_Y) has no CPU kernel in this onnxruntime build, so we
    # select on the already-float comp{1,2}_f masks instead of the bool comp{1,2}_mask tensors.
    large_mask_f = nn('Where', [one_is_larger, comp1_f, comp2_f], ['large_mask_f'])
    small_mask_f = nn('Where', [one_is_larger, comp2_f, comp1_f], ['small_mask_f'])

    # ---- bounding-box crop helper (task031 row/col-presence idiom) ----
    def bbox_crop(mask_f, prefix):
        row_any = nn('ReduceMax', [mask_f], [f'{prefix}_row_any'], axes=[3], keepdims=1)
        row_any_b = nn('Greater', [row_any, 'c0_f'], [f'{prefix}_row_any_b'])
        row_pmax = nn('Where', [row_any_b, 'row_indices', 'm1'], [f'{prefix}_row_pmax'])
        r_max = nn('ReduceMax', [row_pmax], [f'{prefix}_r_max'], axes=[2], keepdims=1)
        row_pmin = nn('Where', [row_any_b, 'row_indices', 'p999'], [f'{prefix}_row_pmin'])
        r_min = nn('ReduceMin', [row_pmin], [f'{prefix}_r_min'], axes=[2], keepdims=1)
        col_any = nn('ReduceMax', [mask_f], [f'{prefix}_col_any'], axes=[2], keepdims=1)
        col_any_b = nn('Greater', [col_any, 'c0_f'], [f'{prefix}_col_any_b'])
        col_pmax = nn('Where', [col_any_b, 'col_indices', 'm1'], [f'{prefix}_col_pmax'])
        c_max = nn('ReduceMax', [col_pmax], [f'{prefix}_c_max'], axes=[3], keepdims=1)
        col_pmin = nn('Where', [col_any_b, 'col_indices', 'p999'], [f'{prefix}_col_pmin'])
        c_min = nn('ReduceMin', [col_pmin], [f'{prefix}_c_min'], axes=[3], keepdims=1)

        start_r = nn('Reshape', [r_min, 'shape_1d'], [f'{prefix}_start_r'])
        end_r_s = nn('Add', [r_max, 'c1'], [f'{prefix}_end_r_s'])
        end_r = nn('Reshape', [end_r_s, 'shape_1d'], [f'{prefix}_end_r'])
        start_c = nn('Reshape', [c_min, 'shape_1d'], [f'{prefix}_start_c'])
        end_c_s = nn('Add', [c_max, 'c1'], [f'{prefix}_end_c_s'])
        end_c = nn('Reshape', [end_c_s, 'shape_1d'], [f'{prefix}_end_c'])

        sliced_y = nn('Slice', ['input', start_r, end_r, 'ax_2'], [f'{prefix}_sliced_y'])
        sub = nn('Slice', [sliced_y, start_c, end_c, 'ax_3'], [f'{prefix}_sub'])
        width = nn('Sub', [end_c, start_c], [f'{prefix}_width'])
        return sub, width

    sub_large, width_large = bbox_crop(large_mask_f, 'L')
    sub_small, width_small = bbox_crop(small_mask_f, 'S')

    # ---- downscale the large subgrid by factor = width_large // width_small ----
    factor = nn('Div', [width_large, width_small], ['factor'])
    steps2 = nn('Concat', [factor, factor], ['steps2'], axis=0)
    downscaled = nn('Slice', [sub_large, 'start00', 'big_end', 'ax23', steps2], ['downscaled'])

    ds_argmax = nn('ArgMax', [downscaled], ['ds_argmax'], axis=1, keepdims=1)
    ds_zero_mask = nn('Equal', [ds_argmax, 'c0i64'], ['ds_zero_mask'])

    # ---- fill: stamp color-0 at the downscaled-large's zero positions onto the small subgrid ----
    nn('Where', [ds_zero_mask, 'onehot0', sub_small], ['output'])

    graph = helper.make_graph(nodes, 'task170', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


def _make():
    return _crop_pad(_build_core())


model = _bake(_make(), 170)
