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
def _mask(m):
    """Same-shape task: zero the polluted 30x30 border via an input-presence mask."""
    _rename_output(m,"oh_raw")
    m.graph.node.append(helper.make_node("ReduceMax",["input"],["presence_m"],axes=[1],keepdims=1))
    m.graph.node.append(helper.make_node("Mul",["oh_raw","presence_m"],["output"]))
    _set_out_shape(m,[1,10,30,30]); return m
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR","/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p=_os.path.join(base,"data","task%03d.json"%t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json"%t)
def _reps(t,k=8):
    d=json.load(open(_resolve_task_json(t)))
    exs=sorted(d["train"]+d["test"]+d["arc-gen"], key=lambda e:(len(e["input"]),len(e["input"][0])))
    idx=set([0,len(exs)-1]) | set(int(j*(len(exs)-1)/(k-1)) for j in range(1,k-1))
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
    good=set(vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi))
    good |= set(x.name for x in list(m.graph.input)+list(m.graph.output))
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

# ===== task069: ARC-DSL solve_321b1fc6, transcribed + verified against arc_dsl_ref =====
#
# Ground truth (arc_dsl_ref/solvers.py::solve_321b1fc6):
#   x1 = objects(I, F, F, T)        # univalued=False (any-color merge), diagonal=False (4-conn),
#                                     # without_bg=True (bg excluded)
#   x2 = colorfilter(x1, EIGHT)      # the (pure color-8) "marker" objects
#   x3 = difference(x1, x2)          # everything else -- the "template" object(s)
#   x4 = first(x3)                   # the (single) template object
#   x5 = cover(I, x4)                # remove the template, replacing with background
#   x6 = normalize(x4)               # template shifted so its ulcorner is at the origin
#   x7 = lbind(shift, x6)
#   x8 = apply(ulcorner, x2)         # every marker object's upper-left corner
#   x9 = mapply(x7, x8)              # normalized template re-shifted to each marker's ulcorner
#   O  = paint(x5, x9)               # stamp all those copies onto the cleared grid
#
# Verified in pure numpy (BFS 4-connected flood fill, bg fixed = 0) against all 264
# train+test+arc-gen examples in data/task069.json: nfail=0. Also confirmed: EVERY example has
# EXACTLY ONE non-8 (template) object (no "first"-tie-break ambiguity ever needed), and
# bg=mostcolor(grid) == bg=0 in all 264 examples (fixed-bg=0, the ordinary ARC convention,
# matches exactly -- same finding as task048/task170).
#
# Object grouping / stamping strategy (Loop/Scan/NonZero-free):
#  - General (any-color) 4-connected foreground-gated max-propagation (same idiom as task048's
#    has2, but seeded at "is foreground AND not color 8" instead of "is color 2") tags every
#    foreground cell with whether its connected component contains a non-8 color -> template_mask.
#  - Since the data always has exactly one template object, its GLOBAL ulcorner (tr,tc) is found
#    directly via the row/col-presence idiom (task031-style ReduceMax/Where/ReduceMin) applied to
#    template_mask -- no per-object separation needed for the template.
#  - Marker (color-8) objects can be numerous and at arbitrary positions, so instead of an
#    explicit per-object loop, a SECOND propagation (min-propagate row index and, separately, min-
#    propagate col index, gated on "neighbor is color 8", 4-connected) gives every color-8 cell its
#    own component's ulcorner (comp_min_row, comp_min_col) as a per-pixel tensor -- this is the
#    per-pixel generalization of `apply(ulcorner, x2)` (row/col tracked independently, NOT the
#    flat-index-minimal cell, since a non-rectangular object's bounding-box corner need not be an
#    actual occupied cell).
#  - For every output cell (y,x), if it's part of a marker object, the DSL rule wants
#    template_color[y - comp_min_row + tr, x - comp_min_col + tc] (the template cell at the same
#    offset within its own object's bounding box, translated to sit at the template's own
#    coordinates) -- a per-pixel DYNAMIC lookup into the (per-example) template pattern, computed
#    with GatherND(template_color_2d, stacked_indices, batch_dims=0) instead of a Loop over
#    markers or per-instance dynamic Slice/patch extraction.
#  - Final value per cell: 0 (bg) if in the template's own object (cover), else the GatherND-ed
#    template color if in a marker object, else the original color. ArgMax->OneHot reconstructs
#    the one-hot output; the standard `_mask` presence multiply zeroes the >actual-grid padding.
#  - Max BFS eccentricity (from the flat-index-minimal root cell, over 4-connected foreground-
#    gated connectivity) measured across all 264 examples, for BOTH the general (non-8) objects
#    and the color-8-only objects, is 7 -> R_ITERS=8 gives a 1-iteration safety margin (same
#    margin convention as task048/task170).

R_ITERS = 8
SENT = 100000
OFFSETS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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
    addK('c0i64', [0], np.int64)
    addK('c8i64', [8], np.int64)
    addK('c29i64', [29], np.int64)
    addK('c0f', [0.0], np.float32)
    addK('c05f', [0.5], np.float32)
    addK('sent_i64', [SENT], np.int64)
    addK('pads_hw', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK('p999', [999], np.int64)
    addK('row_indices', np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK('col_indices', np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK('shape_30_30', [30, 30], np.int64)
    addK('shape_1_1_30_30', [1, 1, 30, 30], np.int64)
    addK('shape_1_30_30', [1, 30, 30], np.int64)
    addK('depth10', [10], np.int64)
    addK('oh_vals', [0.0, 1.0], np.float32)

    starts = []
    ends = []
    for k, (di, dj) in enumerate(OFFSETS4):
        s = addK(f'st{k}', [1 + di, 1 + dj], np.int64)
        e = addK(f'en{k}', [31 + di, 31 + dj], np.int64)
        starts.append(s)
        ends.append(e)

    def four_slices(padded_name, prefix):
        outs = []
        for k in range(4):
            oname = f'{prefix}_sl{k}'
            nn('Slice', [padded_name, starts[k], ends[k], 'ax23_const'], [oname])
            outs.append(oname)
        return outs

    addK('ax23_const', [2, 3], np.int64)

    # ---- color id & foreground / is-8 flags ----
    nn('ArgMax', ['input'], ['color_idx64'], axis=1, keepdims=1)
    nn('Equal', ['color_idx64', 'c0i64'], ['is_bg_bool'])
    nn('Not', ['is_bg_bool'], ['fg_bool'])
    nn('Equal', ['color_idx64', 'c8i64'], ['is8_bool'])
    nn('Not', ['is8_bool'], ['not_is8_bool'])
    nn('And', ['fg_bool', 'not_is8_bool'], ['non8fg_bool'])
    nn('Cast', ['non8fg_bool'], ['non8fg_f'], to=F)

    # ---- fixed (iteration-independent) "is 4-neighbor foreground" mask, for general merge ----
    nn('Cast', ['fg_bool'], ['fg_i64'], to=I64)
    nn('Pad', ['fg_i64', 'pads_hw', 'c0i64'], ['padded_fg_general'], mode='constant')
    shifted_fg = four_slices('padded_fg_general', 'nbrfgg')
    nbr_is_fg = [nn('Cast', [s], [f'nbr_is_fg_{k}'], to=onnx.TensorProto.BOOL) for k, s in enumerate(shifted_fg)]

    # ---- has-non-8 flag: max-propagate through general foreground-gated 4-connectivity ----
    has2 = 'non8fg_f'
    for it in range(R_ITERS):
        padded_has2 = nn('Pad', [has2, 'pads_hw', 'c0f'], [f'padded_has2_it{it}'], mode='constant')
        shifted_has2 = four_slices(padded_has2, f'has2_it{it}')
        running = has2
        for k in range(4):
            cand = nn('Where', [nbr_is_fg[k], shifted_has2[k], 'c0f'], [f'cand_it{it}_{k}'])
            running = nn('Max', [running, cand], [f'max_it{it}_{k}'])
        has2 = running

    nn('Greater', [has2, 'c05f'], ['has_non8_bool'])
    nn('And', ['fg_bool', 'has_non8_bool'], ['template_mask_bool'])
    nn('Cast', ['template_mask_bool'], ['template_mask_f'], to=F)
    nn('Not', ['template_mask_bool'], ['not_template_bool'])
    nn('And', ['is8_bool', 'not_template_bool'], ['marker_mask_bool'])

    # ---- template's global ulcorner (tr,tc) -- exactly one template object in this task ----
    row_any = nn('ReduceMax', ['template_mask_f'], ['row_any'], axes=[3], keepdims=1)
    row_any_b = nn('Greater', [row_any, 'c0f'], ['row_any_b'])
    row_where = nn('Where', [row_any_b, 'row_indices', 'p999'], ['row_where'])
    tr_scalar = nn('ReduceMin', ['row_where'], ['tr_scalar'], axes=[2], keepdims=1)
    col_any = nn('ReduceMax', ['template_mask_f'], ['col_any'], axes=[2], keepdims=1)
    col_any_b = nn('Greater', [col_any, 'c0f'], ['col_any_b'])
    col_where = nn('Where', [col_any_b, 'col_indices', 'p999'], ['col_where'])
    tc_scalar = nn('ReduceMin', ['col_where'], ['tc_scalar'], axes=[3], keepdims=1)

    # ---- is8-gated 4-connectivity mask, for per-marker-object ulcorner propagation ----
    nn('Cast', ['is8_bool'], ['is8_i64'], to=I64)
    nn('Pad', ['is8_i64', 'pads_hw', 'c0i64'], ['padded_is8'], mode='constant')
    shifted_is8 = four_slices('padded_is8', 'nbris8')
    nbr_is_is8 = [nn('Cast', [s], [f'nbr_is_is8_{k}'], to=onnx.TensorProto.BOOL) for k, s in enumerate(shifted_is8)]

    row_prop = nn('Where', ['is8_bool', 'row_indices', 'sent_i64'], ['row_seed'])
    col_prop = nn('Where', ['is8_bool', 'col_indices', 'sent_i64'], ['col_seed'])
    for it in range(R_ITERS):
        padded_row = nn('Pad', [row_prop, 'pads_hw', 'sent_i64'], [f'padded_row_it{it}'], mode='constant')
        shifted_row = four_slices(padded_row, f'rowp_it{it}')
        running_row = row_prop
        for k in range(4):
            cand = nn('Where', [nbr_is_is8[k], shifted_row[k], 'sent_i64'], [f'candr_it{it}_{k}'])
            running_row = nn('Min', [running_row, cand], [f'minr_it{it}_{k}'])
        row_prop = running_row

        padded_col = nn('Pad', [col_prop, 'pads_hw', 'sent_i64'], [f'padded_col_it{it}'], mode='constant')
        shifted_col = four_slices(padded_col, f'colp_it{it}')
        running_col = col_prop
        for k in range(4):
            cand2 = nn('Where', [nbr_is_is8[k], shifted_col[k], 'sent_i64'], [f'candc_it{it}_{k}'])
            running_col = nn('Min', [running_col, cand2], [f'minc_it{it}_{k}'])
        col_prop = running_col

    comp_min_row = row_prop
    comp_min_col = col_prop

    # ---- target lookup coordinates into the template's own (global) coordinate frame ----
    tmp_r = nn('Sub', ['row_indices', comp_min_row], ['tmp_r'])
    target_row = nn('Add', [tmp_r, tr_scalar], ['target_row'])
    tmp_c = nn('Sub', ['col_indices', comp_min_col], ['tmp_c'])
    target_col = nn('Add', [tmp_c, tc_scalar], ['target_col'])

    target_row_c = nn('Clip', [target_row, 'c0i64', 'c29i64'], ['target_row_c'])
    target_col_c = nn('Clip', [target_col, 'c0i64', 'c29i64'], ['target_col_c'])

    tr_flat = nn('Reshape', [target_row_c, 'shape_30_30'], ['tr_flat'])
    tc_flat = nn('Reshape', [target_col_c, 'shape_30_30'], ['tc_flat'])
    tr_u = nn('Unsqueeze', [tr_flat], ['tr_u'], axes=[2])
    tc_u = nn('Unsqueeze', [tc_flat], ['tc_u'], axes=[2])
    gather_idx = nn('Concat', [tr_u, tc_u], ['gather_idx'], axis=2)

    # ---- template's own color pattern (2D, background elsewhere) ----
    template_color_full = nn('Where', ['template_mask_bool', 'color_idx64', 'c0i64'], ['template_color_full'])
    template_color_2d = nn('Reshape', [template_color_full, 'shape_30_30'], ['template_color_2d'])

    gathered = nn('GatherND', [template_color_2d, gather_idx], ['gathered'], batch_dims=0)
    gathered_r = nn('Reshape', [gathered, 'shape_1_1_30_30'], ['gathered_r'])

    inner = nn('Where', ['marker_mask_bool', gathered_r, 'color_idx64'], ['inner'])
    final_color = nn('Where', ['template_mask_bool', 'c0i64', inner], ['final_color'])

    pred_idx = nn('Reshape', [final_color, 'shape_1_30_30'], ['pred_idx'])
    nn('OneHot', [pred_idx, 'depth10', 'oh_vals'], ['output'], axis=1)

    graph = helper.make_graph(nodes, 'task069', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


def _make():
    return _mask(_build_core())


model = _bake(_make(), 69)
