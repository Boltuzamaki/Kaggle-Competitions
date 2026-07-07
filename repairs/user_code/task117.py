import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64; BOOL = TensorProto.BOOL

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

# ===== task117: ARC-DSL solve_4c5c2cf0, transcribed + verified against arc_dsl_ref =====
#
# Ground truth (arc_dsl_ref/solvers.py::solve_4c5c2cf0) mirrors a "wing" object across an "anchor"
# object's own center, in both the horizontal and vertical direction (and both combined), and
# paints the result. Verified in pure numpy against ALL 265 train+test+arc-gen examples:
#
#  - EVERY example has EXACTLY 2 monochromatic connected components, AND exactly 2 distinct
#    non-background colors present overall -- i.e. each present color forms exactly one component,
#    so per-COLOR analysis (bbox + rotation-symmetry test) is equivalent to the DSL's per-OBJECT
#    analysis. No connected-components / flood-fill needed at all (avoids Loop/Scan).
#  - Exactly ONE of the 2 present colors has a square bounding box whose own subgrid equals its
#    own 90-degree rotation (`extract`'s condition) in ALL 265 examples -- this "anchor" color is
#    always uniquely determined (no frozenset-hash-order ambiguity: the condition genuinely picks
#    out one color in every single case, confirmed by direct check against arc_dsl_ref).
#  - Rule: let A = anchor color (unique per-color bbox rmin/rmax/cmin/cmax with h==w and
#    subgrid==rot90(subgrid)); center c0,c1 = (rmin_A + h_A//2, cmin_A + w_A//2) (bbox-based
#    center, matching dsl.center on the object's own indices). Let B = the OTHER present color.
#    For every cell (i,j) of color B, additionally paint color B at (2*c0-i, j), (i, 2*c1-j), and
#    (2*c0-i, 2*c1-j) (dropping any that fall outside the grid) -- i.e. mirror B's cells across the
#    anchor's center row and center column (identity + hmirror + vmirror + both). Color A (and any
#    cell not reached by these mirrors) is left unchanged. This exactly reproduces the literal DSL
#    even where its "first(objects(...))" tie-breaks would otherwise be ambiguous (the 75/265
#    arc-gen cases where A and B are NOT diagonally touching, so `objects(I,F,T,T)` yields TWO
#    multicolor blobs instead of one and the literal DSL's `first(x2)` becomes an arbitrary,
#    unstable frozenset-order pick) -- the per-color reading sidesteps that entirely and gets
#    n_fail=0 on all 265 examples (vs 75 failures/exceptions for the literal object-based DSL).
#
# ONNX strategy (Loop/Scan/NonZero/Unique/Compress-free):
#  - For each of the 9 non-bg colors (python-unrolled, not a runtime loop): compute bbox via the
#    task031-style ReduceMax/Where/ReduceMin row&col-presence idiom; check square (h==w); check
#    rotational self-symmetry by expressing "subgrid==rot90(subgrid)" as a per-pixel condition
#    grid[I,J] == grid[rmax+cmin-J, cmin-rmin+I] for (I,J) in the bbox (derived algebraically from
#    the rot90 definition), evaluated via one GatherND per color (indices built from the row/col
#    index grids + the color's own rmin/rmax/cmin/cmax), masked to "true OR outside bbox", reduced
#    with ReduceMin over the whole grid.
#  - Weighted-sum (multiply-by-indicator-then-add) selection is used throughout to pick out the
#    unique anchor color's scalars (rmin/cmin/h/w/color) and the unique "other" color's per-pixel
#    mask + color id, since exactly one indicator is ever 1.
#  - The 3 mirrored copies of B's mask are each one GatherND lookup (indices = row/col reflected
#    about 2*c0/2*c1, clipped to [0,29], multiplied by an explicit in-bounds validity mask so
#    clipped-but-truly-out-of-range positions correctly contribute nothing).
#  - Final per-pixel color = colorB where any mirrored-B-mask fires, else the original ArgMax
#    color; OneHot reconstructs the one-hot output. Standard `_mask` zeroes the >actual-grid pad.

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

    addK('c0i64', [0], np.int64)
    addK('c1i64', [1], np.int64)
    addK('c2i64', [2], np.int64)
    addK('m1i64', [-1], np.int64)
    addK('p999', [999], np.int64)
    addK('c29i64', [29], np.int64)
    addK('c0f', [0.0], np.float32)
    addK('half', [0.5], np.float32)
    addK('ax1_const', [1], np.int64)
    addK('row_indices', np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK('col_indices', np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK('shape_30_30', [30, 30], np.int64)
    addK('shape_1_1_30_30', [1, 1, 30, 30], np.int64)
    addK('shape_1_30_30', [1, 30, 30], np.int64)
    addK('depth10', [10], np.int64)
    addK('oh_vals', [0.0, 1.0], np.float32)

    nn('ArgMax', ['input'], ['color_idx64'], axis=1, keepdims=1)
    grid_2d = nn('Reshape', ['color_idx64', 'shape_30_30'], ['grid_2d'])

    anchor_color_terms = []
    anchor_rmin_terms = []
    anchor_cmin_terms = []
    anchor_h_terms = []
    anchor_w_terms = []
    other_color_terms = []
    maskB_terms = []

    for c in range(1, 10):
        cst = addK(f'cst{c}', [c], np.int64)
        cen = addK(f'cen{c}', [c + 1], np.int64)
        cval = addK(f'cval{c}', [c], np.int64)

        ch = nn('Slice', ['input', cst, cen, 'ax1_const'], [f'ch{c}'])

        row_any = nn('ReduceMax', [ch], [f'rowany{c}'], axes=[3], keepdims=1)
        row_any_b = nn('Greater', [row_any, 'c0f'], [f'rowanyb{c}'])
        row_wmin = nn('Where', [row_any_b, 'row_indices', 'p999'], [f'rowwmin{c}'])
        rmin_c = nn('ReduceMin', [row_wmin], [f'rmin{c}'], axes=[2], keepdims=1)
        row_wmax = nn('Where', [row_any_b, 'row_indices', 'm1i64'], [f'rowwmax{c}'])
        rmax_c = nn('ReduceMax', [row_wmax], [f'rmax{c}'], axes=[2], keepdims=1)

        col_any = nn('ReduceMax', [ch], [f'colany{c}'], axes=[2], keepdims=1)
        col_any_b = nn('Greater', [col_any, 'c0f'], [f'colanyb{c}'])
        col_wmin = nn('Where', [col_any_b, 'col_indices', 'p999'], [f'colwmin{c}'])
        cmin_c = nn('ReduceMin', [col_wmin], [f'cmin{c}'], axes=[3], keepdims=1)
        col_wmax = nn('Where', [col_any_b, 'col_indices', 'm1i64'], [f'colwmax{c}'])
        cmax_c = nn('ReduceMax', [col_wmax], [f'cmax{c}'], axes=[3], keepdims=1)

        presence_c = nn('Greater', [rmax_c, 'm1i64'], [f'presence{c}'])

        hsub = nn('Sub', [rmax_c, rmin_c], [f'hsub{c}'])
        h_c = nn('Add', [hsub, 'c1i64'], [f'h{c}'])
        wsub = nn('Sub', [cmax_c, cmin_c], [f'wsub{c}'])
        w_c = nn('Add', [wsub, 'c1i64'], [f'w{c}'])
        square_c = nn('Equal', [h_c, w_c], [f'square{c}'])

        rc_sum = nn('Add', [rmax_c, cmin_c], [f'rcsum{c}'])
        TR_full = nn('Sub', [rc_sum, 'col_indices'], [f'TRfull{c}'])
        cr_sum = nn('Sub', [cmin_c, rmin_c], [f'crsum{c}'])
        TC_full = nn('Add', [cr_sum, 'row_indices'], [f'TCfull{c}'])
        TR_exp = nn('Expand', [TR_full, 'shape_1_1_30_30'], [f'TRexp{c}'])
        TC_exp = nn('Expand', [TC_full, 'shape_1_1_30_30'], [f'TCexp{c}'])
        TR_clip = nn('Clip', [TR_exp, 'c0i64', 'c29i64'], [f'TRclip{c}'])
        TC_clip = nn('Clip', [TC_exp, 'c0i64', 'c29i64'], [f'TCclip{c}'])
        TR_flat = nn('Reshape', [TR_clip, 'shape_30_30'], [f'TRflat{c}'])
        TC_flat = nn('Reshape', [TC_clip, 'shape_30_30'], [f'TCflat{c}'])
        TR_u = nn('Unsqueeze', [TR_flat], [f'TRu{c}'], axes=[2])
        TC_u = nn('Unsqueeze', [TC_flat], [f'TCu{c}'], axes=[2])
        gidx = nn('Concat', [TR_u, TC_u], [f'gidx{c}'], axis=2)
        gathered = nn('GatherND', [grid_2d, gidx], [f'gathered{c}'], batch_dims=0)
        gathered_r = nn('Reshape', [gathered, 'shape_1_1_30_30'], [f'gatheredr{c}'])
        eq_c = nn('Equal', ['color_idx64', gathered_r], [f'eqc{c}'])

        ge_r = nn('GreaterOrEqual', ['row_indices', rmin_c], [f'ger{c}'])
        le_r = nn('LessOrEqual', ['row_indices', rmax_c], [f'ler{c}'])
        ge_c = nn('GreaterOrEqual', ['col_indices', cmin_c], [f'gec{c}'])
        le_c = nn('LessOrEqual', ['col_indices', cmax_c], [f'lec{c}'])
        inrow = nn('And', [ge_r, le_r], [f'inrow{c}'])
        incol = nn('And', [ge_c, le_c], [f'incol{c}'])
        inbbox = nn('And', [inrow, incol], [f'inbbox{c}'])
        not_inbbox = nn('Not', [inbbox], [f'notin{c}'])
        ok_or_outside = nn('Or', [eq_c, not_inbbox], [f'okor{c}'])
        ok_i = nn('Cast', [ok_or_outside], [f'oki{c}'], to=I64)
        allok = nn('ReduceMin', [ok_i], [f'allok{c}'], axes=[1, 2, 3], keepdims=1)
        allok_bool = nn('Equal', [allok, 'c1i64'], [f'allokb{c}'])

        rotsym_c = nn('And', [square_c, allok_bool], [f'rotsym{c}'])
        is_anchor_c = nn('And', [presence_c, rotsym_c], [f'isanchor{c}'])
        is_anchor_i = nn('Cast', [is_anchor_c], [f'isanchori{c}'], to=I64)

        anchor_color_terms.append(nn('Mul', [is_anchor_i, cval], [f'anccol{c}']))
        anchor_rmin_terms.append(nn('Mul', [is_anchor_i, rmin_c], [f'ancrmin{c}']))
        anchor_cmin_terms.append(nn('Mul', [is_anchor_i, cmin_c], [f'anccmin{c}']))
        anchor_h_terms.append(nn('Mul', [is_anchor_i, h_c], [f'anch{c}']))
        anchor_w_terms.append(nn('Mul', [is_anchor_i, w_c], [f'ancw{c}']))

        not_anchor_c = nn('Not', [is_anchor_c], [f'notanc{c}'])
        is_other_c = nn('And', [presence_c, not_anchor_c], [f'isother{c}'])
        is_other_i = nn('Cast', [is_other_c], [f'isotheri{c}'], to=I64)
        is_other_f = nn('Cast', [is_other_c], [f'isotherf{c}'], to=F)

        other_color_terms.append(nn('Mul', [is_other_i, cval], [f'othcol{c}']))
        maskB_terms.append(nn('Mul', [ch, is_other_f], [f'maskbterm{c}']))

    def sum_chain(terms, prefix):
        acc = terms[0]
        for i, t in enumerate(terms[1:]):
            acc = nn('Add', [acc, t], [f'{prefix}_acc{i}'])
        return acc

    anchor_rmin = sum_chain(anchor_rmin_terms, 'ancrminsum')
    anchor_cmin = sum_chain(anchor_cmin_terms, 'anccminsum')
    anchor_h = sum_chain(anchor_h_terms, 'anchsum')
    anchor_w = sum_chain(anchor_w_terms, 'ancwsum')
    colorB = sum_chain(other_color_terms, 'othcolsum')
    maskB = sum_chain(maskB_terms, 'maskbsum')

    h_half = nn('Div', [anchor_h, 'c2i64'], ['h_half'])
    w_half = nn('Div', [anchor_w, 'c2i64'], ['w_half'])
    c0 = nn('Add', [anchor_rmin, h_half], ['c0_center'])
    c1 = nn('Add', [anchor_cmin, w_half], ['c1_center'])
    two_c0 = nn('Mul', [c0, 'c2i64'], ['two_c0'])
    two_c1 = nn('Mul', [c1, 'c2i64'], ['two_c1'])

    maskB_2d = nn('Reshape', [maskB, 'shape_30_30'], ['maskB_2d'])

    raw_row_h = nn('Sub', [two_c0, 'row_indices'], ['raw_row_h'])   # [1,1,30,1]
    raw_col_v = nn('Sub', [two_c1, 'col_indices'], ['raw_col_v'])   # [1,1,1,30]

    valid_row_h = nn('And', [nn('GreaterOrEqual', [raw_row_h, 'c0i64'], ['vrh_ge']),
                              nn('LessOrEqual', [raw_row_h, 'c29i64'], ['vrh_le'])], ['valid_row_h'])
    valid_col_v = nn('And', [nn('GreaterOrEqual', [raw_col_v, 'c0i64'], ['vcv_ge']),
                              nn('LessOrEqual', [raw_col_v, 'c29i64'], ['vcv_le'])], ['valid_col_v'])
    valid_row_h_f = nn('Cast', [valid_row_h], ['valid_row_h_f'], to=F)
    valid_col_v_f = nn('Cast', [valid_col_v], ['valid_col_v_f'], to=F)
    valid_both_f = nn('Mul', [valid_row_h_f, valid_col_v_f], ['valid_both_f'])

    clip_row_h = nn('Clip', [raw_row_h, 'c0i64', 'c29i64'], ['clip_row_h'])
    clip_col_v = nn('Clip', [raw_col_v, 'c0i64', 'c29i64'], ['clip_col_v'])

    def gather_mask(row_src, col_src, tag):
        row_exp = nn('Expand', [row_src, 'shape_1_1_30_30'], [f'rowexp_{tag}'])
        col_exp = nn('Expand', [col_src, 'shape_1_1_30_30'], [f'colexp_{tag}'])
        row_flat = nn('Reshape', [row_exp, 'shape_30_30'], [f'rowflat_{tag}'])
        col_flat = nn('Reshape', [col_exp, 'shape_30_30'], [f'colflat_{tag}'])
        row_u = nn('Unsqueeze', [row_flat], [f'rowu_{tag}'], axes=[2])
        col_u = nn('Unsqueeze', [col_flat], [f'colu_{tag}'], axes=[2])
        gidx = nn('Concat', [row_u, col_u], [f'gidx_{tag}'], axis=2)
        gathered = nn('GatherND', [maskB_2d, gidx], [f'gathered_{tag}'], batch_dims=0)
        return nn('Reshape', [gathered, 'shape_1_1_30_30'], [f'gatheredr_{tag}'])

    hmask_raw = gather_mask(clip_row_h, 'col_indices', 'h')
    hmask = nn('Mul', [hmask_raw, valid_row_h_f], ['hmask'])

    vmask_raw = gather_mask('row_indices', clip_col_v, 'v')
    vmask = nn('Mul', [vmask_raw, valid_col_v_f], ['vmask'])

    bmask_raw = gather_mask(clip_row_h, clip_col_v, 'b')
    bmask = nn('Mul', [bmask_raw, valid_both_f], ['bmask'])

    additional_mask = nn('Max', [nn('Max', [hmask, vmask], ['maxhv']), bmask], ['additional_mask'])
    additional_bool = nn('Greater', [additional_mask, 'half'], ['additional_bool'])

    final_color = nn('Where', [additional_bool, colorB, 'color_idx64'], ['final_color'])
    pred_idx = nn('Reshape', [final_color, 'shape_1_30_30'], ['pred_idx'])
    nn('OneHot', [pred_idx, 'depth10', 'oh_vals'], ['output'], axis=1)

    graph = helper.make_graph(nodes, 'task117', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


def _make():
    return _mask(_build_core())


model = _bake(_make(), 117)
