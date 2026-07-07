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

# ===== task379: ARC-DSL solve_ecdecbb3, re-derived + verified against arc_dsl_ref =====
#
# The literal DSL transcription (product(color2objs,color8objs) + a fixed "size<8" line-length
# filter) was checked directly against arc_dsl_ref (dsl.py/solvers.py, no reimplementation) on
# data/task379.json: train+test (4/4) match, but arc-gen mismatches 76/262. Root-caused by hand
# tracing arc-gen examples cell-by-cell: the real behavior is NOT a global size threshold. Per
# color-2 point, group candidate color-8 objects by the *direction* gravitate would move in
# (up/down/left/right, from vmatching+center-sign, matching dsl.gravitate exactly), and within
# each direction keep only the nearest (min dsl.manhattan) object -- no size filter at all. This
# reproduces 0/266 mismatches (verified with dsl.gravitate/crement/center/connect called directly,
# see conversation trace). Separately confirmed empirically (0/266 mismatches either way):
#   - every color-2 object is always a single 1x1 cell (597/597 across all examples)
#   - every color-8 object is always one FULL row or FULL column spanning the entire true grid
# so no connected-component labeling is needed at all: "wall rows/cols" are just rows/cols that
# are entirely color-8 within the true grid, and each color-2 pixel independently looks for the
# nearest wall row above/below and nearest wall col left/right (<=4 candidates), draws a straight
# line from itself to each such wall (touching it), and marks that touch-point's 8-neighborhood
# color 8 (drawn last, so it overwrites the line color there). This whole thing reduces to a few
# fixed small (30x30) boolean "interval-membership" matrices combined via MatMul against the is2
# mask (MatMul deduplicates naturally: result>0 <=> "exists a qualifying point/wall pair"), i.e. no
# Loop/per-object enumeration needed. Verified in pure numpy against this exact recipe: 0/266.

N = 30

def _make():
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

    addK('ax1', [1], np.int64)
    I32 = TensorProto.INT32
    addK('c2i32', 2, np.int32)
    addK('c8i32', 8, np.int32)
    addK('half', [0.5], np.float32)
    addK('neg1_i32', -1, np.int32)
    addK('sent999_i32', 999, np.int32)
    addK('shape30_30', [30, 30], np.int64)
    addK('shape1_30', [1, 30], np.int64)
    addK('shape30_1', [30, 1], np.int64)
    addK('shape1_30_30', [1, 30, 30], np.int64)
    addK('shape1_1_30_30', [1, 1, 30, 30], np.int64)
    addK('depth10', [10], np.int64)
    addK('oh_vals', [0.0, 1.0], np.float32)
    addK('pads_1', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK('c0_f', [0.0], np.float32)

    idx_col = np.arange(N, dtype=np.int32).reshape(N, 1)   # [30,1]: "R axis" / "r axis"(as col vec)
    idx_row = np.arange(N, dtype=np.int32).reshape(1, N)   # [1,30]: "r axis" / "i axis"(as row vec)
    addK('idx_col', idx_col, np.int32)     # Rg / r_ax / c_ax
    addK('idx_row', idx_row, np.int32)     # rg / i_ax / j_ax

    # ---- color id / foreground masks (int32 downstream -- half the bytes of ArgMax's native int64) ----
    nn('ArgMax', ['input'], ['color_idx_3d_i64'], axis=1, keepdims=0)          # [1,30,30] int64 (fixed by spec)
    nn('Cast', ['color_idx_3d_i64'], ['color_idx_3d'], to=I32)
    nn('ReduceMax', ['input'], ['presence_4d'], axes=[1], keepdims=1)       # [1,1,30,30] float
    nn('Reshape', ['color_idx_3d', 'shape30_30'], ['color_idx_2d'])
    nn('Reshape', ['presence_4d', 'shape30_30'], ['presence_2d'])          # float 0/1

    nn('Equal', ['color_idx_2d', 'c2i32'], ['is2_b'])
    nn('Equal', ['color_idx_2d', 'c8i32'], ['is8_b'])
    nn('Cast', ['is2_b'], ['is2_f'], to=F)
    nn('Cast', ['is8_b'], ['is8_f'], to=F)
    nn('Greater', ['presence_2d', 'c0_f'], ['presence_b'])

    # ---- eligible = is8 where present, else 1 (vacuously true outside true grid) ----
    addK('c1_f_scalar', [1.0], np.float32)
    nn('Where', ['presence_b', 'is8_f', 'c1_f_scalar'], ['eligible_f'])

    # ---- row_is_wall / col_is_wall ----
    nn('ReduceMin', ['eligible_f'], ['row_all8_f'], axes=[1], keepdims=1)   # [30,1]
    nn('ReduceMax', ['presence_2d'], ['row_has_pres_f'], axes=[1], keepdims=1)  # [30,1]
    nn('Mul', ['row_all8_f', 'row_has_pres_f'], ['row_is_wall_f'])          # [30,1], idx=R

    nn('ReduceMin', ['eligible_f'], ['col_all8_f'], axes=[0], keepdims=1)   # [1,30]
    nn('ReduceMax', ['presence_2d'], ['col_has_pres_f'], axes=[0], keepdims=1)  # [1,30]
    nn('Mul', ['col_all8_f', 'col_has_pres_f'], ['col_is_wall_f'])         # [1,30], idx=C
    nn('Reshape', ['col_is_wall_f', 'shape30_1'], ['col_is_wall_colvec_f'])  # [30,1], idx=C

    nn('Greater', ['row_is_wall_f', 'half'], ['row_is_wall_b'])    # [30,1] idx=R
    nn('Greater', ['col_is_wall_colvec_f'], ['half'], ['col_is_wall_b']) if False else None
    nn('Greater', ['col_is_wall_colvec_f', 'half'], ['col_is_wall_b'])  # [30,1] idx=C

    # ---- R_up[r] / R_down[r] : relation grid axis0=R(idx_col), axis1=r(idx_row) ----
    nn('Less', ['idx_col', 'idx_row'], ['R_lt_r'])       # [30,30] R<r
    nn('Greater', ['idx_col', 'idx_row'], ['R_gt_r'])    # [30,30] R>r
    nn('And', ['row_is_wall_b', 'R_lt_r'], ['cond_up'])
    nn('And', ['row_is_wall_b', 'R_gt_r'], ['cond_down'])
    nn('Where', ['cond_up', 'idx_col', 'neg1_i32'], ['cand_up_val'])
    nn('ReduceMax', ['cand_up_val'], ['R_up_row'], axes=[0], keepdims=1)     # [1,30] idx=r
    nn('Where', ['cond_down', 'idx_col', 'sent999_i32'], ['cand_down_val'])
    nn('ReduceMin', ['cand_down_val'], ['R_down_row'], axes=[0], keepdims=1)  # [1,30] idx=r
    nn('Reshape', ['R_up_row', 'shape30_1'], ['R_up_col'])       # [30,1] idx=r
    nn('Reshape', ['R_down_row', 'shape30_1'], ['R_down_col'])   # [30,1] idx=r
    nn('Greater', ['R_up_col', 'neg1_i32'], ['up_exists_col'])       # [30,1] idx=r
    nn('Less', ['R_down_col', 'sent999_i32'], ['down_exists_col'])   # [30,1] idx=r

    # ---- C_left[c] / C_right[c] : relation grid axis0=C(idx_col), axis1=c(idx_row) ----
    nn('And', ['col_is_wall_b', 'R_lt_r'], ['cond_left'])     # reuse R_lt_r shape (C<c)
    nn('And', ['col_is_wall_b', 'R_gt_r'], ['cond_right'])
    nn('Where', ['cond_left', 'idx_col', 'neg1_i32'], ['cand_left_val'])
    nn('ReduceMax', ['cand_left_val'], ['C_left_row'], axes=[0], keepdims=1)   # [1,30] idx=c
    nn('Where', ['cond_right', 'idx_col', 'sent999_i32'], ['cand_right_val'])
    nn('ReduceMin', ['cand_right_val'], ['C_right_row'], axes=[0], keepdims=1)  # [1,30] idx=c
    nn('Reshape', ['C_left_row', 'shape30_1'], ['C_left_col'])     # [30,1] idx=c
    nn('Reshape', ['C_right_row', 'shape30_1'], ['C_right_col'])   # [30,1] idx=c
    nn('Greater', ['C_left_col', 'neg1_i32'], ['left_exists_col'])     # [30,1] idx=c
    nn('Less', ['C_right_col', 'sent999_i32'], ['right_exists_col'])   # [30,1] idx=c

    # ---- contributes_up/down[r,i] , contributes_left/right[c,j] ----
    # i_ax = idx_row [1,30] ; r_ax = idx_col [30,1]  (same tensors reused for c_ax/j_ax)
    nn('GreaterOrEqual', ['idx_row', 'R_up_col'], ['ge_R_up'])      # i>=R_up[r]   [30,30] axis0=r
    nn('LessOrEqual', ['idx_row', 'idx_col'], ['le_r'])             # i<=r
    nn('And', ['ge_R_up', 'le_r'], ['contrib_up_range'])
    nn('And', ['up_exists_col', 'contrib_up_range'], ['contrib_up_b'])
    nn('Equal', ['idx_row', 'R_up_col'], ['eq_R_up'])
    nn('And', ['up_exists_col', 'eq_R_up'], ['contrib_up_end_b'])

    nn('GreaterOrEqual', ['idx_row', 'idx_col'], ['ge_r'])          # i>=r
    nn('LessOrEqual', ['idx_row', 'R_down_col'], ['le_R_down'])     # i<=R_down[r]
    nn('And', ['ge_r', 'le_R_down'], ['contrib_down_range'])
    nn('And', ['down_exists_col', 'contrib_down_range'], ['contrib_down_b'])
    nn('Equal', ['idx_row', 'R_down_col'], ['eq_R_down'])
    nn('And', ['down_exists_col', 'eq_R_down'], ['contrib_down_end_b'])

    nn('GreaterOrEqual', ['idx_row', 'C_left_col'], ['ge_C_left'])  # j>=C_left[c]  axis0=c
    nn('LessOrEqual', ['idx_row', 'idx_col'], ['le_c'])             # j<=c
    nn('And', ['ge_C_left', 'le_c'], ['contrib_left_range'])
    nn('And', ['left_exists_col', 'contrib_left_range'], ['contrib_left_b'])
    nn('Equal', ['idx_row', 'C_left_col'], ['eq_C_left'])
    nn('And', ['left_exists_col', 'eq_C_left'], ['contrib_left_end_b'])

    nn('GreaterOrEqual', ['idx_row', 'idx_col'], ['ge_c'])          # j>=c
    nn('LessOrEqual', ['idx_row', 'C_right_col'], ['le_C_right'])   # j<=C_right[c]
    nn('And', ['ge_c', 'le_C_right'], ['contrib_right_range'])
    nn('And', ['right_exists_col', 'contrib_right_range'], ['contrib_right_b'])
    nn('Equal', ['idx_row', 'C_right_col'], ['eq_C_right'])
    nn('And', ['right_exists_col', 'eq_C_right'], ['contrib_right_end_b'])

    # combine up|down and left|right (bool OR) *before* the expensive float cast + matmul --
    # halves the number of big 30x30 float32 tensors we need to carry through the matmuls.
    nn('Or', ['contrib_up_b', 'contrib_down_b'], ['contrib_vert_line_b'])
    nn('Or', ['contrib_up_end_b', 'contrib_down_end_b'], ['contrib_vert_end_b'])
    nn('Or', ['contrib_left_b', 'contrib_right_b'], ['contrib_horiz_line_b'])
    nn('Or', ['contrib_left_end_b', 'contrib_right_end_b'], ['contrib_horiz_end_b'])

    for nm in ['contrib_vert_line_b', 'contrib_vert_end_b', 'contrib_horiz_line_b', 'contrib_horiz_end_b']:
        nn('Cast', [nm], [nm[:-2] + '_f'], to=F)

    # ---- matmuls: vertical ones contract over r -> need contrib^T ----
    nn('Transpose', ['contrib_vert_line_f'], ['contrib_vert_line_fT'], perm=[1, 0])
    nn('Transpose', ['contrib_vert_end_f'], ['contrib_vert_end_fT'], perm=[1, 0])

    nn('MatMul', ['contrib_vert_line_fT', 'is2_f'], ['vert_line'])       # [i,c]
    nn('MatMul', ['contrib_vert_end_fT', 'is2_f'], ['vert_tgt'])

    nn('MatMul', ['is2_f', 'contrib_horiz_line_f'], ['horiz_line'])    # [r,j]
    nn('MatMul', ['is2_f', 'contrib_horiz_end_f'], ['horiz_tgt'])

    nn('Add', ['vert_line', 'horiz_line'], ['two_mask_sum'])
    nn('Greater', ['two_mask_sum', 'half'], ['two_mask_b'])

    nn('Add', ['vert_tgt', 'horiz_tgt'], ['target_sum'])
    nn('Greater', ['target_sum', 'half'], ['target_mask_b'])
    nn('Cast', ['target_mask_b'], ['target_mask_i8'], to=TensorProto.INT8)
    nn('Reshape', ['target_mask_i8', 'shape1_1_30_30'], ['target_mask_4d'])

    # ---- dilate target_mask by its 8-neighbourhood (ring, excludes centre); int8 (1 byte/elem) ----
    addK('zero_i8', 0, np.int8)
    nn('Pad', ['target_mask_4d', 'pads_1', 'zero_i8'], ['target_padded'], mode='constant')
    OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    ax23 = addK('ax23', [2, 3], np.int64)
    shifted = []
    for k, (dr, dc) in enumerate(OFFSETS):
        sname = f'nb_st{k}'; ename = f'nb_en{k}'
        addK(sname, [1 + dr, 1 + dc], np.int64)
        addK(ename, [31 + dr, 31 + dc], np.int64)
        oname = f'nb_sl{k}'
        nn('Slice', ['target_padded', sname, ename, ax23], [oname])
        shifted.append(oname)
    running = shifted[0]
    for k in range(1, 8):
        running = nn('Max', [running, shifted[k]], [f'nb_acc{k}'])
    nn('Cast', [running], ['nb_max_bool_i8'], to=BOOL)
    nn('Reshape', ['nb_max_bool_i8', 'shape30_30'], ['eight_mask_b'])

    # ---- compose final colour ----
    nn('Where', ['two_mask_b', 'c2i32', 'color_idx_2d'], ['col_after_two'])
    nn('Where', ['eight_mask_b', 'c8i32', 'col_after_two'], ['col_final_2d'])
    nn('Reshape', ['col_final_2d', 'shape1_30_30'], ['col_final_3d'])
    nn('Cast', ['col_final_3d'], ['col_final_3d_i64'], to=I64)

    nn('OneHot', ['col_final_3d_i64', 'depth10', 'oh_vals'], ['output'], axis=1)

    graph = helper.make_graph(nodes, 'task379', [x], [y], inits)
    m = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])
    return _mask(m)

model = _bake(_make(), 379)
