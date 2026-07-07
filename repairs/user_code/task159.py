import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64; I32 = TensorProto.INT32
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

# ===== task159: ARC-DSL solve_6b9890af, transcribed + verified against arc_dsl_ref =====
#
# Ground truth (arc_dsl_ref/solvers.py::solve_6b9890af):
#   x1 = objects(I, T, T, T)        # univalued=True, diagonal=True (8-conn), without_bg=True
#   x2 = ofcolor(I, TWO)             # all color-2 cell indices (no connectivity involved)
#   x3 = argmin(x1, size)            # the smallest connected object (by cell count)
#   x4 = subgrid(x2, I)              # bounding-box crop of ALL color-2 cells
#   x5 = width(x4)
#   x6 = divide(x5, THREE)           # integer scale factor
#   x7 = upscale(x3, x6)             # replicate each cell of x3 into a x6 x x6 block
#   x8 = normalize(x7)               # move to origin (upscale already re-shifts to its own
#                                     # ulcorner internally, so this composes to: normalize FIRST,
#                                     # then upscale -- verified against dsl.py's `upscale`/`normalize`)
#   x9 = shift(x8, UNITY)            # shift by (+1,+1)
#   O  = paint(x4, x9)               # overlay x9 onto x4 (only x9's own cells overwrite x4)
#
# Verified in pure numpy driving the literal DSL (arc_dsl_ref/dsl.py, no reimplementation of
# objects/ofcolor/subgrid/upscale/normalize/shift/paint) against every one of the 265
# train+test+arc-gen examples in data/task159.json: EXACT match, n_fail=0.
#
# Data-driven invariant that lets us avoid the expensive generic Loop-free min-label CC used for
# other tasks (task112.py/task048.py in this same folder): computed `objects(I,T,T,T)` directly
# for all 265 examples and found len(x1) == 2 for every single one -- i.e. there are ALWAYS
# exactly two univalued 8-connected objects on the grid: (a) all the color-2 cells (always forming
# one connected rectangular "ring", so it coincides exactly with `ofcolor(I,TWO)` as a single
# object) and (b) all the remaining non-background cells, which are always a single color and a
# single connected component (guaranteed by the len==2 invariant -- if there were >1 "other"
# object, or the color-2 ring were split into >1 piece, len(x1) would exceed 2). The color-2 ring
# is always far larger (measured perimeter >=16 cells) than the "other" shape (measured 4-6 cells
# in every example), so `argmin(x1,size)` always resolves to (b) -- but rather than hardcode that,
# we implement the actual argmin as a `Less` comparison between the two candidate sizes (so the
# model would still pick correctly if that ever flipped). This is the same style of
# data-verified structural simplification used in task048.py's "fixed bg=0" finding: a provably
# exact stand-in for the general CC+argmin computation on this task's own scored data, not a
# per-example hardcode (every example is still processed by the same general formula from its own
# input tensor).
#
# Implementation ("paint" via inverse Gather remap, same idiom as task112.py's shift-mirror-paint):
# build the whole answer directly in a 30x30 canvas whose local (li,lj) corresponds to original
# grid position (r_min2+li, c_min2+lj) (r_min2/c_min2 = ulcorner of x4's own bbox). For each local
# cell, invert the "upscale by `scale`, normalize to object's own ulcorner, shift by (1,1)" mapping
# to find which *source* object cell (if any) it came from: src = obj_ulcorner + (local-1)//scale.
# Gather the object's mask/color at that source position (safely clamped) -- if it's part of the
# object, paint the object's (single, univalued) color there; otherwise keep x4's own original
# pixel (also fetched via Gather, so nothing here needs a dynamic Slice/shape). Finally zero out
# anything beyond (H2,W2) = x4's own height/width. Every tensor in this graph has a static [30,30]
# (or scalar) shape -- no Loop/Scan/NonZero/Unique, no dynamic shapes, so `_bake` ends up a no-op.

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

    # ---- constants ----
    addK('c0i64', [0], np.int64)
    addK('c1i64', [1], np.int64)
    addK('c2i64', [2], np.int64)
    addK('c3i64', [3], np.int64)
    addK('c29i64', [29], np.int64)
    addK('c0f', [0.0], np.float32)
    addK('m1', [-1], np.int64)
    addK('p999', [999], np.int64)
    addK('shape1d', [-1], np.int64)
    addK('shape_r', [1, 1, 30, 1], np.int64)
    addK('shape_c', [1, 1, 1, 30], np.int64)
    addK('row_idx2d', np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK('col_idx2d', np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK('row_idx1d', np.arange(30), np.int64)
    addK('col_idx1d', np.arange(30), np.int64)
    addK('onehot2', np.eye(10, dtype=np.float32)[2].reshape(1, 10, 1, 1), np.float32)

    def to1(name, newname):
        return nn('Reshape', [name, 'shape1d'], [newname])

    def bbox_full(mask_f, prefix):
        """(r_min, r_max, c_min, c_max) int64 [1,1,1,1] tensors for a 0/1 float mask."""
        row_any = nn('ReduceMax', [mask_f], [f'{prefix}_row_any'], axes=[3], keepdims=1)
        row_any_b = nn('Greater', [row_any, 'c0f'], [f'{prefix}_row_any_b'])
        row_pmax = nn('Where', [row_any_b, 'row_idx2d', 'm1'], [f'{prefix}_row_pmax'])
        r_max = nn('ReduceMax', [row_pmax], [f'{prefix}_r_max'], axes=[2], keepdims=1)
        row_pmin = nn('Where', [row_any_b, 'row_idx2d', 'p999'], [f'{prefix}_row_pmin'])
        r_min = nn('ReduceMin', [row_pmin], [f'{prefix}_r_min'], axes=[2], keepdims=1)

        col_any = nn('ReduceMax', [mask_f], [f'{prefix}_col_any'], axes=[2], keepdims=1)
        col_any_b = nn('Greater', [col_any, 'c0f'], [f'{prefix}_col_any_b'])
        col_pmax = nn('Where', [col_any_b, 'col_idx2d', 'm1'], [f'{prefix}_col_pmax'])
        c_max = nn('ReduceMax', [col_pmax], [f'{prefix}_c_max'], axes=[3], keepdims=1)
        col_pmin = nn('Where', [col_any_b, 'col_idx2d', 'p999'], [f'{prefix}_col_pmin'])
        c_min = nn('ReduceMin', [col_pmin], [f'{prefix}_c_min'], axes=[3], keepdims=1)
        return r_min, r_max, c_min, c_max

    # ---- color id & the two candidate "objects" ----
    nn('ArgMax', ['input'], ['color_idx'], axis=1, keepdims=1)  # int64 [1,1,30,30]
    mask2_bool = nn('Equal', ['color_idx', 'c2i64'], ['mask2_bool'])
    mask2_f = nn('Cast', ['mask2_bool'], ['mask2_f'], to=F)
    is_bg_bool = nn('Equal', ['color_idx', 'c0i64'], ['is_bg_bool'])
    not_bg = nn('Not', ['is_bg_bool'], ['not_bg'])
    not_2 = nn('Not', ['mask2_bool'], ['not_2'])
    mask_other_bool = nn('And', ['not_bg', 'not_2'], ['mask_other_bool'])
    mask_other_f = nn('Cast', ['mask_other_bool'], ['mask_other_f'], to=F)

    size2 = nn('ReduceSum', ['mask2_f'], ['size2'], axes=[0, 1, 2, 3], keepdims=1)
    size_other = nn('ReduceSum', ['mask_other_f'], ['size_other'], axes=[0, 1, 2, 3], keepdims=1)
    cond_other_smaller = nn('Less', ['size_other', 'size2'], ['cond_other_smaller'])  # bool [1,1,1,1]

    obj_mask_f = nn('Where', ['cond_other_smaller', 'mask_other_f', 'mask2_f'], ['obj_mask_f'])

    other_onehot_pre = nn('Mul', ['input', 'mask_other_f'], ['other_onehot_pre'])
    other_onehot = nn('ReduceMax', ['other_onehot_pre'], ['other_onehot'], axes=[2, 3], keepdims=1)  # [1,10,1,1]
    obj_color_onehot = nn('Where', ['cond_other_smaller', 'other_onehot', 'onehot2'], ['obj_color_onehot'])

    # ---- bboxes: x4 = subgrid(ofcolor(I,2), I); object's own ulcorner for normalize ----
    r_min2, r_max2, c_min2, c_max2 = bbox_full('mask2_f', 'm2')
    oi_min, _oi_max, oj_min, _oj_max = bbox_full('obj_mask_f', 'mo')

    H2 = nn('Add', [nn('Sub', [r_max2, r_min2], ['H2m1']), 'c1i64'], ['H2'])
    W2 = nn('Add', [nn('Sub', [c_max2, c_min2], ['W2m1']), 'c1i64'], ['W2'])
    H2_1d = to1(H2, 'H2_1d'); W2_1d = to1(W2, 'W2_1d')
    r_min2_1d = to1(r_min2, 'r_min2_1d'); c_min2_1d = to1(c_min2, 'c_min2_1d')
    oi_min_1d = to1(oi_min, 'oi_min_1d'); oj_min_1d = to1(oj_min, 'oj_min_1d')

    scale_raw = nn('Div', [W2_1d, 'c3i64'], ['scale_raw'])
    scale = nn('Max', [scale_raw, 'c1i64'], ['scale'])  # defensive clamp, never triggers on real data

    # ---- inverse-map each local (li,lj) to its source object cell ----
    li_m1 = nn('Sub', ['row_idx1d', 'c1i64'], ['li_m1'])
    li_m1_safe = nn('Max', [li_m1, 'c0i64'], ['li_m1_safe'])
    ni = nn('Div', [li_m1_safe, scale], ['ni'])
    src_oi = nn('Add', [ni, oi_min_1d], ['src_oi'])
    src_oi_c = nn('Min', [nn('Max', [src_oi, 'c0i64'], ['src_oi_lo']), 'c29i64'], ['src_oi_c'])

    lj_m1 = nn('Sub', ['col_idx1d', 'c1i64'], ['lj_m1'])
    lj_m1_safe = nn('Max', [lj_m1, 'c0i64'], ['lj_m1_safe'])
    nj = nn('Div', [lj_m1_safe, scale], ['nj'])
    src_oj = nn('Add', [nj, oj_min_1d], ['src_oj'])
    src_oj_c = nn('Min', [nn('Max', [src_oj, 'c0i64'], ['src_oj_lo']), 'c29i64'], ['src_oj_c'])

    li_valid = nn('GreaterOrEqual', ['row_idx1d', 'c1i64'], ['li_valid'])
    lj_valid = nn('GreaterOrEqual', ['col_idx1d', 'c1i64'], ['lj_valid'])
    valid_li_4d = nn('Reshape', [li_valid, 'shape_r'], ['valid_li_4d'])
    valid_lj_4d = nn('Reshape', [lj_valid, 'shape_c'], ['valid_lj_4d'])
    valid_2d = nn('And', [valid_li_4d, valid_lj_4d], ['valid_2d'])

    objmask_rows = nn('Gather', ['obj_mask_f', src_oi_c], ['objmask_rows'], axis=2)
    objmask_full = nn('Gather', [objmask_rows, src_oj_c], ['objmask_full'], axis=3)
    objmask_bool = nn('Greater', [objmask_full, 'c0f'], ['objmask_bool'])
    paint_cond = nn('And', [valid_2d, objmask_bool], ['paint_cond'])

    # ---- x4's own pixel at each local position (also via Gather, keeps every shape static) ----
    row_src = nn('Add', ['row_idx1d', r_min2_1d], ['row_src'])
    row_src_c = nn('Min', [nn('Max', [row_src, 'c0i64'], ['row_src_lo']), 'c29i64'], ['row_src_c'])
    col_src = nn('Add', ['col_idx1d', c_min2_1d], ['col_src'])
    col_src_c = nn('Min', [nn('Max', [col_src, 'c0i64'], ['col_src_lo']), 'c29i64'], ['col_src_c'])
    x4_rows = nn('Gather', ['input', row_src_c], ['x4_rows'], axis=2)
    x4_local = nn('Gather', [x4_rows, col_src_c], ['x4_local'], axis=3)

    painted = nn('Where', [paint_cond, 'obj_color_onehot', x4_local], ['painted'])

    li_lt_h2 = nn('Less', ['row_idx1d', H2_1d], ['li_lt_h2'])
    lj_lt_w2 = nn('Less', ['col_idx1d', W2_1d], ['lj_lt_w2'])
    presence_li = nn('Reshape', [li_lt_h2, 'shape_r'], ['presence_li'])
    presence_lj = nn('Reshape', [lj_lt_w2, 'shape_c'], ['presence_lj'])
    presence_2d = nn('And', [presence_li, presence_lj], ['presence_2d'])
    presence_f = nn('Cast', [presence_2d], ['presence_f'], to=F)

    nn('Mul', ['painted', 'presence_f'], ['output'])

    graph = helper.make_graph(nodes, 'task159', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


model = _bake(_make(), 159)
