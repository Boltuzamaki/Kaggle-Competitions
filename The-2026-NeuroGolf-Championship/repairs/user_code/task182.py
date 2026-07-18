import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64
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

# ===== task182: ARC-DSL solve_776ffc46, transcribed + verified against arc_dsl_ref =====
#
# Ground truth intent (arc_dsl_ref/solvers.py::solve_776ffc46): find the connected
# (4-conn, univalued, bg=0) color-5 object whose own cell-set exactly equals the outline
# of its bounding box (a hollow rectangular "frame"). The region strictly inside that
# frame (the "inbox") holds a small "key" pattern -- its non-background cells, normalized,
# define a shape signature and a single key color. Every OTHER connected object anywhere
# on the grid (of any single color) whose own cell-set (translated) exactly matches that
# shape signature gets recolored to the key color (frame + everything else untouched).
#
# Verified in pure numpy (not the literal DSL primitives, but the same semantics) against
# all 267 train+test+arc-gen examples in data/task182.json: n_fail=0. Two data-driven
# invariants (checked computationally, not assumed) let this be built as static ONNX:
#   1. The frame's bounding box is ALWAYS exactly 7x7 (so its interior "inbox" is ALWAYS
#      exactly 5x5) -- across all 267 examples, frame_bbox_size == (7,7) every time.
#   2. Exactly one 7x7 window with an all-color-5 border ("hollow frame") exists per grid
#      -- verified across all 267 examples (some grids also contain OTHER, non-outline
#      color-5 shapes/decoys, but only one location ever satisfies the strict "all 24
#      border cells are 5" test, matching the true single valid frame).
#
# This turns the generic "find object whose cells == its own bbox outline" search into a
# small, fixed-size (7x7) static-kernel Conv scan over the whole 30x30 canvas (no dynamic
# shapes needed for that part).
#
# For the "find every other object with the same normalized shape" step: rather than a
# general (Loop/Scan-free is impossible for) flood-fill connected-component labeler, this
# exploits that a candidate object's own shape is *sliding-window testable*: for every
# anchor position, check that (a) all 5x5-mask-shape positions are the SAME single non-bg
# color v (correlation with a dynamic 0/1 mask-derived Conv kernel, grouped over all 9
# candidate colors at once), AND (b) every cell immediately (4-conn) adjacent to a mask
# cell but not itself a mask cell is NOT color v (this is the exact non-extension /
# maximality condition for "this connected component equals exactly the mask", verified
# to be necessary+sufficient by an earlier numpy prototype: using a *coarser* "whole
# surrounding ring must be background" condition instead of this precise 4-adjacency
# dilation caused 2/267 real failures where an unrelated same-colored object sat nearby
# in the ring but outside 4-adjacency -- the 4-adjacency-only version reproduces the true
# connected-component semantics exactly, n_fail=0 on all 267).
#
# Once each of the 24x24 candidate anchors is scored true/false (in a single grouped Conv
# pass over all 9 colors), the result is scattered back onto the 30x30 canvas via 25
# statically-enumerated (dr,dc) offsets (covering every possible cell of the fixed 5x5
# mask footprint) using Pad-shift + gate-by-mask5[dr,dc] + accumulate -- every op here has
# a static shape; the only dynamic shape (the 5x5 interior slice, always exactly 5x5 in
# practice) is resolved by the standard `_bake` helper.

def _make():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    inits = []
    nodes = []

    def addK(name, arr, dtype):
        inits.append(_K(name, arr, dtype))
        return name

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    # ---- constants ----
    addK('c0i64', [0], np.int64)
    addK('c1i64', [1], np.int64)
    addK('c5i64', [5], np.int64)
    addK('c6i64', [6], np.int64)
    addK('c0f', [0.0], np.float32)
    addK('shape1d', [-1], np.int64)
    addK('shape1111', [1, 1, 1, 1], np.int64)
    addK('ax23', [2, 3], np.int64)

    # 7x7 border-only kernel (fixed: border cells=1, interior=0) -> finds the hollow frame
    border_k = np.zeros((1, 1, 7, 7), dtype=np.float32)
    border_k[0, 0, 0, :] = 1.0; border_k[0, 0, 6, :] = 1.0
    border_k[0, 0, :, 0] = 1.0; border_k[0, 0, :, 6] = 1.0
    addK('border_k', border_k, np.float32)
    addK('c24f', [24.0], np.float32)

    addK('row_idx24', np.arange(24, dtype=np.int64).reshape(1, 1, 24, 1), np.int64)
    addK('col_idx24', np.arange(24, dtype=np.int64).reshape(1, 1, 1, 24), np.int64)

    # cross (plus, no center) kernel for 4-neighbor dilation
    cross_k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32).reshape(1, 1, 3, 3)
    addK('cross_k', cross_k, np.float32)

    # ---- channel 5 presence -> find the 7x7 hollow frame ----
    is5 = nn('Slice', ['input', 'c5i64', 'c6i64', 'c1i64'], ['is5'])  # [1,1,30,30]
    border_sum = nn('Conv', [is5, 'border_k'], ['border_sum'], kernel_shape=[7, 7], pads=[0, 0, 0, 0])  # [1,1,24,24]
    frame_match = nn('Equal', [border_sum, 'c24f'], ['frame_match'])
    frame_match_f = nn('Cast', [frame_match], ['frame_match_f'], to=F)
    frame_match_i = nn('Cast', [frame_match], ['frame_match_i'], to=I64)

    r0_w = nn('Mul', [frame_match_i, 'row_idx24'], ['r0_w'])
    c0_w = nn('Mul', [frame_match_i, 'col_idx24'], ['c0_w'])
    r0 = nn('ReduceSum', [r0_w, 'ax23'], ['r0'], keepdims=1)  # [1,1,1,1] int64
    c0 = nn('ReduceSum', [c0_w, 'ax23'], ['c0'], keepdims=1)

    start_r = nn('Add', [r0, 'c1i64'], ['start_r'])   # r0+1
    end_r = nn('Add', [r0, 'c6i64'], ['end_r'])       # r0+6
    start_c = nn('Add', [c0, 'c1i64'], ['start_c'])
    end_c = nn('Add', [c0, 'c6i64'], ['end_c'])
    start_r1 = nn('Reshape', [start_r, 'shape1d'], ['start_r1'])
    end_r1 = nn('Reshape', [end_r, 'shape1d'], ['end_r1'])
    start_c1 = nn('Reshape', [start_c, 'shape1d'], ['start_c1'])
    end_c1 = nn('Reshape', [end_c, 'shape1d'], ['end_c1'])

    # ---- extract the (always 5x5) interior "inbox" from the raw one-hot input ----
    addK('ax2', [2], np.int64); addK('ax3', [3], np.int64); addK('ax1', [1], np.int64)
    addK('c10i64', [10], np.int64)
    interior_y = nn('Slice', ['input', start_r1, end_r1, 'ax2'], ['interior_y'])       # [1,10,5,w]
    interior = nn('Slice', [interior_y, start_c1, end_c1, 'ax3'], ['interior'])        # [1,10,5,5]

    interior_colors = nn('Slice', [interior, 'c1i64', 'c10i64', 'ax1'], ['interior_colors'])  # channels 1..9, [1,9,5,5]
    mask5 = nn('ReduceMax', [interior_colors], ['mask5'], axes=[1], keepdims=1)           # [1,1,5,5] 0/1
    key_color_1to9 = nn('ReduceMax', [interior_colors], ['key_color_1to9'], axes=[2, 3], keepdims=1)  # [1,9,1,1]
    addK('zero_ch', np.zeros((1, 1, 1, 1), dtype=np.float32), np.float32)
    key_color_full = nn('Concat', ['zero_ch', key_color_1to9], ['key_color_full'], axis=1)  # [1,10,1,1]

    total = nn('ReduceSum', [mask5, 'ax23'], ['total'], keepdims=1)  # [1,1,1,1] float

    # ---- padded_mask (7x7) & required-background 4-adjacency ring ----
    addK('pads_1111', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    padded_mask = nn('Pad', ['mask5', 'pads_1111'], ['padded_mask'], mode='constant')  # [1,1,7,7]
    dil_raw = nn('Conv', ['padded_mask', 'cross_k'], ['dil_raw'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    dil_bool = nn('Greater', [dil_raw, 'c0f'], ['dil_bool'])
    dil_f = nn('Cast', [dil_bool], ['dil_f'], to=F)
    addK('one_f', [1.0], np.float32)
    one_minus_mask = nn('Sub', ['one_f', 'padded_mask'], ['one_minus_mask'])
    required_bg = nn('Mul', ['dil_f', 'one_minus_mask'], ['required_bg'])  # [1,1,7,7]

    # ---- expand kernels to 9 groups (one per non-bg color) & grouped conv over whole canvas ----
    addK('shape9177', [9, 1, 7, 7], np.int64)
    padded_mask_exp = nn('Expand', ['padded_mask', 'shape9177'], ['padded_mask_exp'])
    required_bg_exp = nn('Expand', ['required_bg', 'shape9177'], ['required_bg_exp'])

    # The window's own top-left (real_anchor) can legitimately fall as far as -(1+min_dr)
    # to (28-max_dr) for a mask whose occupied local rows/cols span [min_dr,max_dr] within
    # 0..4 -- worst case over all possible masks is real_anchor in [-5, 28] (34 values), so
    # pad the search canvas by 5 on each side before the VALID 7x7 conv scan.
    addK('pads_5555', [0, 0, 5, 5, 0, 0, 5, 5], np.int64)
    input_1to9_raw = nn('Slice', ['input', 'c1i64', 'c10i64', 'ax1'], ['input_1to9_raw'])  # [1,9,30,30]
    input_1to9 = nn('Pad', ['input_1to9_raw', 'pads_5555'], ['input_1to9'], mode='constant')  # [1,9,40,40]
    sum_mask_v = nn('Conv', ['input_1to9', 'padded_mask_exp'], ['sum_mask_v'], kernel_shape=[7, 7], pads=[0, 0, 0, 0], group=9)  # [1,9,34,34]
    sum_bg_v = nn('Conv', ['input_1to9', 'required_bg_exp'], ['sum_bg_v'], kernel_shape=[7, 7], pads=[0, 0, 0, 0], group=9)

    match_mask_ok = nn('Equal', ['sum_mask_v', 'total'], ['match_mask_ok'])
    addK('c0f_bg', [0.0], np.float32)
    match_bg_ok = nn('Equal', ['sum_bg_v', 'c0f_bg'], ['match_bg_ok'])
    match_v = nn('And', ['match_mask_ok', 'match_bg_ok'], ['match_v'])
    match_v_f = nn('Cast', [match_v], ['match_v_f'], to=F)
    match_any = nn('ReduceMax', [match_v_f], ['match_any'], axes=[1], keepdims=1)  # [1,1,34,34], a_idx = real_anchor+5

    # ---- scatter match_any back onto the 30x30 canvas for every (dr,dc) mask-footprint cell ----
    # canvas_row = real_anchor+1+dr = a_idx+(dr-4); canvas_col = a_idx_col+(dc-4).
    # pad_before/after (possibly negative -> ONNX Pad crops, supported since opset11) are
    # computed so the 34-wide a_idx axis lands exactly on the 30-wide canvas at this offset.
    recolor_acc = None
    for dr in range(5):
        for dc in range(5):
            pb_r, pa_r = dr - 4, -dr
            pb_c, pa_c = dc - 4, -dc
            pads_name = f'pads_{dr}_{dc}'
            addK(pads_name, [0, 0, pb_r, pb_c, 0, 0, pa_r, pa_c], np.int64)
            shifted = nn('Pad', ['match_any', pads_name], [f'shifted_{dr}_{dc}'], mode='constant')  # [1,1,30,30]
            g_start = addK(f'gstart_{dr}_{dc}', [dr, dc], np.int64)
            g_end = addK(f'gend_{dr}_{dc}', [dr + 1, dc + 1], np.int64)
            gate = nn('Slice', ['mask5', g_start, g_end, 'ax23'], [f'gate_{dr}_{dc}'])  # [1,1,1,1]
            contrib = nn('Mul', [shifted, gate], [f'contrib_{dr}_{dc}'])
            if recolor_acc is None:
                recolor_acc = contrib
            else:
                recolor_acc = nn('Add', [recolor_acc, contrib], [f'acc_{dr}_{dc}'])

    recolor_bool = nn('Greater', [recolor_acc, 'c0f'], ['recolor_bool'])
    out = nn('Where', [recolor_bool, 'key_color_full', 'input'], ['output'])

    graph = helper.make_graph(nodes, 'task182', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


model = _bake(_mask(_make()), 182)
