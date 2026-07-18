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

# ===== task198: ARC-DSL solve_83302e8f, transcribed + verified against arc_dsl_ref =====
#
# Ground truth (arc_dsl_ref/solvers.py::solve_83302e8f):
#   x1 = objects(I, T, F, F)         # univalued=True, diagonal=False (4-conn), without_bg=False
#   x2 = colorfilter(x1, ZERO)       # keep only the background-coloured (0) objects
#   x3 = sfilter(x2, square)         # square(): FILLED rectangle (area==h*w) AND h==w
#   x4 = difference(x2, x3)          # the non-square background objects
#   O  = paint(paint(I, recolor(THREE, merge(x3))), recolor(FOUR, merge(x4)))
#
# Verified in pure numpy (scipy 4-connected labeling of colour-0 regions, matching
# arc_dsl_ref/dsl.py's `square()` definition literally -- filled rectangle AND h==w) against
# ALL 266 train+test+arc-gen examples in data/task198.json: n_fail=0.
#
# One data-driven refinement needed beyond the literal DSL reading: `square()` as literally
# written treats a 1x1 region (h=w=1, area=1=h*w) as "square" -> recolor 3. But EVERY 1x1
# background-colour region in the real arc-gen data (240 occurrences) is recoloured 4, not 3.
# Also confirmed: in this dataset, "filled" (area==h*w) and "h==w" are exactly equivalent --
# there is not a single filled-but-non-square or square-but-filled-with-h==1 case among the
# 3574 background regions across all 266 examples (other than the 1x1 case). So the true
# generalizing predicate is: is_square3 = filled(area==h*w) AND area>1; everything else
# (including all size-1 regions, and all non-filled/non-square regions) -> 4.
#
# Connected-component labeling (no Loop/Scan/NonZero/Unique): Loop-free iterative min-label
# propagation, the same idiom as task048/task112, but gated on "neighbour is background colour
# 0" (not "is foreground"), with 4-connectivity only (matching diagonal=False), and packing
# FIVE per-cell state channels through a SINGLE Min-propagation each iteration (channels: flat
# init-label, row, col, -row, -col -- the two negated channels let Min double as Max for the
# row/col upper bounds, so bbox extremes converge alongside the component-id in one pass):
# after convergence, channel0 == the minimum flat index within the 4-connected background
# component (a stable per-component id), channels1/2 == that component's (r_min,c_min), and
# channels3/4 negated == (r_max,c_max). "Is-background" (colour==0) is additionally gated on
# the true-grid-extent presence mask (ReduceMax over all 10 input channels) so the all-zero
# 30x30 zero-padding beyond the real grid can never falsely bridge into a real background
# region (input's one-hot encoding leaves ALL 10 channels 0 outside the true grid, so presence
# correctly separates real background cells from padding).
#
# Component AREA (needed for the "filled" check, since bbox alone can't detect a hole/notch)
# is computed via an exact pairwise reduction (no Unique/groupby op needed): flatten the
# converged component-id to a length-900 vector L and the background mask to a length-900
# vector B; area[p] = sum_q (L[p]==L[q]) * B[q], done as one 900x900 Equal+Mul+ReduceSum (cheap,
# fully static, no dynamic shapes).
#
# R_ITERS=40 (numpy-simulated the EXACT propagation algorithm below at increasing iteration
# counts against every one of the 266 examples: n_fail=0 first at R=30, stays 0 through R=100;
# 40 gives a comfortable margin over the true worst-case component graph diameter ~<=39
# measured on this data).

R_ITERS = 40
SENT = 100000.0
OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # N, S, W, E (4-connectivity only)


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
    addK('c0f', [0.0], np.float32)
    addK('half', [0.5], np.float32)
    addK('onef', [1.0], np.float32)
    addK('onehalf', [1.5], np.float32)
    addK('col3f', [3.0], np.float32)
    addK('col4f', [4.0], np.float32)
    addK('sent', [SENT], np.float32)
    addK('pads_hw', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK('ax23', [2, 3], np.int64)
    addK('depth10', [10], np.int64)
    addK('oh_vals', [0.0, 1.0], np.float32)
    addK('shape1_30_30', [1, 30, 30], np.int64)
    addK('shape900', [900], np.int64)
    addK('shape900_1', [900, 1], np.int64)
    addK('shape1_900', [1, 900], np.int64)
    addK('shape1_1_30_30', [1, 1, 30, 30], np.int64)

    row_idx = np.arange(30).reshape(30, 1) * np.ones((1, 30))
    col_idx = np.arange(30).reshape(1, 30) * np.ones((30, 1))
    init_label = (np.arange(30).reshape(30, 1) * 30 + np.arange(30).reshape(1, 30)).astype(np.float32)
    state0 = np.stack([init_label, row_idx, col_idx, -row_idx, -col_idx], axis=0).reshape(1, 5, 30, 30).astype(np.float32)
    addK('state0', state0, np.float32)

    # per-offset slice start/end constants (fixed, hoisted outside all loops)
    starts = []
    ends = []
    for k, (di, dj) in enumerate(OFFSETS):
        s = addK(f'st{k}', [1 + di, 1 + dj], np.int64)
        e = addK(f'en{k}', [31 + di, 31 + dj], np.int64)
        starts.append(s)
        ends.append(e)

    def four_slices(padded_name, prefix):
        outs = []
        for k in range(4):
            oname = f'{prefix}_sl{k}'
            nn('Slice', [padded_name, starts[k], ends[k], 'ax23'], [oname])
            outs.append(oname)
        return outs

    # ---- colour / background / true-grid-extent presence ----
    nn('ArgMax', ['input'], ['color_idx64'], axis=1, keepdims=1)
    nn('Cast', ['color_idx64'], ['color_idx_f'], to=F)
    nn('Equal', ['color_idx64', 'c0i64'], ['is_zero_color'])
    nn('ReduceMax', ['input'], ['presence_bg'], axes=[1], keepdims=1)
    nn('Greater', ['presence_bg', 'half'], ['is_real'])
    nn('And', ['is_zero_color', 'is_real'], ['is_bg_bool'])
    nn('Cast', ['is_bg_bool'], ['is_bg_f'], to=F)

    # ---- fixed (iteration-independent) "is 4-neighbor background" masks ----
    nn('Pad', ['is_bg_f', 'pads_hw', 'c0f'], ['padded_isbg'], mode='constant')
    shifted_isbg = four_slices('padded_isbg', 'nbrbg')
    nbr_is_bg = []
    for k, sbg in enumerate(shifted_isbg):
        nbr_is_bg.append(nn('Greater', [sbg, 'half'], [f'nbr_is_bg_{k}']))

    # ---- connected components + bbox extremes: Loop-free iterative min-label propagation ----
    # state channels: [flat_label, row, col, -row, -col]; Min propagation doubles as Max for
    # the negated row/col channels, converging bbox extremes alongside the component id.
    state = 'state0'
    for it in range(R_ITERS):
        padded_state = nn('Pad', [state, 'pads_hw', 'sent'], [f'padded_state_it{it}'], mode='constant')
        shifted_states = four_slices(padded_state, f'st_it{it}')
        running = state
        for k in range(4):
            cand = nn('Where', [nbr_is_bg[k], shifted_states[k], 'sent'], [f'cand_it{it}_{k}'])
            running = nn('Min', [running, cand], [f'min_it{it}_{k}'])
        state = running

    # channel splits (label / rmin / cmin / -rmax / -cmax) via explicit start/end constants
    addK('cc0', [0], np.int64); addK('cc1', [1], np.int64); addK('cc2', [2], np.int64)
    addK('cc3', [3], np.int64); addK('cc4', [4], np.int64); addK('cc5', [5], np.int64)
    addK('ax1', [1], np.int64)
    label_ch = nn('Slice', [state, 'cc0', 'cc1', 'ax1'], ['label_ch'])
    rmin_ch = nn('Slice', [state, 'cc1', 'cc2', 'ax1'], ['rmin_ch'])
    cmin_ch = nn('Slice', [state, 'cc2', 'cc3', 'ax1'], ['cmin_ch'])
    negrmax_ch = nn('Slice', [state, 'cc3', 'cc4', 'ax1'], ['negrmax_ch'])
    negcmax_ch = nn('Slice', [state, 'cc4', 'cc5', 'ax1'], ['negcmax_ch'])
    rmax_ch = nn('Neg', [negrmax_ch], ['rmax_ch'])
    cmax_ch = nn('Neg', [negcmax_ch], ['cmax_ch'])

    h_ch = nn('Add', [nn('Sub', [rmax_ch, rmin_ch], ['h_m1']), 'onef'], ['h_ch'])
    w_ch = nn('Add', [nn('Sub', [cmax_ch, cmin_ch], ['w_m1']), 'onef'], ['w_ch'])
    hw_ch = nn('Mul', [h_ch, w_ch], ['hw_ch'])

    # ---- component area via exact pairwise reduction (no Unique/groupby needed) ----
    L_flat = nn('Reshape', [label_ch, 'shape900'], ['L_flat'])
    L_row = nn('Reshape', [L_flat, 'shape900_1'], ['L_row'])
    L_col = nn('Reshape', [L_flat, 'shape1_900'], ['L_col'])
    eq_lbl = nn('Equal', [L_row, L_col], ['eq_lbl'])
    eq_lbl_f = nn('Cast', [eq_lbl], ['eq_lbl_f'], to=F)
    isbg_flat = nn('Reshape', ['is_bg_f', 'shape1_900'], ['isbg_flat'])
    weighted = nn('Mul', [eq_lbl_f, isbg_flat], ['weighted'])
    area_flat = nn('ReduceSum', [weighted], ['area_flat'], axes=[1], keepdims=0)
    area_ch = nn('Reshape', [area_flat, 'shape1_1_30_30'], ['area_ch'])

    filled_bool = nn('Equal', [area_ch, hw_ch], ['filled_bool'])
    more_than_1 = nn('Greater', [area_ch, 'onehalf'], ['more_than_1'])
    is_color3_bool = nn('And', [filled_bool, nn('And', [more_than_1, 'is_bg_bool'], ['fm1'])], ['is_color3_bool'])

    tmp_col = nn('Where', [is_color3_bool, 'col3f', 'col4f'], ['tmp_col'])
    final_idx_f = nn('Where', ['is_bg_bool', tmp_col, 'color_idx_f'], ['final_idx_f'])
    final_idx_sq = nn('Reshape', [final_idx_f, 'shape1_30_30'], ['final_idx_sq'])
    final_idx_i64 = nn('Cast', [final_idx_sq], ['final_idx_i64'], to=I64)

    nn('OneHot', [final_idx_i64, 'depth10', 'oh_vals'], ['oh_raw'], axis=1)

    graph = helper.make_graph(nodes, 'task198', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


def _make_masked():
    return _mask(_make())


model = _bake(_make_masked(), 198)
