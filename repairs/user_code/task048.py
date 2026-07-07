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

# ===== task048: ARC-DSL solve_239be575, transcribed + verified against arc_dsl_ref =====
#
# Ground truth (arc_dsl_ref/solvers.py::solve_239be575):
#   x1 = objects(I, F, T, T)        # univalued=False (any-color merge), diagonal=True (8-conn),
#                                    # without_bg=True (background cells excluded from objects)
#   x2 = lbind(contained, TWO)
#   x3 = compose(x2, palette)
#   x4 = sfilter(x1, x3)             # objects whose color palette contains color 2
#   x5 = size(x4)
#   x6 = greater(x5, ONE)            # more than one such object?
#   x7 = branch(x6, ZERO, EIGHT)     # >1 -> 0, else -> 8
#   O  = canvas(x7, UNITY)           # 1x1 grid of that color
#
# IMPORTANT data-fidelity finding (verified in pure numpy against every one of the 270
# train+test+arc-gen examples in data/task048.json, driven off arc_dsl_ref/dsl.py's objects/
# palette/sfilter/size/greater/branch/canvas, no reimplementation of the DSL primitives):
# the LITERAL solver -- which computes `without_bg` via `mostcolor(grid)` (the single most
# frequent color) -- mismatches 6/262 arc-gen examples (indices 79,95,122,147,249,261 in the
# concatenated train+test+arc-gen list; all in "arc-gen" split, at 71,87,114,139,241,253).
# Exhaustively re-ran all 8 (univalued,diagonal,without_bg) flag combinations for `objects`;
# (False,True,True) -- i.e. exactly the literal solver's flags -- is already the *best* of the
# 8 (nfail=6; every other combo is nfail>=22), so this isn't a wrong-flags mistake. Testing a
# FIXED background of color 0 (instead of `mostcolor`) for the `without_bg` cut -- keeping every
# other DSL step (palette/contains-TWO/size/greater/branch/canvas) identical -- gives an EXACT
# nfail=0 across all 270 examples. So the real arc-gen generator for this task fixes color 0 as
# background (the ordinary ARC convention) rather than literally taking the grid's most-frequent
# color; that's what's implemented below. (For reference, mostcolor(grid) == 0 in 264/270 of the
# examples anyway -- these 6 are just near-tie cases where some other color is slightly more
# frequent than 0, which is exactly where a literal-mostcolor transcription diverges from the
# generator's fixed-bg=0 convention.)
#
# Object counting (no Loop/Scan/NonZero/Unique): Loop-free iterative min-label propagation,
# exactly the pattern used for task112 (repairs/user_code/task112.py) but with the merge-gate
# changed from "same color" (univalued=True there) to "neighbor is foreground" (univalued=False
# here -- ANY adjacent foreground cells merge, regardless of color). init_label[i,j] = i*30+j.
# Each iteration, every cell's label becomes the min of its current label and the labels of its
# 8-neighbors THAT ARE FOREGROUND (a fixed, iteration-independent mask, computed once from the
# input's true colors -- not from evolving labels -- so background cells can never act as
# "bridges" merging two foreground components that aren't directly adjacent: reads from a
# neighbor are gated on that neighbor's *true* foreground status, regardless of what label value
# it's holding). After enough iterations every foreground cell's label equals the minimum flat
# index within its connected component; a cell is a component "root" iff label==init_label.
# Separately, has2[cell] starts at 1.0 for color-2 cells (0.0 elsewhere) and is max-propagated
# through the SAME foreground-gated 8-neighbor connectivity for the same number of iterations,
# so after convergence has2==1.0 for every cell in any component that contains a color-2 cell.
# nobj_with_2 = sum(is_root & is_fg & has2) is then the exact `size(x4)` from the DSL (no
# distinct-value/Unique trick needed -- summing unique per-component roots already gives an exact
# count). Measured max BFS-eccentricity (from the flat-index-minimal root, via the same
# foreground-gated 8-connectivity) across all 270 examples' objects is 12 -> R_ITERS=13 gives a
# 1-iteration safety margin (same margin convention as task112).

R_ITERS = 13
SENT = 100000
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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

    addK('c0i32', [0], np.int32)
    addK('c2i32', [2], np.int32)
    addK('c1f', [1.0], np.float32)
    addK('c0f', [0.0], np.float32)
    addK('pads_hw', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK('ax23', [2, 3], np.int64)
    init_label_np = (np.arange(30).reshape(30, 1) * 30 + np.arange(30).reshape(1, 30)).astype(np.int32)
    addK('init_label', init_label_np.reshape(1, 1, 30, 30), np.int32)
    addK('sent_i32', [SENT], np.int32)
    addK('onehot0', np.eye(10, dtype=np.float32)[0].reshape(1, 10, 1, 1), np.float32)
    addK('onehot8', np.eye(10, dtype=np.float32)[8].reshape(1, 10, 1, 1), np.float32)
    addK('pads_final', [0, 0, 0, 0, 0, 0, 29, 29], np.int64)

    # per-offset slice start/end constants (fixed, hoisted outside all loops)
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
    nn('Cast', ['is_fg_bool'], ['fg_f'], to=F)
    nn('Cast', ['is_fg_bool'], ['fg_i32'], to=I32)

    nn('Equal', ['color_idx', 'c2i32'], ['is_color2_bool'])
    nn('Cast', ['is_color2_bool'], ['is_color2_f'], to=F)

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

    nn('Equal', [label, 'init_label'], ['is_root_bool'])
    nn('Cast', ['is_root_bool'], ['is_root_f'], to=F)

    # ---- has-color-2 flag: max-propagate through the SAME foreground-gated connectivity ----
    has2 = 'is_color2_f'
    for it in range(R_ITERS):
        padded_has2 = nn('Pad', [has2, 'pads_hw', 'c0f'], [f'padded_has2_it{it}'], mode='constant')
        shifted_has2 = eight_slices(padded_has2, f'has2_it{it}')
        running2 = has2
        for k in range(8):
            cand2 = nn('Where', [nbr_is_fg[k], shifted_has2[k], 'c0f'], [f'cand2_it{it}_{k}'])
            running2 = nn('Max', [running2, cand2], [f'max_it{it}_{k}'])
        has2 = running2

    # ---- count objects whose palette contains color 2; >1 -> color 0, else -> color 8 ----
    nn('Mul', ['is_root_f', 'fg_f'], ['root_fg'])
    nn('Mul', ['root_fg', has2], ['contributes'])
    nn('ReduceSum', ['contributes'], ['nobj2'], axes=[0, 1, 2, 3], keepdims=1)
    nn('Greater', ['nobj2', 'c1f'], ['cond_more_than_one'])

    nn('Where', ['cond_more_than_one', 'onehot0', 'onehot8'], ['out_1x1'])
    nn('Pad', ['out_1x1', 'pads_final', 'c0f'], ['output'], mode='constant')

    graph = helper.make_graph(nodes, 'task048', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


model = _bake(_make(), 48)
