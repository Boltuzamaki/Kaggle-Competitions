import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64

# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n, a, d): return numpy_helper.from_array(np.array(a, dtype=d), name=n)
def _rename_output(m, new):
    for nd in m.graph.node:
        for i, o in enumerate(nd.output):
            if o == "output": nd.output[i] = new; return
def _set_out_shape(m, dims):
    tt = m.graph.output[0].type.tensor_type
    tt.elem_type = TensorProto.FLOAT
    del tt.shape.dim[:]
    for d in dims: tt.shape.dim.add().dim_value = d
def _mask(m):
    """Same-shape task: zero the polluted 30x30 border via an input-presence mask."""
    _rename_output(m, "oh_raw")
    m.graph.node.append(helper.make_node("ReduceMax", ["input"], ["presence_m"], axes=[1], keepdims=1))
    m.graph.node.append(helper.make_node("Mul", ["oh_raw", "presence_m"], ["output"]))
    _set_out_shape(m, [1, 10, 30, 30]); return m
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json" % t)
def _reps(t, k=8):
    d = json.load(open(_resolve_task_json(t)))
    exs = sorted(d["train"] + d["test"] + d["arc-gen"], key=lambda e: (len(e["input"]), len(e["input"][0])))
    idx = set([0, len(exs) - 1]) | set(int(j * (len(exs) - 1) / (k - 1)) for j in range(1, k - 1))
    out = []
    for i in sorted(idx):
        g = exs[i]["input"]; a = np.zeros((1, 10, 30, 30), np.float32)
        for r, row in enumerate(g):
            for c, v in enumerate(row): a[0][v][r][c] = 1.0
        out.append(a)
    return out
def _bake(m, t):
    import onnxruntime as _ort
    inf = onnx.shape_inference.infer_shapes(_copy.deepcopy(m), strict_mode=True)
    def sym(vi): return any(dd.HasField("dim_param") or not dd.HasField("dim_value") for dd in vi.type.tensor_type.shape.dim)
    good = set(vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi))
    good |= set(x.name for x in list(m.graph.input) + list(m.graph.output))
    missing = []
    for nd in m.graph.node:
        for o in nd.output:
            if o and o != "output" and o not in good and o not in missing: missing.append(o)
    if not missing: return m
    tmp = _copy.deepcopy(m)
    for nm in missing:
        vi = onnx.ValueInfoProto(); vi.name = nm
        tmp.graph.output.append(vi)
    so = _ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = _ort.InferenceSession(tmp.SerializeToString(), so)
    mx = {}; dt = {}
    for inp in _reps(t):
        for nm, arr in zip(missing, s.run(missing, {"input": inp})):
            sh = list(arr.shape); mx[nm] = [max(a, b) for a, b in zip(mx[nm], sh)] if nm in mx else sh; dt[nm] = arr.dtype
    keep = [vi for vi in m.graph.value_info if vi.name not in missing]
    del m.graph.value_info[:]; m.graph.value_info.extend(keep)
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("bool"): TensorProto.BOOL, np.dtype("int32"): TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m

# ===== task133: ARC-DSL solve_57aa92db (template-stamp-at-seed transformation) =====
#
# Ground truth (arc_dsl_ref/solvers.py::solve_57aa92db) verified EXACT (n_fail=0) on all
# 267 train+test+arc-gen examples in data/task133.json, via two independent from-scratch
# reimplementations (a scipy-connected-components version and a dense-tensor / same-label-
# matrix version using only ops available in ONNX -- see scratchpad proto.py / proto2.py):
#
#   1. bg is always 0 (verified). x1 = 8-connected (king-move) multicolor foreground
#      components; x8 = the one maximizing (max colorcount - min colorcount) among its own
#      palette -- this is the "template" (a small multicolor blob with one rare "marker"
#      color, verified to appear as EXACTLY one cell in the template in all 267 cases).
#   2. x9 = that rare marker color. Every other same-color (x9) 4-connected component
#      elsewhere in the grid is a "seed": verified to always be a SOLID w x w square
#      (never any other shape). Its surrounding-ring ("outbox") in the original grid has
#      exactly one non-background color (verified) -- the seed's context/paint color.
#   3. For each seed: scale a copy of the template's own shape (its literal foreground
#      cells, not its filled bounding box) by factor=w, position it so the (scaled) marker
#      cell block lands exactly on the seed square, and recolor every stamped cell to the
#      seed's context color. Exact placement formula (independently re-derived and verified
#      against the literal DSL chain, since a naive reading of the DSL's `upscale` composition
#      is subtly wrong when the marker isn't at the template's own corner -- see stamp_check.py):
#         for template cell offset (ti,tj) from the template's own upper-left corner, and
#         marker offset (mi,mj) = marker's own (ti,tj):
#           out_row = SeedTop + (ti-mi)*factor + io   (io in [0,factor))
#           out_col = SeedLeft + (tj-mj)*factor + jo   (jo in [0,factor))
#      Zero conflicts found between different seeds' stamped regions across the dataset.
#   4. Composite order: paint stamps onto the original grid, then restore the ORIGINAL
#      grid's foreground on top (paint(x27, merge(x2)) in the DSL) -- equivalent to
#      final = where(original_is_foreground, original_color, stamped_or_original).
#
# ONNX translation (no Loop/Scan/NonZero/Unique/Compress, opset 12/13, static 30x30):
#  - Both connected-component passes use the Loop-free iterative min-label-propagation +
#    NxN same-component matrix pattern established in repairs/user_code/task364.py /
#    task046.py (900x900 boolean same-label matrices over the flattened 30x30 grid).
#    R_ITERS chosen with margin over the measured max BFS-eccentricity across the whole
#    dataset (8-conn over all foreground: measured max 7, R_ITERS1=10; 4-conn restricted to
#    color==x9 cells: measured max 6, R_ITERS2=9).
#  - Per-component per-color pixel counts (needed for the argmax-spread template search)
#    computed via 9 separate masked-sum reductions (colors 1..9) over the 900x900 same-
#    label(8-conn) matrix; spread = elementwise max-across-colors minus min-across-colors-
#    that-are-actually-present (zero-count colors excluded from the min via a BIG-sentinel
#    Where, matching the task046/task364 "Where bool, real_value, sentinel, then Reduce"
#    idiom used throughout this codebase). Verified no ties in the argmax across all 267
#    examples (each example has a unique maximal-spread component).
#  - x9 (rarest color in the template) found via the same per-color-stacking + ArgMin idiom
#    as repairs/user_code/task049.py's adjacency-color search.
#  - Up to 4 seeds are selected via TopK (K=4, smallest key, largest=0) over a per-cell key
#    that is the seed component's root flat-index when it is a genuine seed (root of an
#    x9-colored 4-connected component that is NOT the template's own single marker cell) and
#    a BIG sentinel otherwise -- exactly the task046 pattern, including its required
#    exists-gate (val < threshold) so that "phantom" slots beyond the true seed count (whose
#    TopK index can otherwise land on a real object's non-root cell) contribute nothing.
#  - Each selected seed's context color is read via a ring-shaped boolean mask built purely
#    from broadcast comparisons of the fixed 30x30 row/col index grids against the seed's
#    (data-dependent) bounding box +/- 1 -- no dynamic-shape slicing needed, since cells
#    outside the true grid extent are already background(0) in the zero-padded 30x30 tensor
#    and therefore silently contribute nothing to the ReduceMax lookup either way.
#  - Stamp placement avoids enumerating template cells: for every output cell (R,C) and
#    every candidate seed, the corresponding template-space coordinate is computed via
#    Floor-based floor-division (float Div + Floor, correct for negative offsets, since
#    ONNX Div truncates toward zero) and looked up in the (already-computed, dense 30x30)
#    template-membership mask with GatherND -- separable row-only and col-only index
#    vectors are outer-broadcast into a (30,30,2) index tensor via Expand+Unsqueeze+Concat.
#    Out-of-[0,29]-range lookups are computed from the UNCLIPPED coordinate (before Clip,
#    which is only applied to keep GatherND's indices valid) so they can be explicitly
#    masked out rather than silently aliasing a real template cell at the boundary.
#  - Final compositing (paint stamps, then restore original foreground on top) plus the
#    standard same-shape `_mask` helper (copied verbatim from repairs/user_code/task049.py)
#    for zeroing the polluted 30x30 border naturally also clips any stamp geometry that
#    would have bled past the TRUE (non-padded) grid extent, matching the DSL's own
#    bounds-checked `paint`/`toobject`.

R_ITERS1 = 10   # 8-connected (all-foreground) label propagation -- measured max ecc 7
R_ITERS2 = 9    # 4-connected (single-color x9) label propagation -- measured max ecc 6
MAXSEEDS = 4    # measured max seed count across dataset
BIG = 1.0e6
NEGBIG = -1.0e6
DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _make():
    inits, nodes = [], []

    def addK(n, a, d):
        inits.append(_K(n, a, d))
        return n

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    # ---- constants ----
    addK('ax1', [1], np.int64)
    addK('ax23', [2, 3], np.int64)
    addK('c0f', [0.0], np.float32)
    addK('c1f', [1.0], np.float32)
    addK('c29f', [29.0], np.float32)
    addK('m1f', [-1.0], np.float32)
    addK('bigf', [BIG], np.float32)
    addK('negbigf', [NEGBIG], np.float32)
    addK('threshf', [BIG - 1.0], np.float32)
    addK('pads4', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK('shape900', [900], np.int64)
    addK('shape900_1', [900, 1], np.int64)
    addK('shape1_900', [1, 900], np.int64)
    addK('shape1', [1], np.int64)
    addK('shape1111', [1, 1, 1, 1], np.int64)
    addK('shape30_1', [30, 1], np.int64)
    addK('shape1_30', [1, 30], np.int64)
    addK('shape_30_30', [30, 30], np.int64)
    addK('shape_1_1_30_30', [1, 1, 30, 30], np.int64)
    addK('k4', [4], np.int64)
    addK('depth10', [10], np.int64)
    addK('oh_vals', [0.0, 1.0], np.float32)

    row_flat_np = np.repeat(np.arange(30), 30).astype(np.float32)
    col_flat_np = np.tile(np.arange(30), 30).astype(np.float32)
    init_label_np = np.arange(900).astype(np.float32).reshape(1, 1, 30, 30)
    init_label_flat_np = np.arange(900).astype(np.float32)
    row_grid_np = row_flat_np.reshape(1, 1, 30, 30)
    col_grid_np = col_flat_np.reshape(1, 1, 30, 30)
    row_const_np = np.arange(30).astype(np.float32)
    col_const_np = np.arange(30).astype(np.float32)

    addK('row_flat_c', row_flat_np, np.float32)
    addK('col_flat_c', col_flat_np, np.float32)
    addK('init_label_grid', init_label_np, np.float32)
    addK('init_label_flat', init_label_flat_np, np.float32)
    addK('row_grid_c', row_grid_np, np.float32)
    addK('col_grid_c', col_grid_np, np.float32)
    addK('row_const', row_const_np, np.float32)
    addK('col_const', col_const_np, np.float32)

    for c in range(1, 10):
        addK('colv%d' % c, [float(c)], np.float32)

    # ---- shift helper (30x30 grid, pad 1 each side; works for any |di|,|dj|<=1) ----
    def shiftN(src, prefix, pad_value_name, dirs):
        padded = nn('Pad', [src, 'pads4', pad_value_name], ['%s_padded' % prefix], mode='constant')
        outs = []
        for k, (di, dj) in enumerate(dirs):
            sh = addK('%s_st%d' % (prefix, k), [1 + di, 1 + dj], np.int64)
            eh = addK('%s_en%d' % (prefix, k), [1 + di + 30, 1 + dj + 30], np.int64)
            o = nn('Slice', [padded, sh, eh, 'ax23'], ['%s_sl%d' % (prefix, k)])
            outs.append(o)
        return outs

    def label_prop(gate_grid, dirs, iters, prefix):
        """Loop-free min-label propagation restricted to gate_grid (both endpoints must satisfy it)."""
        gate_shift = shiftN(gate_grid, prefix + '_gs', 'c0f', dirs)
        gates = []
        for k in range(len(dirs)):
            gb = nn('Greater', [gate_shift[k], 'c0f'], ['%s_gb_%d' % (prefix, k)])
            gk = nn('And', [gate_bool_of[prefix], gb], ['%s_gate_%d' % (prefix, k)])
            gates.append(gk)
        label = 'init_label_grid'
        for it in range(iters):
            shifted = shiftN(label, '%s_it%d' % (prefix, it), 'bigf', dirs)
            cur = label
            for k in range(len(dirs)):
                cand = nn('Where', [gates[k], shifted[k], 'bigf'], ['%s_cand_%d_%d' % (prefix, it, k)])
                cur = nn('Min', [cur, cand], ['%s_min_%d_%d' % (prefix, it, k)])
            label = cur
        return label

    # gate_bool_of stores the boolean gate-grid tensor name for each label_prop call
    gate_bool_of = {}

    # ---- color grid & foreground mask ----
    nn('ArgMax', ['input'], ['color_idx64'], axis=1, keepdims=1)
    nn('Cast', ['color_idx64'], ['color_f'], to=F)
    fg_bool = nn('Greater', ['color_f', 'c0f'], ['fg_bool'])
    fg_f = nn('Cast', [fg_bool], ['fg_f'], to=F)

    # ==== pass 1: 8-connected multicolor components over ALL foreground ====
    gate_bool_of['p1'] = fg_bool
    label1 = label_prop('fg_f', DIRS8, R_ITERS1, 'p1')

    label1_flat = nn('Reshape', [label1, 'shape900'], ['label1_flat'])
    fg_flat = nn('Reshape', [fg_bool, 'shape900'], ['fg_flat'])
    color_flat = nn('Reshape', ['color_f', 'shape900'], ['color_flat'])

    label1_col = nn('Reshape', [label1_flat, 'shape900_1'], ['label1_col'])
    label1_row = nn('Reshape', [label1_flat, 'shape1_900'], ['label1_row'])
    fg_col = nn('Reshape', [fg_flat, 'shape900_1'], ['fg_col'])
    fg_row = nn('Reshape', [fg_flat, 'shape1_900'], ['fg_row'])
    color_row = nn('Reshape', [color_flat, 'shape1_900'], ['color_row'])

    lab1_eq = nn('Equal', [label1_col, label1_row], ['lab1_eq'])
    same1_0 = nn('And', [lab1_eq, fg_col], ['same1_0'])
    same_label1 = nn('And', [same1_0, fg_row], ['same_label1'])  # (900,900)

    count_c = {}
    for c in range(1, 10):
        is_c_row = nn('Equal', [color_row, 'colv%d' % c], ['is_c_row_%d' % c])
        combined = nn('And', [same_label1, is_c_row], ['combined_%d' % c])
        combined_f = nn('Cast', [combined], ['combined_f_%d' % c], to=F)
        count_c[c] = nn('ReduceSum', [combined_f], ['count_c_%d' % c], axes=[1], keepdims=0)  # (900,)

    max_count = nn('Max', [count_c[c] for c in range(1, 10)], ['max_count'])
    min_cands = []
    for c in range(1, 10):
        gtz = nn('Greater', [count_c[c], 'c0f'], ['gtz_%d' % c])
        mc_ = nn('Where', [gtz, count_c[c], 'bigf'], ['min_cand_%d' % c])
        min_cands.append(mc_)
    min_count = nn('Min', min_cands, ['min_count'])
    spread = nn('Sub', [max_count, min_count], ['spread'])  # (900,)

    spread_gate = nn('Where', [fg_flat, spread, 'negbigf'], ['spread_gate'])
    global_max_spread = nn('ReduceMax', [spread_gate], ['global_max_spread'], axes=[0], keepdims=0)
    tmpl_eq = nn('Equal', [spread, global_max_spread], ['tmpl_eq'])
    template_mask_flat = nn('And', [tmpl_eq, fg_flat], ['template_mask_flat'])  # (900,)
    template_mask_f = nn('Cast', [template_mask_flat], ['template_mask_f'], to=F)

    Ttop_pre = nn('Where', [template_mask_flat, 'row_flat_c', 'bigf'], ['Ttop_pre'])
    Ttop = nn('ReduceMin', [Ttop_pre], ['Ttop'], axes=[0], keepdims=0)
    Tleft_pre = nn('Where', [template_mask_flat, 'col_flat_c', 'bigf'], ['Tleft_pre'])
    Tleft = nn('ReduceMin', [Tleft_pre], ['Tleft'], axes=[0], keepdims=0)

    # ---- x9: rarest color present in the template ----
    adj = []
    for c in range(1, 10):
        tc_ = nn('Mul', [count_c[c], template_mask_f], ['tmpl_count_pre_%d' % c])
        tcount = nn('ReduceMax', [tc_], ['tmpl_count_%d' % c], axes=[0], keepdims=0)
        gtz = nn('Greater', [tcount, 'c0f'], ['tmpl_gtz_%d' % c])
        adjv = nn('Where', [gtz, tcount, 'bigf'], ['tmpl_adj_%d' % c])
        adj1 = nn('Reshape', [adjv, 'shape1'], ['tmpl_adj1_%d' % c])
        adj.append(adj1)
    stacked = nn('Concat', adj, ['stacked_tmplcolors'], axis=0)  # (9,)
    xidx = nn('ArgMin', [stacked], ['x9_idx0'], axis=0, keepdims=1)  # (1,) int64, 0-based over colors 1..9
    xidx_f = nn('Cast', [xidx], ['x9_idx_f'], to=F)
    x9_f = nn('Add', [xidx_f, 'c1f'], ['x9_f'])  # (1,) float color value

    marker_eq = nn('Equal', [color_flat, x9_f], ['marker_eq'])
    marker_mask_flat = nn('And', [template_mask_flat, marker_eq], ['marker_mask_flat'])  # (900,)

    mr_pre = nn('Where', [marker_mask_flat, 'row_flat_c', 'm1f'], ['mr_pre'])
    mr = nn('ReduceMax', [mr_pre], ['mr'], axes=[0], keepdims=0)
    mc_pre = nn('Where', [marker_mask_flat, 'col_flat_c', 'm1f'], ['mc_pre'])
    mc = nn('ReduceMax', [mc_pre], ['mc'], axes=[0], keepdims=0)
    x13_row = nn('Sub', [mr, Ttop], ['x13_row'])
    x13_col = nn('Sub', [mc, Tleft], ['x13_col'])

    # ==== pass 2: 4-connected components restricted to color==x9 ====
    is_x9_flat = nn('Equal', [color_flat, x9_f], ['is_x9_flat'])
    is_x9_grid = nn('Reshape', [is_x9_flat, 'shape_1_1_30_30'], ['is_x9_grid'])
    is_x9_f = nn('Cast', [is_x9_grid], ['is_x9_f'], to=F)
    gate_bool_of['p2'] = is_x9_grid
    label_seed = label_prop('is_x9_f', DIRS4, R_ITERS2, 'p2')
    label_seed_flat = nn('Reshape', [label_seed, 'shape900'], ['label_seed_flat'])

    ls_col = nn('Reshape', [label_seed_flat, 'shape900_1'], ['ls_col'])
    ls_row = nn('Reshape', [label_seed_flat, 'shape1_900'], ['ls_row'])
    x9_col = nn('Reshape', [is_x9_flat, 'shape900_1'], ['x9_col'])
    x9_row = nn('Reshape', [is_x9_flat, 'shape1_900'], ['x9_row'])
    ls_eq = nn('Equal', [ls_col, ls_row], ['ls_eq'])
    same2_0 = nn('And', [ls_eq, x9_col], ['same2_0'])
    same_label_seed = nn('And', [same2_0, x9_row], ['same_label_seed'])  # (900,900)

    row_row = nn('Reshape', ['row_flat_c', 'shape1_900'], ['row_row'])
    col_row = nn('Reshape', ['col_flat_c', 'shape1_900'], ['col_row'])
    rmin_pre = nn('Where', [same_label_seed, row_row, 'bigf'], ['rmin_pre'])
    rmin_seed = nn('ReduceMin', [rmin_pre], ['rmin_seed'], axes=[1], keepdims=0)
    rmax_pre = nn('Where', [same_label_seed, row_row, 'negbigf'], ['rmax_pre'])
    rmax_seed = nn('ReduceMax', [rmax_pre], ['rmax_seed'], axes=[1], keepdims=0)
    cmin_pre = nn('Where', [same_label_seed, col_row, 'bigf'], ['cmin_pre'])
    cmin_seed = nn('ReduceMin', [cmin_pre], ['cmin_seed'], axes=[1], keepdims=0)
    cmax_pre = nn('Where', [same_label_seed, col_row, 'negbigf'], ['cmax_pre'])
    cmax_seed = nn('ReduceMax', [cmax_pre], ['cmax_seed'], axes=[1], keepdims=0)
    factor_seed0 = nn('Sub', [rmax_seed, rmin_seed], ['factor_seed0'])
    factor_seed = nn('Add', [factor_seed0, 'c1f'], ['factor_seed'])  # (900,)

    marker_seed_label_pre = nn('Where', [marker_mask_flat, label_seed_flat, 'm1f'], ['marker_seed_label_pre'])
    marker_seed_label = nn('ReduceMax', [marker_seed_label_pre], ['marker_seed_label'], axes=[0], keepdims=0)

    is_root_seed0 = nn('Equal', [label_seed_flat, 'init_label_flat'], ['is_root_seed0'])
    is_root_seed = nn('And', [is_root_seed0, is_x9_flat], ['is_root_seed'])
    is_marker_comp = nn('Equal', [label_seed_flat, marker_seed_label], ['is_marker_comp'])
    not_marker_comp = nn('Not', [is_marker_comp], ['not_marker_comp'])
    is_real_seed = nn('And', [is_root_seed, not_marker_comp], ['is_real_seed'])
    key_flat = nn('Where', [is_real_seed, 'init_label_flat', 'bigf'], ['key_flat'])
    key_2d = nn('Reshape', [key_flat, 'shape1_900'], ['key_2d'])
    vals_topk, idx_topk = nn('TopK', [key_2d, 'k4'], ['vals_topk', 'idx_topk'], axis=1, largest=0, sorted=1)

    # (reshape TopK outputs (1,4) -> (4,) for per-slot Slice)
    addK('shape4', [4], np.int64)
    vals4 = nn('Reshape', [vals_topk, 'shape4'], ['vals4'])
    idxs4 = nn('Reshape', [idx_topk, 'shape4'], ['idxs4'])

    canvas = None
    addK('ax0', [0], np.int64)

    for p in range(MAXSEEDS):
        pidx = addK('pidx_%d' % p, [p], np.int64)
        pidx1 = addK('pidx1_%d' % p, [p + 1], np.int64)
        idx_p = nn('Slice', [idxs4, pidx, pidx1, 'ax0'], ['idx_p_%d' % p])          # (1,) int64
        val_p = nn('Slice', [vals4, pidx, pidx1, 'ax0'], ['val_p_%d' % p])          # (1,) float
        exists_bool = nn('Less', [val_p, 'threshf'], ['exists_bool_%d' % p])
        exists_f = nn('Cast', [exists_bool], ['exists_f_%d' % p], to=F)

        SeedTop = nn('Gather', [rmin_seed, idx_p], ['SeedTop_%d' % p], axis=0)      # (1,)
        SeedBot = nn('Gather', [rmax_seed, idx_p], ['SeedBot_%d' % p], axis=0)
        SeedLeft = nn('Gather', [cmin_seed, idx_p], ['SeedLeft_%d' % p], axis=0)
        SeedRight = nn('Gather', [cmax_seed, idx_p], ['SeedRight_%d' % p], axis=0)
        factor_p = nn('Gather', [factor_seed, idx_p], ['factor_p_%d' % p], axis=0)

        SeedTop4 = nn('Reshape', [SeedTop, 'shape1111'], ['SeedTop4_%d' % p])
        SeedBot4 = nn('Reshape', [SeedBot, 'shape1111'], ['SeedBot4_%d' % p])
        SeedLeft4 = nn('Reshape', [SeedLeft, 'shape1111'], ['SeedLeft4_%d' % p])
        SeedRight4 = nn('Reshape', [SeedRight, 'shape1111'], ['SeedRight4_%d' % p])

        rTm1 = nn('Sub', [SeedTop4, 'c1f'], ['rTm1_%d' % p])
        rBp1 = nn('Add', [SeedBot4, 'c1f'], ['rBp1_%d' % p])
        cLm1 = nn('Sub', [SeedLeft4, 'c1f'], ['cLm1_%d' % p])
        cRp1 = nn('Add', [SeedRight4, 'c1f'], ['cRp1_%d' % p])

        eq_top = nn('Equal', ['row_grid_c', rTm1], ['eq_top_%d' % p])
        eq_bot = nn('Equal', ['row_grid_c', rBp1], ['eq_bot_%d' % p])
        row_edge = nn('Or', [eq_top, eq_bot], ['row_edge_%d' % p])
        col_ge = nn('GreaterOrEqual', ['col_grid_c', cLm1], ['col_ge_%d' % p])
        col_le = nn('LessOrEqual', ['col_grid_c', cRp1], ['col_le_%d' % p])
        col_in = nn('And', [col_ge, col_le], ['col_in_%d' % p])
        band_h = nn('And', [row_edge, col_in], ['band_h_%d' % p])

        eq_left = nn('Equal', ['col_grid_c', cLm1], ['eq_left_%d' % p])
        eq_right = nn('Equal', ['col_grid_c', cRp1], ['eq_right_%d' % p])
        col_edge = nn('Or', [eq_left, eq_right], ['col_edge_%d' % p])
        row_ge = nn('GreaterOrEqual', ['row_grid_c', rTm1], ['row_ge_%d' % p])
        row_le = nn('LessOrEqual', ['row_grid_c', rBp1], ['row_le_%d' % p])
        row_in = nn('And', [row_ge, row_le], ['row_in_%d' % p])
        band_v = nn('And', [col_edge, row_in], ['band_v_%d' % p])

        ring = nn('Or', [band_h, band_v], ['ring_%d' % p])
        ring_f = nn('Cast', [ring], ['ring_f_%d' % p], to=F)
        ring_colorvals = nn('Mul', ['color_f', ring_f], ['ring_colorvals_%d' % p])
        ctxcolor = nn('ReduceMax', [ring_colorvals], ['ctxcolor_%d' % p], axes=[0, 1, 2, 3], keepdims=0)

        # per-row / per-col template-space coordinate via floor division
        rowdiff = nn('Sub', ['row_const', SeedTop], ['rowdiff_%d' % p])
        rel_i = nn('Floor', [nn('Div', [rowdiff, factor_p], ['rel_i_div_%d' % p])], ['rel_i_%d' % p])
        tr0 = nn('Add', [Ttop, x13_row], ['tr0_%d' % p])
        tr = nn('Add', [tr0, rel_i], ['tr_%d' % p])   # (30,)
        tr_ge = nn('GreaterOrEqual', [tr, 'c0f'], ['tr_ge_%d' % p])
        tr_le = nn('LessOrEqual', [tr, 'c29f'], ['tr_le_%d' % p])
        valid_r = nn('And', [tr_ge, tr_le], ['valid_r_%d' % p])
        tr_clip = nn('Clip', [tr, 'c0f', 'c29f'], ['tr_clip_%d' % p])
        tr_idx = nn('Cast', [tr_clip], ['tr_idx_%d' % p], to=I64)

        coldiff = nn('Sub', ['col_const', SeedLeft], ['coldiff_%d' % p])
        rel_j = nn('Floor', [nn('Div', [coldiff, factor_p], ['rel_j_div_%d' % p])], ['rel_j_%d' % p])
        tc0 = nn('Add', [Tleft, x13_col], ['tc0_%d' % p])
        tcc = nn('Add', [tc0, rel_j], ['tc_%d' % p])   # (30,)
        tc_ge = nn('GreaterOrEqual', [tcc, 'c0f'], ['tc_ge_%d' % p])
        tc_le = nn('LessOrEqual', [tcc, 'c29f'], ['tc_le_%d' % p])
        valid_c = nn('And', [tc_ge, tc_le], ['valid_c_%d' % p])
        tc_clip = nn('Clip', [tcc, 'c0f', 'c29f'], ['tc_clip_%d' % p])
        tc_idx = nn('Cast', [tc_clip], ['tc_idx_%d' % p], to=I64)

        tr_idx_col = nn('Reshape', [tr_idx, 'shape30_1'], ['tr_idx_col_%d' % p])
        tc_idx_row = nn('Reshape', [tc_idx, 'shape1_30'], ['tc_idx_row_%d' % p])
        tr_idx_2d = nn('Expand', [tr_idx_col, 'shape_30_30'], ['tr_idx_2d_%d' % p])
        tc_idx_2d = nn('Expand', [tc_idx_row, 'shape_30_30'], ['tc_idx_2d_%d' % p])
        tr_idx_3d = nn('Unsqueeze', [tr_idx_2d], ['tr_idx_3d_%d' % p], axes=[2])
        tc_idx_3d = nn('Unsqueeze', [tc_idx_2d], ['tc_idx_3d_%d' % p], axes=[2])
        gnd_indices = nn('Concat', [tr_idx_3d, tc_idx_3d], ['gnd_indices_%d' % p], axis=2)

        template_mask_2d = nn('Reshape', [template_mask_f, 'shape_30_30'], ['template_mask_2d_%d' % p]) if p == 0 else 'template_mask_2d_0'
        gathered = nn('GatherND', [template_mask_2d, gnd_indices], ['gathered_%d' % p], batch_dims=0)

        valid_r_col = nn('Reshape', [valid_r, 'shape30_1'], ['valid_r_col_%d' % p])
        valid_c_row = nn('Reshape', [valid_c, 'shape1_30'], ['valid_c_row_%d' % p])
        valid2d = nn('And', [valid_r_col, valid_c_row], ['valid2d_%d' % p])
        valid2d_f = nn('Cast', [valid2d], ['valid2d_f_%d' % p], to=F)

        slot_mask_f = nn('Mul', [gathered, valid2d_f], ['slot_mask_f_%d' % p])
        slot_mask_f2 = nn('Mul', [slot_mask_f, exists_f], ['slot_mask_f2_%d' % p])
        slot_color2d = nn('Mul', [slot_mask_f2, ctxcolor], ['slot_color2d_%d' % p])
        slot_color4d = nn('Reshape', [slot_color2d, 'shape_1_1_30_30'], ['slot_color4d_%d' % p])

        canvas = slot_color4d if canvas is None else nn('Max', [canvas, slot_color4d], ['canvas_%d' % p])

    final_stamped = canvas
    is_stamped = nn('Greater', [final_stamped, 'c0f'], ['is_stamped'])
    painted = nn('Where', [is_stamped, final_stamped, 'color_f'], ['painted'])
    final_out = nn('Where', [fg_bool, 'color_f', painted], ['final_out'])

    idx64 = nn('Cast', [final_out], ['idx64_pre'], to=I64)
    idx_sq = nn('Squeeze', [idx64], ['idx_sq'], axes=[1])
    nn('OneHot', [idx_sq, 'depth10', 'oh_vals'], ['output'], axis=1)

    graph = helper.make_graph(nodes, 'task133', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


def build_model():
    return _mask(_make())


model = _bake(build_model(), 133)
