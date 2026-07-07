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

# ===== task054: ARC-DSL solve_264363fd, transcribed + verified against arc_dsl_ref =====
#
# Ground truth intent (arc_dsl_ref/solvers.py::solve_264363fd): find the smallest
# (4-connected, multicolor, background-excluded) object in the grid -- a small
# "key" marker whose bounding box is 5-tall and/or 5-wide. Its center cell's
# color is the "anchor color"; the cell adjacent to the center (UP if the key is
# 5-tall, else RIGHT) gives the "line color". Remove the key object (fill with
# background). In the remaining grid, every other object that contains a cell of
# the anchor color gets a vertical and/or horizontal frontier line (colored with
# the line color, spanning that object's own bounding box) drawn through every
# such cell -- vertical if height==5, horizontal if width==5, both if neither
# (matching the literal branch/branch/combine construction in the DSL, verified
# to reproduce all 266 train+test+arc-gen examples exactly). Separately, the
# key's own shape (recentered on its anchor-color cell) is re-stamped, in its
# original colors, on top of every occurrence of the anchor color anywhere in
# the (key-removed) grid. Finally, every cell that was background in the
# key-removed grid is forced back to background (undoes any line/stamp overflow
# outside real object pixels).
#
# Verified in pure numpy (independent from-scratch reconstruction, no DSL
# interpreter reuse) against every one of the 266 train+test+arc-gen examples in
# data/task054.json: n_fail==0. Max key-object bounding box measured 5x5 (never
# larger), max connected-component BFS eccentricity from its flat-index-minimal
# root measured 31 (across all examples) -- R_ITERS=34 gives a safety margin.
# No ties were ever observed for "smallest object by cell count".
#
# ONNX translation (no Loop/Scan/NonZero/Unique/Compress):
#  - Connected components (4-connectivity): Loop-free iterative min-label
#    propagation (same established pattern as repairs/user_code/task048.py /
#    task112.py), gated on each neighbor's FIXED true-foreground status (never
#    the evolving label), so background cells can never bridge two components.
#  - Per-object aggregates (cell count for "smallest object"; bounding-box
#    min/max row/col for "big object" frontier lines) computed via a single
#    (900,900) pairwise label-equality matrix (labels are unique per component
#    after convergence) reduced with Sum/Min/Max -- avoids a second propagation
#    pass entirely (bg2==bg and non-key components' labels are unaffected by
#    removing the key, verified over all 266 examples, so the ORIGINAL grid's
#    connected-component labels are reused for the key-removed grid's objects
#    too, just re-masked to exclude the key's own cells).
#  - Frontier-line placement and the key-shape re-stamp are both expressed as
#    static shift-by-fixed-offset (Pad+Slice) + dynamic-value GatherND lookups
#    (index values are runtime-computed, but every shape is static at trace
#    time) -- no dynamic-shape ops, so _bake() below is a no-op safety net.

R_ITERS = 34
OFFS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STAMP_OFFS = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3)]


def build_054():
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

    addK('c0_f', [0.0], np.float32)
    addK('half_f', [0.5], np.float32)
    addK('c5_i64', [5], np.int64)
    addK('c1_i64', [1], np.int64)
    addK('c2_i64', [2], np.int64)
    addK('c0_i64', [0], np.int64)
    addK('c29_i64', [29], np.int64)
    addK('cm1_i64', [-1], np.int64)
    addK('sent_big_i64', [90000], np.int64)
    addK('sent_big_f', [1.0e6], np.float32)
    addK('depth10', [10], np.int64)
    addK('oh_vals', [0.0, 1.0], np.float32)
    addK('shape900', [900], np.int64)
    addK('shape30x30', [30, 30], np.int64)
    addK('shape1x900', [1, 900], np.int64)
    addK('shape900x1', [900, 1], np.int64)
    addK('shape1', [-1], np.int64)
    addK('shape1x2', [1, 2], np.int64)
    addK('shape1x1x30x30', [1, 1, 30, 30], np.int64)
    addK('shape1x1x1x1', [1, 1, 1, 1], np.int64)
    addK('shape1x30x30', [1, 30, 30], np.int64)
    addK('pads1_hw', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    addK('pads2_hw', [0, 0, 2, 2, 0, 0, 2, 2], np.int64)
    addK('ax23', [2, 3], np.int64)

    init_label_np = (np.arange(30).reshape(30, 1) * 30 + np.arange(30).reshape(1, 30)).astype(np.int64)
    addK('init_label', init_label_np.reshape(1, 1, 30, 30), np.int64)
    row_idx_np = (np.arange(30).reshape(1, 1, 30, 1) * np.ones((1, 1, 1, 30), dtype=np.int64))
    col_idx_np = (np.arange(30).reshape(1, 1, 1, 30) * np.ones((1, 1, 30, 1), dtype=np.int64))
    addK('row_idx_grid', row_idx_np, np.int64)
    addK('col_idx_grid', col_idx_np, np.int64)

    for dv in range(-2, 3):
        addK(f'off_r_{dv}', [dv], np.int64)
        addK(f'off_c_{dv}', [dv], np.int64)

    starts1 = []
    ends1 = []
    for k, (di, dj) in enumerate(OFFS4):
        s = addK(f'st1_{k}', [1 + di, 1 + dj], np.int64)
        e = addK(f'en1_{k}', [31 + di, 31 + dj], np.int64)
        starts1.append(s)
        ends1.append(e)

    def four_slices(padded_name, prefix, starts, ends):
        outs = []
        for k in range(4):
            oname = f'{prefix}_sl{k}'
            nn('Slice', [padded_name, starts[k], ends[k], 'ax23'], [oname])
            outs.append(oname)
        return outs

    for (dr, dc) in STAMP_OFFS:
        addK(f'st2_{dr}_{dc}', [2 - dr, 2 - dc], np.int64)
        addK(f'en2_{dr}_{dc}', [32 - dr, 32 - dc], np.int64)

    # ---- color / background / foreground ----
    nn('ArgMax', ['input'], ['color_idx'], axis=1, keepdims=1)
    nn('ReduceSum', ['input'], ['counts'], axes=[2, 3], keepdims=1)
    nn('ArgMax', ['counts'], ['bg_idx'], axis=1, keepdims=1)

    nn('ReduceMax', ['input'], ['presence_f'], axes=[1], keepdims=1)
    nn('Greater', ['presence_f', 'c0_f'], ['presence_bool'])

    nn('Equal', ['color_idx', 'bg_idx'], ['is_bg_bool'])
    nn('Not', ['is_bg_bool'], ['not_bg_bool'])
    nn('And', ['not_bg_bool', 'presence_bool'], ['fg1_bool'])
    nn('Cast', ['fg1_bool'], ['fg1_f'], to=F)
    nn('Cast', ['fg1_bool'], ['fg1_i64'], to=I64)

    nn('Pad', ['fg1_i64', 'pads1_hw'], ['padded_fg_1'], mode='constant')
    nbr_fg_shift = four_slices('padded_fg_1', 'nbrfg', starts1, ends1)
    nbr_is_fg = []
    for k, sfg in enumerate(nbr_fg_shift):
        nbr_is_fg.append(nn('Cast', [sfg], [f'nbr_is_fg_{k}'], to=BOOL))

    # ---- connected components (4-conn): Loop-free min-label propagation ----
    label = 'init_label'
    for it in range(R_ITERS):
        padded_label = nn('Pad', [label, 'pads1_hw', 'sent_big_i64'], [f'padlab_it{it}'], mode='constant')
        shifted = four_slices(padded_label, f'lab_it{it}', starts1, ends1)
        running = label
        for k in range(4):
            cand = nn('Where', [nbr_is_fg[k], shifted[k], 'sent_big_i64'], [f'cand_it{it}_{k}'])
            running = nn('Min', [running, cand], [f'minlab_it{it}_{k}'])
        label = running

    # ---- pairwise (900,900) label-equality matrix ----
    nn('Reshape', [label, 'shape900'], ['label_flat'])
    nn('Reshape', ['fg1_f', 'shape900'], ['fg1_flat'])
    nn('Reshape', ['row_idx_grid', 'shape900'], ['row_flat'])
    nn('Reshape', ['col_idx_grid', 'shape900'], ['col_flat'])
    nn('Reshape', ['color_idx', 'shape900'], ['color_flat'])

    nn('Reshape', ['label_flat', 'shape900x1'], ['label_col'])
    nn('Reshape', ['label_flat', 'shape1x900'], ['label_row'])
    nn('Equal', ['label_col', 'label_row'], ['EQ1_bool'])
    nn('Cast', ['EQ1_bool'], ['EQ1_f'], to=F)

    nn('Reshape', ['fg1_flat', 'shape1x900'], ['fg1_flat_row'])
    nn('Mul', ['EQ1_f', 'fg1_flat_row'], ['EQ1_masked_fg1'])
    nn('ReduceSum', ['EQ1_masked_fg1'], ['size1_flat'], axes=[1], keepdims=0)

    nn('Reshape', ['size1_flat', 'shape1x1x30x30'], ['size1_grid'])
    nn('Where', ['fg1_bool', 'size1_grid', 'sent_big_f'], ['masked_size'])
    nn('ReduceMin', ['masked_size'], ['min_size'], axes=[0, 1, 2, 3], keepdims=1)

    nn('Equal', ['size1_grid', 'min_size'], ['is_min_size_bool'])
    nn('And', ['fg1_bool', 'is_min_size_bool'], ['key_mask_bool'])
    nn('Cast', ['key_mask_bool'], ['key_mask_f'], to=F)

    # ---- key object bbox ----
    nn('Where', ['key_mask_bool', 'row_idx_grid', 'sent_big_i64'], ['rowk_min_in'])
    nn('ReduceMin', ['rowk_min_in'], ['r0'], axes=[0, 1, 2, 3], keepdims=1)
    nn('Where', ['key_mask_bool', 'row_idx_grid', 'cm1_i64'], ['rowk_max_in'])
    nn('ReduceMax', ['rowk_max_in'], ['r1'], axes=[0, 1, 2, 3], keepdims=1)
    nn('Where', ['key_mask_bool', 'col_idx_grid', 'sent_big_i64'], ['colk_min_in'])
    nn('ReduceMin', ['colk_min_in'], ['c0'], axes=[0, 1, 2, 3], keepdims=1)
    nn('Where', ['key_mask_bool', 'col_idx_grid', 'cm1_i64'], ['colk_max_in'])
    nn('ReduceMax', ['colk_max_in'], ['c1'], axes=[0, 1, 2, 3], keepdims=1)

    nn('Sub', ['r1', 'r0'], ['h_m1'])
    nn('Add', ['h_m1', 'c1_i64'], ['height'])
    nn('Sub', ['c1', 'c0'], ['w_m1'])
    nn('Add', ['w_m1', 'c1_i64'], ['width'])
    nn('Equal', ['height', 'c5_i64'], ['h5_bool'])
    nn('Equal', ['width', 'c5_i64'], ['w5_bool'])

    nn('Add', ['r0', 'r1'], ['r_sum'])
    nn('Div', ['r_sum', 'c2_i64'], ['cr'])
    nn('Add', ['c0', 'c1'], ['c_sum'])
    nn('Div', ['c_sum', 'c2_i64'], ['cc'])

    # ---- keycolor & linecolor via GatherND on a 2D view of the grid ----
    nn('Reshape', ['color_idx', 'shape30x30'], ['color2d'])
    nn('Reshape', ['key_mask_f', 'shape30x30'], ['keymask2d'])

    def gather_at(grid2d_name, r_name, c_name, out_name):
        r1d = nn('Reshape', [r_name, 'shape1'], [f'{out_name}_r1d'])
        c1d = nn('Reshape', [c_name, 'shape1'], [f'{out_name}_c1d'])
        idx2 = nn('Concat', [r1d, c1d], [f'{out_name}_idx2'], axis=0)
        idx = nn('Reshape', [idx2, 'shape1x2'], [f'{out_name}_idx'])
        return nn('GatherND', [grid2d_name, idx], [out_name])

    keycolor = gather_at('color2d', 'cr', 'cc', 'keycolor_1')

    nn('Sub', ['cr', 'c1_i64'], ['cr_m1'])
    nn('Where', ['h5_bool', 'cr_m1', 'cr'], ['nr'])
    nn('Add', ['cc', 'c1_i64'], ['cc_p1'])
    nn('Where', ['h5_bool', 'cc', 'cc_p1'], ['nc'])
    linecolor = gather_at('color2d', 'nr', 'nc', 'linecolor_1')

    # ---- fg2 = fg1 minus the key object ----
    nn('Not', ['key_mask_bool'], ['not_key_bool'])
    nn('And', ['fg1_bool', 'not_key_bool'], ['fg2_bool'])
    nn('Reshape', ['fg2_bool', 'shape900'], ['fg2_flat_bool'])
    nn('Reshape', ['fg2_flat_bool', 'shape1x900'], ['fg2_flat_bool_row'])

    # ---- per-cell "big object" bbox reusing EQ1 masked by fg2 ----
    nn('Reshape', ['row_flat', 'shape1x900'], ['row_row'])
    nn('Reshape', ['col_flat', 'shape1x900'], ['col_row'])

    nn('And', ['EQ1_bool', 'fg2_flat_bool_row'], ['valid2_bool'])
    nn('Where', ['valid2_bool', 'row_row', 'sent_big_i64'], ['cand_rmin'])
    nn('ReduceMin', ['cand_rmin'], ['rmin2_flat'], axes=[1], keepdims=0)
    nn('Where', ['valid2_bool', 'row_row', 'cm1_i64'], ['cand_rmax'])
    nn('ReduceMax', ['cand_rmax'], ['rmax2_flat'], axes=[1], keepdims=0)
    nn('Where', ['valid2_bool', 'col_row', 'sent_big_i64'], ['cand_cmin'])
    nn('ReduceMin', ['cand_cmin'], ['cmin2_flat'], axes=[1], keepdims=0)
    nn('Where', ['valid2_bool', 'col_row', 'cm1_i64'], ['cand_cmax'])
    nn('ReduceMax', ['cand_cmax'], ['cmax2_flat'], axes=[1], keepdims=0)

    # ---- anchor mask (fg2 & color==keycolor) ----
    nn('Equal', ['color_flat', keycolor], ['is_keycolor_flat'])
    nn('And', ['fg2_flat_bool', 'is_keycolor_flat'], ['anchor_flat_bool'])

    nn('Reshape', ['h5_bool', 'shape1'], ['h5_flat'])
    nn('Reshape', ['w5_bool', 'shape1'], ['w5_flat'])
    nn('Not', ['w5_flat'], ['not_w5_flat'])
    nn('Or', ['h5_flat', 'not_w5_flat'], ['V_flat'])
    nn('Not', ['h5_flat'], ['not_h5_flat'])
    nn('Or', ['w5_flat', 'not_h5_flat'], ['H_flat'])

    # ---- frontier-line paint via (900,900) broadcast ----
    nn('Reshape', ['row_flat', 'shape900x1'], ['row_t'])
    nn('Reshape', ['col_flat', 'shape900x1'], ['col_t'])
    nn('Reshape', ['col_flat', 'shape1x900'], ['col_a'])
    nn('Reshape', ['row_flat', 'shape1x900'], ['row_a'])
    nn('Reshape', ['anchor_flat_bool', 'shape1x900'], ['anchor_a'])
    nn('Reshape', ['rmin2_flat', 'shape1x900'], ['rmin2_a'])
    nn('Reshape', ['rmax2_flat', 'shape1x900'], ['rmax2_a'])
    nn('Reshape', ['cmin2_flat', 'shape1x900'], ['cmin2_a'])
    nn('Reshape', ['cmax2_flat', 'shape1x900'], ['cmax2_a'])

    nn('Equal', ['col_t', 'col_a'], ['colmatch'])
    nn('LessOrEqual', ['rmin2_a', 'row_t'], ['rge'])
    nn('LessOrEqual', ['row_t', 'rmax2_a'], ['rle'])
    nn('And', ['colmatch', 'rge'], ['condv1'])
    nn('And', ['condv1', 'rle'], ['condv2'])
    nn('And', ['condv2', 'anchor_a'], ['cond_v'])
    nn('Cast', ['cond_v'], ['cond_v_f'], to=F)
    nn('ReduceMax', ['cond_v_f'], ['vert_paint_flat_f'], axes=[1], keepdims=0)
    nn('Greater', ['vert_paint_flat_f', 'half_f'], ['vert_paint_flat_bool'])

    nn('Equal', ['row_t', 'row_a'], ['rowmatch'])
    nn('LessOrEqual', ['cmin2_a', 'col_t'], ['cge'])
    nn('LessOrEqual', ['col_t', 'cmax2_a'], ['cle'])
    nn('And', ['rowmatch', 'cge'], ['condh1'])
    nn('And', ['condh1', 'cle'], ['condh2'])
    nn('And', ['condh2', 'anchor_a'], ['cond_h'])
    nn('Cast', ['cond_h'], ['cond_h_f'], to=F)
    nn('ReduceMax', ['cond_h_f'], ['horiz_paint_flat_f'], axes=[1], keepdims=0)
    nn('Greater', ['horiz_paint_flat_f', 'half_f'], ['horiz_paint_flat_bool'])

    nn('And', ['V_flat', 'vert_paint_flat_bool'], ['vpaint_gated'])
    nn('And', ['H_flat', 'horiz_paint_flat_bool'], ['hpaint_gated'])
    nn('Or', ['vpaint_gated', 'hpaint_gated'], ['line_paint_flat_bool'])

    nn('Where', ['line_paint_flat_bool', linecolor, 'color_flat'], ['grid_after_lines_flat'])
    nn('Reshape', ['grid_after_lines_flat', 'shape1x1x30x30'], ['grid_after_lines'])

    # ---- key-shape re-stamp: 5x5 window around each anchor, colors from the original key object ----
    nn('Cast', ['anchor_flat_bool'], ['anchor_flat_f'], to=F)
    nn('Reshape', ['anchor_flat_f', 'shape1x1x30x30'], ['anchor_grid_f'])
    nn('Pad', ['anchor_grid_f', 'pads2_hw'], ['anchor_padded'], mode='constant')

    stamp_color = None
    for (dr, dc) in STAMP_OFFS:
        tag = f'{dr}_{dc}'
        rr = nn('Add', ['cr', f'off_r_{dr}'], [f'rr_{tag}'])
        cc2 = nn('Add', ['cc', f'off_c_{dc}'], [f'cc2_{tag}'])
        ge_r = nn('GreaterOrEqual', [rr, 'c0_i64'], [f'ger_{tag}'])
        le_r = nn('LessOrEqual', [rr, 'c29_i64'], [f'ler_{tag}'])
        ge_c = nn('GreaterOrEqual', [cc2, 'c0_i64'], [f'gec_{tag}'])
        le_c = nn('LessOrEqual', [cc2, 'c29_i64'], [f'lec_{tag}'])
        inb1 = nn('And', [ge_r, le_r], [f'inb1_{tag}'])
        inb2 = nn('And', [ge_c, le_c], [f'inb2_{tag}'])
        inb = nn('And', [inb1, inb2], [f'inb_{tag}'])

        rr_c = nn('Clip', [rr, 'c0_i64', 'c29_i64'], [f'rrc_{tag}'])
        cc_c = nn('Clip', [cc2, 'c0_i64', 'c29_i64'], [f'ccc_{tag}'])

        color_off = gather_at('color2d', rr_c, cc_c, f'coloroff_{tag}')
        keymask_off = gather_at('keymask2d', rr_c, cc_c, f'keymaskoff_{tag}')
        is_key_off = nn('Greater', [keymask_off, 'half_f'], [f'iskeyoff_{tag}'])
        valid_k = nn('And', [inb, is_key_off], [f'validk_{tag}'])
        valid_k_b = nn('Reshape', [valid_k, 'shape1x1x1x1'], [f'validkb_{tag}'])
        color_off_b = nn('Reshape', [color_off, 'shape1x1x1x1'], [f'coloroffb_{tag}'])

        shifted_anchor = nn('Slice', ['anchor_padded', f'st2_{dr}_{dc}', f'en2_{dr}_{dc}', 'ax23'], [f'shiftanc_{tag}'])
        shifted_anchor_bool = nn('Greater', [shifted_anchor, 'half_f'], [f'shiftancbool_{tag}'])
        painted_k = nn('And', [shifted_anchor_bool, valid_k_b], [f'paintedk_{tag}'])

        if stamp_color is None:
            stamp_color = nn('Where', [painted_k, color_off_b, 'cm1_i64'], [f'stampcolor_{tag}'])
        else:
            stamp_color = nn('Where', [painted_k, color_off_b, stamp_color], [f'stampcolor_{tag}'])

    nn('Equal', [stamp_color, 'cm1_i64'], ['stamp_empty_bool'])
    nn('Not', ['stamp_empty_bool'], ['stamp_present_bool'])
    nn('Where', ['stamp_present_bool', stamp_color, 'grid_after_lines'], ['grid_after_stamp'])

    # ---- final fill: force cells that were background (in the key-removed grid) back to bg ----
    nn('Where', ['fg2_bool', 'grid_after_stamp', 'bg_idx'], ['final_color'])

    nn('Reshape', ['final_color', 'shape1x30x30'], ['final_color_sq'])
    nn('OneHot', ['final_color_sq', 'depth10', 'oh_vals'], ['oh_raw'], axis=1)
    nn('Mul', ['oh_raw', 'presence_f'], ['output'])

    graph = helper.make_graph(nodes, 'task054', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


def _make():
    return build_054()

model = _bake(_make(), 54)
