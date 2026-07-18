import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime
F = TensorProto.FLOAT; I64 = TensorProto.INT64; I32 = TensorProto.INT32; BOOL = TensorProto.BOOL

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

# ===== task101: ARC-DSL solve_447fd412 intent, re-derived + verified against arc_dsl_ref =====
#
# Rule (verified in pure numpy against every one of the 266 train+test+arc-gen examples in
# data/task101.json, n_fail=0): grid has connected components (8-connectivity, any-color merge,
# background=0 excluded). The "template" is the unique component with the most distinct colors
# (always 2 here: a "majority" color + a "marker" color). Every other component is either a
# solid block of ONLY the marker color, or a solid block of ONLY the majority color, each scaled
# by some integer factor k in {1,2,3,4} relative to the template's own local pattern. For each
# such found occurrence (search done for BOTH the marker-only sub-pattern and the majority-only
# sub-pattern, at all 4 scales, requiring the immediate 1-cell "outbox" ring around that
# sub-pattern's own bounding box to be background), the FULL template pattern (majority+marker
# cells together), scaled by that same k, is painted into the grid anchored at the found offset.
#
# ONNX translation (no Loop/Scan/NonZero/Unique/Compress): connected components + per-component
# distinct-color count computed via the Loop-free iterative min-label / max-propagate idiom
# (same pattern as task048/task112, generalized to 9 color channels at once instead of 1).
# The template's local pattern is extracted into a small FIXED 10x10 canvas (1-cell margin on
# all sides so a sub-pattern's own bbox edge touching the template's bbox edge still has room
# for its outbox ring) via a dynamic (but always exactly-10x10) Slice on a pre-padded, mask-gated
# copy of the input. The "search across every possible offset" step -- normally a Python loop --
# is instead done as a single Conv (cross-correlation: kernel = the small pattern zero-padded
# into a fixed 40x40 canvas, tests every offset at once) and the "paint the matched template
# copies back" step is a single ConvTranspose (scatters a full copy of the template at every
# valid offset). This exact Conv/ConvTranspose reformulation (with a 1-cell canvas margin fix
# for the ring check) was verified to reproduce n_fail=0 in pure numpy before transcription.
# Empirically (this task's own train+test+arc-gen set): max template bbox dimension is 4, and
# the needed alignment shift range is [-2,14]; the fixed sizes below (LOCAL_CONTENT=8,
# shift range covering [-40,29]) carry generous safety margin over both.

R_ITERS = 8
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
MARGIN = 1
LOCAL_CONTENT = 8
LOCAL_CANVAS = LOCAL_CONTENT + 2 * MARGIN   # 10
PREPAD = 10
KMAX = 4
KERNEL = LOCAL_CANVAS * KMAX                # 40
PAD_TL = KERNEL                             # 40
PAD_BR = KERNEL - 1                         # 39 -> exactly 70 shift positions [-40,29]
OUTN = 30 + PAD_TL + PAD_BR - KERNEL + 1     # 70


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

    for v in range(0, 11):
        addK(f'c{v}i64', [v], np.int64)
    addK('c0f', [0.0], np.float32)
    addK('c1f', [1.0], np.float32)
    addK('c0i32', [0], np.int32)
    addK('sent_i32', [100000], np.int32)
    addK('m1', [-1], np.int64)
    addK('p999', [999], np.int64)
    addK('ax1', [1], np.int64)
    addK('ax23', [2, 3], np.int64)
    addK('shape1d', [-1], np.int64)
    addK('depth10', [10], np.int64)
    addK('oh_vals_f', [0.0, 1.0], np.float32)
    addK('pads_hw', [0, 0, 1, 1, 0, 0, 1, 1], np.int64)
    init_label_np = (np.arange(30).reshape(30, 1) * 30 + np.arange(30).reshape(1, 30)).astype(np.int32)
    addK('init_label', init_label_np.reshape(1, 1, 30, 30), np.int32)
    addK('row_idx30', np.arange(30).reshape(1, 1, 30, 1), np.int64)
    addK('col_idx30', np.arange(30).reshape(1, 1, 1, 30), np.int64)
    addK('row_idx10', np.arange(LOCAL_CANVAS).reshape(1, 1, LOCAL_CANVAS, 1), np.int64)
    addK('col_idx10', np.arange(LOCAL_CANVAS).reshape(1, 1, 1, LOCAL_CANVAS), np.int64)
    addK('row_idxK', np.arange(KERNEL).reshape(1, 1, KERNEL, 1), np.int64)
    addK('col_idxK', np.arange(KERNEL).reshape(1, 1, 1, KERNEL), np.int64)
    addK('c_prepad_minus_margin', [PREPAD - MARGIN], np.int64)
    addK('c_local_canvas', [LOCAL_CANVAS], np.int64)
    addK('pads_prepad', [0, 0, PREPAD, PREPAD, 0, 0, PREPAD, PREPAD], np.int64)
    addK('pads_grid', [0, 0, PAD_TL, PAD_TL, 0, 0, PAD_BR, PAD_BR], np.int64)
    addK('crop_start', [PAD_TL, PAD_TL], np.int64)
    addK('crop_end', [PAD_TL + 30, PAD_TL + 30], np.int64)
    addK('zero_30x30', np.zeros((1, 1, 30, 30), dtype=np.float32), np.float32)

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

    # ---- foreground / color idx ----
    nn('ArgMax', ['input'], ['color_idx64'], axis=1, keepdims=1)
    nn('Cast', ['color_idx64'], ['color_idx'], to=I32)
    nn('Equal', ['color_idx', 'c0i32'], ['is_bg'])
    nn('Not', ['is_bg'], ['is_fg_bool'])
    nn('Cast', ['is_fg_bool'], ['fg_f'], to=F)
    nn('Cast', ['is_fg_bool'], ['fg_i32'], to=I32)

    # ---- fixed 8-neighbor "is fg" masks ----
    nn('Pad', ['fg_i32', 'pads_hw', 'c0i32'], ['padded_fg'], mode='constant')
    shifted_fg = eight_slices('padded_fg', 'nbrfg')
    nbr_is_fg = []
    for k, sfg in enumerate(shifted_fg):
        nbr_is_fg.append(nn('Cast', [sfg], [f'nbr_is_fg_{k}'], to=BOOL))

    # ---- connected components: Loop-free iterative min-label propagation ----
    label = 'init_label'
    for it in range(R_ITERS):
        padded_label = nn('Pad', [label, 'pads_hw', 'sent_i32'], [f'padded_label_it{it}'], mode='constant')
        shifted = eight_slices(padded_label, f'lab_it{it}')
        running = label
        for k in range(8):
            cand = nn('Where', [nbr_is_fg[k], shifted[k], 'sent_i32'], [f'cand_it{it}_{k}'])
            running = nn('Min', [running, cand], [f'min_it{it}_{k}'])
        label = running

    # ---- has_c for colors 1..9 at once: max-propagate through the SAME fg-gated connectivity ----
    nn('Slice', ['input', 'c1i64', 'c10i64', 'ax1'], ['has_c_init'])  # [1,9,30,30]
    has_c = 'has_c_init'
    for it in range(R_ITERS):
        padded_hc = nn('Pad', [has_c, 'pads_hw', 'c0f'], [f'padded_hc_it{it}'], mode='constant')
        shifted = eight_slices(padded_hc, f'hc_it{it}')
        running = has_c
        for k in range(8):
            cand = nn('Where', [nbr_is_fg[k], shifted[k], 'c0f'], [f'canhc_it{it}_{k}'])
            running = nn('Max', [running, cand], [f'maxhc_it{it}_{k}'])
        has_c = running
    nn('ReduceSum', [has_c], ['numcolors_map'], axes=[1], keepdims=1)  # [1,1,30,30]

    # ---- template = component with max distinct colors ----
    nn('Mul', ['numcolors_map', 'fg_f'], ['nc_masked'])
    nn('ReduceMax', ['nc_masked'], ['global_max'], axes=[0, 1, 2, 3], keepdims=1)
    nn('Equal', ['numcolors_map', 'global_max'], ['is_template_eq'])
    nn('And', ['is_template_eq', 'is_fg_bool'], ['template_mask_bool'])
    nn('Cast', ['template_mask_bool'], ['template_mask'], to=F)  # [1,1,30,30]

    # ---- bbox (top-left only needed) of template_mask on the full 30x30 canvas ----
    nn('ReduceMax', ['template_mask'], ['t_row_any'], axes=[3], keepdims=1)
    nn('Greater', ['t_row_any', 'c0f'], ['t_row_any_b'])
    nn('Where', ['t_row_any_b', 'row_idx30', 'p999'], ['t_row_pmin'])
    nn('ReduceMin', ['t_row_pmin'], ['t_r0_4d'], axes=[2], keepdims=1)
    nn('ReduceMax', ['template_mask'], ['t_col_any'], axes=[2], keepdims=1)
    nn('Greater', ['t_col_any', 'c0f'], ['t_col_any_b'])
    nn('Where', ['t_col_any_b', 'col_idx30', 'p999'], ['t_col_pmin'])
    nn('ReduceMin', ['t_col_pmin'], ['t_c0_4d'], axes=[3], keepdims=1)
    nn('Reshape', ['t_r0_4d', 'shape1d'], ['t_r0_1d'])
    nn('Reshape', ['t_c0_4d', 'shape1d'], ['t_c0_1d'])
    nn('Add', ['t_r0_1d', 'c_prepad_minus_margin'], ['start_r'])
    nn('Add', ['t_c0_1d', 'c_prepad_minus_margin'], ['start_c'])
    nn('Add', ['start_r', 'c_local_canvas'], ['end_r'])
    nn('Add', ['start_c', 'c_local_canvas'], ['end_c'])
    nn('Concat', ['start_r', 'start_c'], ['starts_rc'], axis=0)
    nn('Concat', ['end_r', 'end_c'], ['ends_rc'], axis=0)

    # ---- extract template's own local pattern into a fixed 10x10 canvas (1-cell margin) ----
    nn('Mul', ['input', 'template_mask'], ['template_masked_onehot'])
    nn('Pad', ['template_masked_onehot', 'pads_prepad', 'c0f'], ['padded_template'], mode='constant')
    nn('Slice', ['padded_template', 'starts_rc', 'ends_rc', 'ax23'], ['local_onehot'])  # dyn -> [1,10,10,10]

    # ---- majority color / majority+marker masks within the local canvas ----
    nn('Slice', ['local_onehot', 'c1i64', 'c10i64', 'ax1'], ['local_fg_onehot'])  # [1,9,10,10]
    nn('ReduceSum', ['local_fg_onehot'], ['counts9'], axes=[2, 3], keepdims=1)  # [1,9,1,1]
    nn('ArgMax', ['counts9'], ['maj_idx0'], axis=1, keepdims=0)  # [1,1,1] in 0..8
    nn('Add', ['maj_idx0', 'c1i64'], ['maj_idx'])  # 1..9
    nn('OneHot', ['maj_idx', 'depth10', 'oh_vals_f'], ['majority_onehot'], axis=1)  # [1,10,1,1]
    nn('Mul', ['local_onehot', 'majority_onehot'], ['local_onehot_maj_masked'])
    nn('ReduceSum', ['local_onehot_maj_masked'], ['local_majority_mask'], axes=[1], keepdims=1)  # [1,1,10,10]
    nn('ReduceMax', ['local_fg_onehot'], ['local_fg_mask'], axes=[1], keepdims=1)  # [1,1,10,10]
    nn('Sub', ['local_fg_mask', 'local_majority_mask'], ['local_marker_mask'])  # [1,1,10,10]

    def bbox10(mask_name, prefix):
        nn('ReduceMax', [mask_name], [f'{prefix}_row_any'], axes=[3], keepdims=1)
        nn('Greater', [f'{prefix}_row_any', 'c0f'], [f'{prefix}_row_any_b'])
        nn('Where', [f'{prefix}_row_any_b', 'row_idx10', 'p999'], [f'{prefix}_row_pmin'])
        r0 = nn('ReduceMin', [f'{prefix}_row_pmin'], [f'{prefix}_r0'], axes=[2], keepdims=1)
        nn('Where', [f'{prefix}_row_any_b', 'row_idx10', 'm1'], [f'{prefix}_row_pmax'])
        r1m1 = nn('ReduceMax', [f'{prefix}_row_pmax'], [f'{prefix}_r1m1'], axes=[2], keepdims=1)
        r1 = nn('Add', [r1m1, 'c1i64'], [f'{prefix}_r1'])
        nn('ReduceMax', [mask_name], [f'{prefix}_col_any'], axes=[2], keepdims=1)
        nn('Greater', [f'{prefix}_col_any', 'c0f'], [f'{prefix}_col_any_b'])
        nn('Where', [f'{prefix}_col_any_b', 'col_idx10', 'p999'], [f'{prefix}_col_pmin'])
        c0 = nn('ReduceMin', [f'{prefix}_col_pmin'], [f'{prefix}_c0'], axes=[3], keepdims=1)
        nn('Where', [f'{prefix}_col_any_b', 'col_idx10', 'm1'], [f'{prefix}_col_pmax'])
        c1m1 = nn('ReduceMax', [f'{prefix}_col_pmax'], [f'{prefix}_c1m1'], axes=[3], keepdims=1)
        c1 = nn('Add', [c1m1, 'c1i64'], [f'{prefix}_c1'])
        return r0, r1, c0, c1

    maj_bbox = bbox10('local_majority_mask', 'maj')
    mark_bbox = bbox10('local_marker_mask', 'mark')

    def upsample_pad(name_in, C, k, out_name):
        shp_in = addK(f'{out_name}_shpin', [1, C, 10, 1, 10, 1], np.int64)
        shp_out = addK(f'{out_name}_shpout', [1, C, 10, k, 10, k], np.int64)
        shp_fin = addK(f'{out_name}_shpfin', [1, C, 10 * k, 10 * k], np.int64)
        r1 = nn('Reshape', [name_in, shp_in], [f'{out_name}_r1'])
        e1 = nn('Expand', [r1, shp_out], [f'{out_name}_e1'])
        r2 = nn('Reshape', [e1, shp_fin], [f'{out_name}_r2'])
        if 10 * k < KERNEL:
            padk = addK(f'{out_name}_padk', [0, 0, 0, 0, 0, 0, KERNEL - 10 * k, KERNEL - 10 * k], np.int64)
            nn('Pad', [r2, padk, 'c0f'], [out_name], mode='constant')
        else:
            nn('Identity', [r2], [out_name])
        return out_name

    def ring_mask(bbox, k, prefix):
        r0, r1, c0, c1 = bbox
        kk = addK(f'{prefix}_kk', [k], np.int64)
        sr0 = nn('Mul', [r0, kk], [f'{prefix}_sr0'])
        sr1 = nn('Mul', [r1, kk], [f'{prefix}_sr1'])
        sc0 = nn('Mul', [c0, kk], [f'{prefix}_sc0'])
        sc1 = nn('Mul', [c1, kk], [f'{prefix}_sc1'])
        or0 = nn('Sub', [sr0, 'c1i64'], [f'{prefix}_or0'])
        or1 = nn('Add', [sr1, 'c1i64'], [f'{prefix}_or1'])
        oc0 = nn('Sub', [sc0, 'c1i64'], [f'{prefix}_oc0'])
        oc1 = nn('Add', [sc1, 'c1i64'], [f'{prefix}_oc1'])
        row_ge_o = nn('GreaterOrEqual', ['row_idxK', or0], [f'{prefix}_rgeo'])
        row_lt_o = nn('Less', ['row_idxK', or1], [f'{prefix}_rlto'])
        row_outer = nn('And', [row_ge_o, row_lt_o], [f'{prefix}_rout'])
        col_ge_o = nn('GreaterOrEqual', ['col_idxK', oc0], [f'{prefix}_cgeo'])
        col_lt_o = nn('Less', ['col_idxK', oc1], [f'{prefix}_clto'])
        col_outer = nn('And', [col_ge_o, col_lt_o], [f'{prefix}_cout'])
        outer = nn('And', [row_outer, col_outer], [f'{prefix}_outer'])
        row_ge_i = nn('GreaterOrEqual', ['row_idxK', sr0], [f'{prefix}_rges'])
        row_lt_i = nn('Less', ['row_idxK', sr1], [f'{prefix}_rlts'])
        row_inner = nn('And', [row_ge_i, row_lt_i], [f'{prefix}_rinn'])
        col_ge_i = nn('GreaterOrEqual', ['col_idxK', sc0], [f'{prefix}_cges'])
        col_lt_i = nn('Less', ['col_idxK', sc1], [f'{prefix}_clts'])
        col_inner = nn('And', [col_ge_i, col_lt_i], [f'{prefix}_cinn'])
        inner = nn('And', [row_inner, col_inner], [f'{prefix}_inner'])
        not_inner = nn('Not', [inner], [f'{prefix}_notinner'])
        ring_b = nn('And', [outer, not_inner], [f'{prefix}_ringb'])
        ring_f = nn('Cast', [ring_b], [f'{prefix}_ringf'], to=F)
        return ring_f

    nn('Pad', ['input', 'pads_grid', 'c0f'], ['grid_padded'], mode='constant')  # [1,10,109,109]
    nn('Pad', ['fg_f', 'pads_grid', 'c0f'], ['fg_padded'], mode='constant')  # [1,1,109,109]

    paint_accum = None
    for k in range(1, KMAX + 1):
        scaled_full = upsample_pad('local_onehot', 10, k, f'scaled_full_k{k}')
        scaled_maj = upsample_pad('local_majority_mask', 1, k, f'scaled_maj_k{k}')
        scaled_mark = upsample_pad('local_marker_mask', 1, k, f'scaled_mark_k{k}')

        for pname, scaled_mask, bbox in [('maj', scaled_maj, maj_bbox), ('mark', scaled_mark, mark_bbox)]:
            tag = f'{pname}_k{k}'
            req = nn('Mul', [scaled_full, scaled_mask], [f'req_{tag}'])
            total_req = nn('ReduceSum', [req], [f'totreq_{tag}'], axes=[0, 1, 2, 3], keepdims=1)
            ring = ring_mask(bbox, k, f'ring_{tag}')

            match_map = nn('Conv', ['grid_padded', req], [f'matchmap_{tag}'],
                           kernel_shape=[KERNEL, KERNEL], strides=[1, 1], pads=[0, 0, 0, 0], group=1)
            ring_map = nn('Conv', ['fg_padded', ring], [f'ringmap_{tag}'],
                          kernel_shape=[KERNEL, KERNEL], strides=[1, 1], pads=[0, 0, 0, 0], group=1)

            eq_match = nn('Equal', [match_map, total_req], [f'eqm_{tag}'])
            eq_ring = nn('Equal', [ring_map, 'c0f'], [f'eqr_{tag}'])
            gt_req = nn('Greater', [total_req, 'c0f'], [f'gtr_{tag}'])
            v1 = nn('And', [eq_match, eq_ring], [f'v1_{tag}'])
            valid = nn('And', [v1, gt_req], [f'valid_{tag}'])
            valid_f = nn('Cast', [valid], [f'validf_{tag}'], to=F)

            paint_big = nn('ConvTranspose', [valid_f, scaled_full], [f'paintbig_{tag}'],
                           kernel_shape=[KERNEL, KERNEL], strides=[1, 1], pads=[0, 0, 0, 0], group=1)
            paint_crop = nn('Slice', [paint_big, 'crop_start', 'crop_end', 'ax23'], [f'paintcrop_{tag}'])

            if paint_accum is None:
                paint_accum = paint_crop
            else:
                paint_accum = nn('Add', [paint_accum, paint_crop], [f'accum_after_{tag}'])

    nn('Slice', [paint_accum, 'c1i64', 'c10i64', 'ax1'], ['paint_accum_fg'])
    nn('ReduceSum', ['paint_accum_fg'], ['painted_sum'], axes=[1], keepdims=1)
    nn('Greater', ['painted_sum', 'c0f'], ['painted_any'])

    nn('Slice', ['input', 'c0i64', 'c1i64', 'ax1'], ['in_ch0'])
    nn('Where', ['painted_any', 'zero_30x30', 'in_ch0'], ['out_ch0'])

    out_channels = ['out_ch0']
    for c in range(1, 10):
        in_c = nn('Slice', ['input', f'c{c}i64', f'c{c+1}i64', 'ax1'], [f'in_ch{c}'])
        acc_c = nn('Slice', [paint_accum, f'c{c}i64', f'c{c+1}i64', 'ax1'], [f'acc_ch{c}'])
        gt_c = nn('Greater', [acc_c, 'c0f'], [f'gtc_{c}'])
        paintedc_f = nn('Cast', [gt_c], [f'paintedcf_{c}'], to=F)
        outc = nn('Where', ['painted_any', paintedc_f, in_c], [f'out_ch{c}'])
        out_channels.append(outc)

    nn('Concat', out_channels, ['output'], axis=1)

    graph = helper.make_graph(nodes, 'task101', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])


model = _bake(_make(), 101)
