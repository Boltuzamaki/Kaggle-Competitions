import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64
NEG = -100000  # sentinel for "reverse whole axis" via Slice(steps=-1)


def _K(n, a, d=np.int64):
    return numpy_helper.from_array(np.array(a, dtype=d), name=n)


def create_model():
    """task158 / solve_6aa20dc0 intent:
    Find the most-multicolor connected object (the 'key'); its majority color
    is the frame color, its minority-color cells are the two corner markers.
    For each of 3 upscale factors (1,2,3) x 5 transforms (identity, vmirror,
    hmirror, dmirror=transpose, cmirror=anti-transpose), search the grid for
    every place the (scaled+transformed) marker pair occurs with a background
    outbox border around it (this outbox-background requirement, borrowed
    from the sibling task solve_447fd412's pattern, is required to reproduce
    the true arc-gen data -- the literal solve_6aa20dc0 without it produces a
    verified extra false-positive paint on one arc-gen example, see below),
    and paint the corresponding scaled+transformed full template there.

    Verified: running the literal reference arc_dsl_ref.solvers.solve_6aa20dc0
    on task158's data reproduces the *exact same* wrong pixels our first (no
    outbox-check) numpy translation produced on arc-gen[38] -- confirming the
    rule-reading is faithful and the mismatch is an artifact of coincidental
    decoy-object placement, not a translation error. Adding the outbox
    (surrounding-must-be-background) requirement -- which the reference DSL
    omits for 6aa20dc0 but does use for the structurally-identical
    solve_447fd412 -- fixes that one case and gives n_fail==0 across all
    train+test+arc-gen (266/266).
    """
    nodes = []
    inits = []
    _const_cache = {}

    def N(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    def const1d(vals, dtype=np.int64):
        key = (tuple(vals), dtype)
        if key in _const_cache:
            return _const_cache[key]
        nm = f'const_{len(_const_cache)}'
        inits.append(_K(nm, vals, dtype))
        _const_cache[key] = nm
        return nm

    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    inits += [
        _K('ax1', [1]), _K('ax23', [2, 3]), _K('ax2', [2]), _K('ax3', [3]),
        _K('c1', [1]),
        _K('zero_f', [0.0], np.float32), _K('one_f', [1.0], np.float32),
        _K('range10', np.arange(10).reshape(1, 10, 1, 1), np.int64),
        _K('shape1d', [-1]),
        _K('m1', [-1]),
        _K('row_idx', np.arange(30).reshape(1, 1, 30, 1), np.int64),
        _K('col_idx', np.arange(30).reshape(1, 1, 1, 30), np.int64),
        _K('p999', [999]),
        _K('shape1111', [1, 1, 1, 1]),
        _K('c0_scalar', 0), _K('c1_scalar', 1),
    ]

    # ================= STAGE 1: background channel (most common color) =================
    N('ReduceSum', ['input', 'ax23'], ['chan_counts'], keepdims=1)
    N('Reshape', ['chan_counts', 'shape1d'], ['chan_counts_1d'])
    N('ArgMax', ['chan_counts_1d'], ['bg_ch_0d'], axis=0, keepdims=1)

    N('Reshape', ['bg_ch_0d', 'shape1111'], ['bg_ch_4d'])
    N('Equal', ['range10', 'bg_ch_4d'], ['is_bg_ch_bool'])
    N('Cast', ['is_bg_ch_bool'], ['is_bg_ch'], to=F)
    N('Sub', ['one_f', 'is_bg_ch'], ['not_bg_ch'])

    N('Gather', ['input', 'bg_ch_0d'], ['bg_plane_c'], axis=1)  # [1,1,30,30]
    N('Sub', ['one_f', 'bg_plane_c'], ['is_fg'])

    # ================= STAGE 2: connected-component "numcolors" via flood fill =================
    # For every non-background cell, propagate (via 32 rounds of 3x3 max-dilation
    # restricted to the foreground mask) which colors are reachable within its own
    # 8-connected component. Summing over non-background colors gives, per cell,
    # numcolors() of its own connected object -- avoiding the need for explicit
    # connected-component labels.
    presence = 'input'
    for it in range(32):
        dil = N('MaxPool', [presence], [f'dil_{it}'], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
        mx = N('Max', [presence, dil], [f'presmax_{it}'])
        presence = N('Mul', [mx, 'is_fg'], [f'pres_{it}'])

    N('Mul', [presence, 'not_bg_ch'], ['presence_nobg'])
    N('ReduceSum', ['presence_nobg', 'ax1'], ['numcolors_map'], keepdims=1)
    N('ReduceMax', ['numcolors_map'], ['max_numcolors'], axes=[2, 3], keepdims=1)
    N('Equal', ['numcolors_map', 'max_numcolors'], ['is_max_bool'])
    N('Cast', ['is_max_bool'], ['is_max_f'], to=F)
    N('Mul', ['is_max_f', 'is_fg'], ['template_mask'])  # cells of the argmax(numcolors) object

    # ================= STAGE 3: bbox of the template object =================
    N('Greater', ['template_mask', 'zero_f'], ['tmpl_bool'])
    N('Cast', ['tmpl_bool'], ['tmpl_f'], to=F)
    N('ReduceMax', ['tmpl_f'], ['row_any_f'], axes=[3], keepdims=1)
    N('Greater', ['row_any_f', 'zero_f'], ['row_any_b'])
    N('Where', ['row_any_b', 'row_idx', 'p999'], ['row_pmin'])
    N('ReduceMin', ['row_pmin'], ['r_min'], axes=[2], keepdims=1)
    N('Where', ['row_any_b', 'row_idx', 'm1'], ['row_pmax'])
    N('ReduceMax', ['row_pmax'], ['r_max'], axes=[2], keepdims=1)
    N('ReduceMax', ['tmpl_f'], ['col_any_f'], axes=[2], keepdims=1)
    N('Greater', ['col_any_f', 'zero_f'], ['col_any_b'])
    N('Where', ['col_any_b', 'col_idx', 'p999'], ['col_pmin'])
    N('ReduceMin', ['col_pmin'], ['c_min'], axes=[3], keepdims=1)
    N('Where', ['col_any_b', 'col_idx', 'm1'], ['col_pmax'])
    N('ReduceMax', ['col_pmax'], ['c_max'], axes=[3], keepdims=1)

    N('Reshape', ['r_min', 'shape1d'], ['r0_1d'])
    N('Reshape', ['c_min', 'shape1d'], ['c0_1d'])
    N('Add', ['r_max', 'c1'], ['r1'])
    N('Add', ['c_max', 'c1'], ['c1_'])
    N('Reshape', ['r1', 'shape1d'], ['r1_1d'])
    N('Reshape', ['c1_', 'shape1d'], ['c1_1d'])

    N('Slice', ['input', 'r0_1d', 'r1_1d', 'ax2'], ['obj_oh_y'])
    N('Slice', ['obj_oh_y', 'c0_1d', 'c1_1d', 'ax3'], ['obj_oh'])          # [1,10,bh,bw]
    N('Slice', ['template_mask', 'r0_1d', 'r1_1d', 'ax2'], ['obj_mask_y'])
    N('Slice', ['obj_mask_y', 'c0_1d', 'c1_1d', 'ax3'], ['obj_mask'])      # [1,1,bh,bw]

    # ================= STAGE 4: majority color (frame) / minority cells (markers) =================
    N('Mul', ['obj_oh', 'obj_mask'], ['obj_oh_masked'])
    N('ReduceSum', ['obj_oh_masked', 'ax23'], ['color_counts'], keepdims=1)
    N('Reshape', ['color_counts', 'shape1d'], ['color_counts_1d'])
    N('ArgMax', ['color_counts_1d'], ['maj_ch_0d'], axis=0, keepdims=1)
    N('Reshape', ['maj_ch_0d', 'shape1111'], ['maj_ch_4d'])
    N('Equal', ['range10', 'maj_ch_4d'], ['is_maj_ch_bool'])
    N('Cast', ['is_maj_ch_bool'], ['is_maj_ch'], to=F)

    N('Mul', ['obj_oh', 'is_maj_ch'], ['obj_is_maj_oh'])
    N('ReduceSum', ['obj_is_maj_oh', 'ax1'], ['obj_is_maj'], keepdims=1)
    N('Sub', ['one_f', 'obj_is_maj'], ['obj_is_minor_raw'])
    N('Mul', ['obj_is_minor_raw', 'obj_mask'], ['minor_mask'])            # [1,1,bh,bw]

    # ---------- geometry helpers ----------
    def reverse_axis(name, axis, tag):
        out = f'{name}_rev{axis}_{tag}'
        N('Slice', [name, const1d([-1]), const1d([NEG]), const1d([axis]), const1d([-1])], [out])
        return out

    def transpose_hw(name, tag):
        out = f'{name}_T_{tag}'
        N('Transpose', [name], [out], perm=[0, 1, 3, 2])
        return out

    def upscale(name, C, factor, tag):
        """ [1,C,bh,bw] (C static) -> [1,C,bh*factor,bw*factor], nearest-neighbor block replication """
        if factor == 1:
            return name
        shp = f'{name}_shape_{tag}'
        N('Shape', [name], [shp])
        bh = f'{name}_bh_{tag}'
        bw = f'{name}_bw_{tag}'
        N('Slice', [shp, const1d([2]), const1d([3]), const1d([0])], [bh])
        N('Slice', [shp, const1d([3]), const1d([4]), const1d([0])], [bw])
        shape6 = f'{name}_shape6_{tag}'
        N('Concat', [const1d([1]), const1d([C]), bh, const1d([1]), bw, const1d([1])], [shape6], axis=0)
        reshaped = f'{name}_rs6_{tag}'
        N('Reshape', [name, shape6], [reshaped])
        tgt6 = f'{name}_tgt6_{tag}'
        N('Concat', [const1d([1]), const1d([C]), bh, const1d([factor]), bw, const1d([factor])], [tgt6], axis=0)
        expanded = f'{name}_exp_{tag}'
        N('Expand', [reshaped, tgt6], [expanded])
        bhf = f'{name}_bhf_{tag}'
        bwf = f'{name}_bwf_{tag}'
        N('Mul', [bh, const1d([factor])], [bhf])
        N('Mul', [bw, const1d([factor])], [bwf])
        shape4 = f'{name}_shape4_{tag}'
        N('Concat', [const1d([1]), const1d([C]), bhf, bwf], [shape4], axis=0)
        out = f'{name}_up_{tag}'
        N('Reshape', [expanded, shape4], [out])
        return out

    def apply_mirror(name, kind, tag):
        if kind == 'identity':
            return name
        if kind == 'vmirror':
            return reverse_axis(name, 3, tag)
        if kind == 'hmirror':
            return reverse_axis(name, 2, tag)
        if kind == 'dmirror':
            return transpose_hw(name, tag)
        if kind == 'cmirror':
            r = reverse_axis(name, 2, tag + 'c1')
            r = reverse_axis(r, 3, tag + 'c2')
            return transpose_hw(r, tag + 'c3')
        raise ValueError(kind)

    def scalar0d(name1d, tag):
        out = f'{name1d}_0d_{tag}'
        N('Squeeze', [name1d, const1d([0])], [out])
        return out

    # ================= STAGE 5: 3 scales x 5 mirrors -> occurrence search + paint =================
    combo_contribs = []
    combo_flags = []
    idx = 0
    for factor in [1, 2, 3]:
        for kind in ['identity', 'vmirror', 'hmirror', 'dmirror', 'cmirror']:
            tag = f'f{factor}_{kind}_{idx}'
            idx += 1

            oh_s = upscale('obj_oh', 10, factor, tag)
            mask_s = upscale('obj_mask', 1, factor, tag)
            minor_s = upscale('minor_mask', 1, factor, tag)

            oh_t = apply_mirror(oh_s, kind, tag + '_oh')
            mask_t = apply_mirror(mask_s, kind, tag + '_mask')
            minor_t = apply_mirror(minor_s, kind, tag + '_minor')

            marker_kernel = f'marker_kernel_{tag}'
            N('Mul', [oh_t, minor_t], [marker_kernel])   # [1,10,kh,kw] -- Conv weight (out=1,in=10)
            full_kernel = f'full_kernel_{tag}'
            N('Mul', [oh_t, mask_t], [full_kernel])       # [1,10,kh,kw] -- ConvTranspose weight (in=1,out=10)

            marker_total = f'marker_total_{tag}'
            N('ReduceSum', [minor_t, 'ax23'], [f'{marker_total}_4d'], keepdims=1)
            N('Reshape', [f'{marker_total}_4d', 'shape1d'], [marker_total])
            has_marker_bool = f'has_marker_bool_{tag}'
            N('Greater', [marker_total, 'zero_f'], [has_marker_bool])
            has_marker_f = f'has_marker_f_{tag}'
            N('Cast', [has_marker_bool], [has_marker_f], to=F)

            kshape = f'kshape_{tag}'
            N('Shape', [oh_t], [kshape])
            kh = f'kh_{tag}'
            kw = f'kw_{tag}'
            N('Slice', [kshape, const1d([2]), const1d([3]), const1d([0])], [kh])
            N('Slice', [kshape, const1d([3]), const1d([4]), const1d([0])], [kw])

            # ---- marker occurrence search via Conv (kernel_shape inferred from runtime W shape) ----
            kh1 = f'kh1_{tag}'
            kw1 = f'kw1_{tag}'
            N('Sub', [kh, const1d([1])], [kh1])
            N('Sub', [kw, const1d([1])], [kw1])
            pads_grid = f'pads_grid_{tag}'
            N('Concat', [const1d([0, 0, 0, 0, 0, 0]), kh1, kw1], [pads_grid], axis=0)
            padded_grid = f'padded_grid_{tag}'
            N('Pad', ['input', pads_grid, 'zero_f'], [padded_grid], mode='constant')
            match_count = f'match_count_{tag}'
            N('Conv', [padded_grid, marker_kernel], [match_count], strides=[1, 1])
            marker_total_4d = f'marker_total_4d_b_{tag}'
            N('Reshape', [marker_total, 'shape1111'], [marker_total_4d])
            valid_marker_bool = f'valid_marker_bool_{tag}'
            N('Equal', [match_count, marker_total_4d], [valid_marker_bool])
            valid_marker = f'valid_marker_{tag}'
            N('Cast', [valid_marker_bool], [valid_marker], to=F)

            # ---- outbox-must-be-background check (see docstring) ----
            kh2 = f'kh2_{tag}'
            kw2 = f'kw2_{tag}'
            N('Add', [kh, const1d([2])], [kh2])
            N('Add', [kw, const1d([2])], [kw2])
            pads_bg = f'pads_bg_{tag}'
            N('Concat', [const1d([0, 0, 1, 1]), const1d([0, 0]), kh, kw], [pads_bg], axis=0)
            padded_bg = f'padded_bg_{tag}'
            N('Pad', ['bg_plane_c', pads_bg, 'one_f'], [padded_bg], mode='constant')

            kh2_0d = scalar0d(kh2, tag + 'h')
            kw2_0d = scalar0d(kw2, tag + 'w')
            row_range = f'row_range_{tag}'
            col_range = f'col_range_{tag}'
            N('Range', ['c0_scalar', kh2_0d, 'c1_scalar'], [row_range])
            N('Range', ['c0_scalar', kw2_0d, 'c1_scalar'], [col_range])
            row_grid = f'row_grid_{tag}'
            col_grid = f'col_grid_{tag}'
            N('Unsqueeze', [row_range, const1d([1])], [row_grid])
            N('Unsqueeze', [col_range, const1d([0])], [col_grid])

            kh1r = f'kh1r_{tag}'
            kw1r = f'kw1r_{tag}'
            N('Add', [kh, const1d([1])], [kh1r])  # kh2 - 1
            N('Add', [kw, const1d([1])], [kw1r])
            kh1r_0d = scalar0d(kh1r, tag + 'h2')
            kw1r_0d = scalar0d(kw1r, tag + 'w2')

            row_top = f'row_top_{tag}'
            row_bot = f'row_bot_{tag}'
            N('Equal', [row_grid, 'c0_scalar'], [row_top])
            N('Equal', [row_grid, kh1r_0d], [row_bot])
            row_edge = f'row_edge_{tag}'
            N('Or', [row_top, row_bot], [row_edge])

            col_left = f'col_left_{tag}'
            col_right = f'col_right_{tag}'
            N('Equal', [col_grid, 'c0_scalar'], [col_left])
            N('Equal', [col_grid, kw1r_0d], [col_right])
            col_edge = f'col_edge_{tag}'
            N('Or', [col_left, col_right], [col_edge])

            ring_bool = f'ring_bool_{tag}'
            N('Or', [row_edge, col_edge], [ring_bool])
            ring_f = f'ring_f_{tag}'
            N('Cast', [ring_bool], [ring_f], to=F)
            ring_kernel = f'ring_kernel_{tag}'
            N('Unsqueeze', [ring_f, const1d([0, 1])], [ring_kernel])

            ring_bg_count = f'ring_bg_count_{tag}'
            N('Conv', [padded_bg, ring_kernel], [ring_bg_count], strides=[1, 1])

            two_kh2 = f'two_kh2_{tag}'
            two_kw2 = f'two_kw2_{tag}'
            N('Mul', [kh2, const1d([2])], [two_kh2])
            N('Mul', [kw2, const1d([2])], [two_kw2])
            sum_kh_kw = f'sum_kh_kw_{tag}'
            N('Add', [two_kh2, two_kw2], [sum_kh_kw])
            ring_total = f'ring_total_{tag}'
            N('Sub', [sum_kh_kw, const1d([4])], [ring_total])
            ring_total_4d = f'ring_total_4d_{tag}'
            N('Reshape', [ring_total, 'shape1111'], [ring_total_4d])
            ring_total_4d_f = f'ring_total_4d_f_{tag}'
            N('Cast', [ring_total_4d], [ring_total_4d_f], to=F)
            valid_outbox_bool = f'valid_outbox_bool_{tag}'
            N('Equal', [ring_bg_count, ring_total_4d_f], [valid_outbox_bool])
            valid_outbox = f'valid_outbox_{tag}'
            N('Cast', [valid_outbox_bool], [valid_outbox], to=F)

            occurrence_mask = f'occ_mask_{tag}'
            N('Mul', [valid_marker, valid_outbox], [f'{occurrence_mask}_a'])
            N('Mul', [f'{occurrence_mask}_a', has_marker_f], [occurrence_mask])  # [1,1,30,30]

            # ---- scatter (paint) the full pattern at every valid occurrence, via ConvTranspose ----
            contrib_full = f'contrib_full_{tag}'
            N('ConvTranspose', [occurrence_mask, full_kernel], [contrib_full], strides=[1, 1])
            contrib = f'contrib_{tag}'
            N('Slice', [contrib_full, const1d([0, 0]), const1d([30, 30]), const1d([2, 3])], [contrib])

            combo_contribs.append(contrib)
            flag = f'flag_{tag}'
            N('ReduceMax', [contrib], [flag], axes=[1], keepdims=1)
            combo_flags.append(flag)

    painted = combo_contribs[0]
    for c in combo_contribs[1:]:
        nm = f'{painted}_max_{c}'
        N('Max', [painted, c], [nm])
        painted = nm
    flagacc = combo_flags[0]
    for c in combo_flags[1:]:
        nm = f'{flagacc}_max_{c}'
        N('Max', [flagacc, c], [nm])
        flagacc = nm

    N('Greater', [flagacc, 'zero_f'], ['painted_bool'])
    N('Greater', [painted, 'zero_f'], ['painted_color_bool'])
    N('Cast', ['painted_color_bool'], ['painted_color'], to=F)
    N('Where', ['painted_bool', 'painted_color', 'input'], ['output'])

    graph = helper.make_graph(nodes, 'task158', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"),
                 r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
        p = _os.path.join(base, "data", "task%03d.json" % t)
        if _os.path.exists(p): return p
    raise FileNotFoundError("task%03d.json" % t)
def _reps(t, k=8):
    import onnxruntime as _ort  # noqa
    d = json.load(open(_resolve_task_json(t)))
    exs = sorted(d["train"] + d["test"] + d["arc-gen"], key=lambda e: (len(e["input"]), len(e["input"][0])))
    idx = set([0, len(exs) - 1]) | {int(j * (len(exs) - 1) / (k - 1)) for j in range(1, k - 1)}
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
    good = {vi.name for vi in inf.graph.value_info if vi.type.tensor_type.HasField("shape") and not sym(vi)}
    good |= {x.name for x in list(m.graph.input) + list(m.graph.output)}
    missing = []
    for nd in m.graph.node:
        for o in nd.output:
            if o and o != "output" and o not in good and o not in missing: missing.append(o)
    if not missing: return m
    tmp = _copy.deepcopy(m)
    for nm in missing:
        vi = onnx.ValueInfoProto(); vi.name = nm; tmp.graph.output.append(vi)
    so = _ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = _ort.InferenceSession(tmp.SerializeToString(), so)
    mx = {}; dt = {}
    for inp in _reps(t):
        for nm, arr in zip(missing, s.run(missing, {"input": inp})):
            sh = list(arr.shape); mx[nm] = [max(a, b) for a, b in zip(mx[nm], sh)] if nm in mx else sh; dt[nm] = arr.dtype
    keep = [vi for vi in m.graph.value_info if vi.name not in missing]
    del m.graph.value_info[:]; m.graph.value_info.extend(keep)
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64,
            np.dtype("bool"): TensorProto.BOOL, np.dtype("int32"): TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m

def _make():
    return create_model()

model = _bake(_make(), 158)
