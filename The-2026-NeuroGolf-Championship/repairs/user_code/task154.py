# task154 (ARC 6855a6e4): "portrait" branch picks orientation from the color-2 marker's
# combined bbox (height>width -> keep as-is, else rotate 90 CW). Two color-5 groups sit
# strictly above/below the color-2 bbox (row-threshold split, no connected-component
# labeling needed - verified: no color-5 cell ever falls inside the 2-bbox row range).
# Each group's own bbox gets vertically mirrored and slid toward the other group with a
# 3-cell gap: this reduces to a pure per-row remap (columns unchanged, involution):
#   dst_row = 2*lr_row_top + 4 - src_row   (top group, using its own bbox's bottom edge)
#   dst_row = 2*ul_row_bot - 4 - src_row   (bottom group, using its own bbox's top edge)
# Implemented as Gather-by-computed-index (self-inverse, so same formula used to pull
# source row for each destination row) + validity mask for out-of-range rows.
# Rotate back at the end if we rotated at the start. Verified 266/266 (train+test+arc-gen).
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

F = TensorProto.FLOAT
I64 = TensorProto.INT64


def create_model():
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    inits = [
        K('ax1', [1]),
        K('rev_idx30', np.arange(29, -1, -1)),
        K('row_idx', np.arange(30).reshape(1, 1, 30, 1), dtype=np.int64),
        K('col_idx', np.arange(30).reshape(1, 1, 1, 30), dtype=np.int64),
        K('p999', [999]), K('m1', [-1]),
        K('c0_i', [0]), K('c2_i', [2]), K('c4_i', [4]), K('c29_i', [29]),
        K('c0_f', [0.0], dtype=np.float32), K('c1_f', [1.0], dtype=np.float32),
        K('shape30', [-1]),
    ]

    def chan(name, c, dst):
        return helper.make_node('Slice', [name, f'c{c}s', f'c{c+1}s', 'ax1'], [dst])

    for c in range(11):
        inits.append(K(f'c{c}s', [c]))

    n = []

    # --- rot90(input) = Transpose(flip_rows(input)) ---
    n.append(helper.make_node('Gather', ['input', 'rev_idx30'], ['in_flipr'], axis=2))
    n.append(helper.make_node('Transpose', ['in_flipr'], ['rot_input'], perm=[0, 1, 3, 2]))

    # --- portrait test on the ORIGINAL input's color-2 bbox ---
    n.append(chan('input', 2, 'ch2_o'))
    n.append(helper.make_node('ReduceMax', ['ch2_o'], ['rowany2_o'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['rowany2_o', 'c0_f'], ['rowb2_o']))
    n.append(helper.make_node('Where', ['rowb2_o', 'row_idx', 'p999'], ['rpmin_o']))
    n.append(helper.make_node('ReduceMin', ['rpmin_o'], ['r2min_o'], axes=[2], keepdims=1))
    n.append(helper.make_node('Where', ['rowb2_o', 'row_idx', 'm1'], ['rpmax_o']))
    n.append(helper.make_node('ReduceMax', ['rpmax_o'], ['r2max_o'], axes=[2], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['ch2_o'], ['colany2_o'], axes=[2], keepdims=1))
    n.append(helper.make_node('Greater', ['colany2_o', 'c0_f'], ['colb2_o']))
    n.append(helper.make_node('Where', ['colb2_o', 'col_idx', 'p999'], ['cpmin_o']))
    n.append(helper.make_node('ReduceMin', ['cpmin_o'], ['c2min_o'], axes=[3], keepdims=1))
    n.append(helper.make_node('Where', ['colb2_o', 'col_idx', 'm1'], ['cpmax_o']))
    n.append(helper.make_node('ReduceMax', ['cpmax_o'], ['c2max_o'], axes=[3], keepdims=1))
    n.append(helper.make_node('Sub', ['r2max_o', 'r2min_o'], ['h2']))
    n.append(helper.make_node('Sub', ['c2max_o', 'c2min_o'], ['w2']))
    n.append(helper.make_node('Greater', ['h2', 'w2'], ['portrait']))

    # --- work = portrait ? input : rot90(input) ---
    n.append(helper.make_node('Where', ['portrait', 'input', 'rot_input'], ['work']))

    # --- color-2 bbox rows on "work" (post-rotation) ---
    n.append(chan('work', 2, 'ch2_w'))
    n.append(helper.make_node('ReduceMax', ['ch2_w'], ['rowany2_w'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['rowany2_w', 'c0_f'], ['rowb2_w']))
    n.append(helper.make_node('Where', ['rowb2_w', 'row_idx', 'p999'], ['rpmin_w']))
    n.append(helper.make_node('ReduceMin', ['rpmin_w'], ['r2min'], axes=[2], keepdims=1))
    n.append(helper.make_node('Where', ['rowb2_w', 'row_idx', 'm1'], ['rpmax_w']))
    n.append(helper.make_node('ReduceMax', ['rpmax_w'], ['r2max'], axes=[2], keepdims=1))

    # --- color-5 mask on "work", split into top/bottom groups by row threshold ---
    n.append(chan('work', 5, 'mask5'))
    n.append(helper.make_node('Less', ['row_idx', 'r2min'], ['top_lt']))
    n.append(helper.make_node('Cast', ['top_lt'], ['top_lt_f'], to=F))
    n.append(helper.make_node('Mul', ['mask5', 'top_lt_f'], ['top_region']))
    n.append(helper.make_node('Greater', ['row_idx', 'r2max'], ['bot_gt']))
    n.append(helper.make_node('Cast', ['bot_gt'], ['bot_gt_f'], to=F))
    n.append(helper.make_node('Mul', ['mask5', 'bot_gt_f'], ['bot_region']))

    # --- lr_r_top (bottom edge of top group), ul_r_bot (top edge of bottom group) ---
    n.append(helper.make_node('ReduceMax', ['top_region'], ['top_rp'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['top_rp', 'c0_f'], ['top_rp_b']))
    n.append(helper.make_node('Where', ['top_rp_b', 'row_idx', 'm1'], ['top_rp_sel']))
    n.append(helper.make_node('ReduceMax', ['top_rp_sel'], ['lr_r_top'], axes=[2], keepdims=1))
    n.append(helper.make_node('ReduceMax', ['bot_region'], ['bot_rp'], axes=[3], keepdims=1))
    n.append(helper.make_node('Greater', ['bot_rp', 'c0_f'], ['bot_rp_b']))
    n.append(helper.make_node('Where', ['bot_rp_b', 'row_idx', 'p999'], ['bot_rp_sel']))
    n.append(helper.make_node('ReduceMin', ['bot_rp_sel'], ['ul_r_bot'], axes=[2], keepdims=1))

    # --- src_r_top = 2*lr_r_top + 4 - row_idx (involution; also used as inverse map) ---
    n.append(helper.make_node('Mul', ['lr_r_top', 'c2_i'], ['two_lrtop']))
    n.append(helper.make_node('Add', ['two_lrtop', 'c4_i'], ['two_lrtop_p4']))
    n.append(helper.make_node('Sub', ['two_lrtop_p4', 'row_idx'], ['src_r_top']))
    n.append(helper.make_node('GreaterOrEqual', ['src_r_top', 'c0_i'], ['vt_ge']))
    n.append(helper.make_node('LessOrEqual', ['src_r_top', 'c29_i'], ['vt_le']))
    n.append(helper.make_node('And', ['vt_ge', 'vt_le'], ['valid_top']))
    n.append(helper.make_node('Cast', ['valid_top'], ['valid_top_f'], to=F))
    n.append(helper.make_node('Max', ['src_r_top', 'c0_i'], ['src_r_top_m']))
    n.append(helper.make_node('Min', ['src_r_top_m', 'c29_i'], ['src_r_top_c']))
    n.append(helper.make_node('Reshape', ['src_r_top_c', 'shape30'], ['src_r_top_idx']))
    n.append(helper.make_node('Gather', ['top_region', 'src_r_top_idx'], ['top_gathered'], axis=2))
    n.append(helper.make_node('Mul', ['top_gathered', 'valid_top_f'], ['top_shifted']))

    # --- src_r_bot = 2*ul_r_bot - 4 - row_idx ---
    n.append(helper.make_node('Mul', ['ul_r_bot', 'c2_i'], ['two_ulbot']))
    n.append(helper.make_node('Sub', ['two_ulbot', 'c4_i'], ['two_ulbot_m4']))
    n.append(helper.make_node('Sub', ['two_ulbot_m4', 'row_idx'], ['src_r_bot']))
    n.append(helper.make_node('GreaterOrEqual', ['src_r_bot', 'c0_i'], ['vb_ge']))
    n.append(helper.make_node('LessOrEqual', ['src_r_bot', 'c29_i'], ['vb_le']))
    n.append(helper.make_node('And', ['vb_ge', 'vb_le'], ['valid_bot']))
    n.append(helper.make_node('Cast', ['valid_bot'], ['valid_bot_f'], to=F))
    n.append(helper.make_node('Max', ['src_r_bot', 'c0_i'], ['src_r_bot_m']))
    n.append(helper.make_node('Min', ['src_r_bot_m', 'c29_i'], ['src_r_bot_c']))
    n.append(helper.make_node('Reshape', ['src_r_bot_c', 'shape30'], ['src_r_bot_idx']))
    n.append(helper.make_node('Gather', ['bot_region', 'src_r_bot_idx'], ['bot_gathered'], axis=2))
    n.append(helper.make_node('Mul', ['bot_gathered', 'valid_bot_f'], ['bot_shifted']))

    n.append(helper.make_node('Max', ['top_shifted', 'bot_shifted'], ['new_five']))
    n.append(helper.make_node('Sub', ['c1_f', 'new_five'], ['inv_new5']))

    # --- cover originals (channel0 absorbs mask5), paint new_five onto channel5,
    #     zero every other channel at the new_five positions ---
    n.append(chan('work', 0, 'c0'))
    n.append(helper.make_node('Add', ['c0', 'mask5'], ['cover_c0']))
    n.append(helper.make_node('Mul', ['cover_c0', 'inv_new5'], ['final_c0']))
    final = ['final_c0']
    for c in range(1, 10):
        if c == 5:
            final.append('new_five')
            continue
        n.append(chan('work', c, f'c{c}'))
        n.append(helper.make_node('Mul', [f'c{c}', 'inv_new5'], [f'final_c{c}']))
        final.append(f'final_c{c}')
    n.append(helper.make_node('Concat', final, ['painted'], axis=1))

    # --- rotate back if we rotated at the start: rot270(painted) ---
    n.append(helper.make_node('Transpose', ['painted'], ['painted_t'], perm=[0, 1, 3, 2]))
    n.append(helper.make_node('Gather', ['painted_t', 'rev_idx30'], ['painted_rot270'], axis=2))

    n.append(helper.make_node('Where', ['portrait', 'painted', 'painted_rot270'], ['output']))

    graph = helper.make_graph(n, 'task154', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# ===== Owned repair scaffolding (Claude, verified vs train+test+arc-gen) =====
import os as _os, copy as _copy
def _K(n, a, d): return numpy_helper.from_array(np.array(a, dtype=d), name=n)
def _resolve_task_json(t):
    for base in [_os.environ.get("PROJECT_DIR", "/project"), r"C:\\Users\\chand\\OneDrive\\Desktop\\get_a_job\\kaggle_competitions\\The 2026 NeuroGolf Championship", "."]:
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
    conv = {np.dtype("float32"): TensorProto.FLOAT, np.dtype("int64"): TensorProto.INT64, np.dtype("bool"): TensorProto.BOOL, np.dtype("int32"): TensorProto.INT32}
    for nm in missing:
        m.graph.value_info.append(helper.make_tensor_value_info(nm, conv.get(dt[nm], TensorProto.FLOAT), mx[nm]))
    return m


def _make():
    return create_model()


model = _bake(_make(), 154)
