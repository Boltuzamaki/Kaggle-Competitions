# --- Available: onnx, helper, numpy_helper, np, TensorProto. Assign final model to `model`. ---
# task349: "outline the color-9 object(s)" -- connected-component + ring/ray drawing.
# This is a genuine object-detection ARC rule, done in static ONNX via parallel
# connected-component labeling (no NonZero/Loop/dynamic-shape ops allowed):
#
#   1. foreground = cells of color 9. label0[pixel] = its own flat index (row*30+col)
#      if foreground, else a sentinel "infinity" (900, one past the last valid index).
#   2. gate_k (k=0..3) = "this pixel AND its 4-connected neighbor in direction k are
#      both foreground" -- computed once via a 1-pixel Slice+Pad shift per direction.
#   3. 32 rounds of parallel min-relaxation: each foreground pixel's label becomes the
#      min of its current label and its same-component neighbors' labels (per gate_k).
#      32 rounds safely exceeds the longest shortest-path on a 30x30 grid, so every
#      foreground pixel converges to the smallest flat index in its own component --
#      i.e. a canonical component id.
#   4. Flatten to 900 pixels and build a [900,900] "same component" matrix (equal
#      label AND both foreground), then for every pixel derive its OWN component's
#      row/col bounding box (via masked ReduceMin/Max over that matrix) and a
#      "thickness" = half the bbox width.
#   5. Per pixel, classify: inside the bbox expanded outward by `thickness` but
#      outside the original bbox -> ring (color 3); strictly beyond the bbox on the
#      same band -> ray (color 1); the object cells themselves -> color 9; else keep
#      background. Finally mask everything to the overall non-background bounding
#      box (so nothing bleeds into the frame's untouched padding).
import numpy as np
F = TensorProto.FLOAT
inits = []
def K(name, arr, dtype=np.float32):
    inits.append(numpy_helper.from_array(np.array(arr, dtype=dtype), name)); return name

# --- scalar/broadcast constants ---
K('half', [0.5]); K('one_f', [1.0]); K('m1f', [-1.0]); K('sentinel', [900.0]); K('bigf', [999.0])
K('zero_f', [0.0]); K('two_f', [2.0]); K('three_f', [3.0]); K('nine_f', [9.0])
K('ax1', [1], np.int64); K('ax23', [2, 3], np.int64)
K('row_idx_f', np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))
K('col_idx_f', np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
flat_idx = (np.arange(30)[:, None] * 30 + np.arange(30)[None, :]).astype(np.float32).reshape(1, 1, 30, 30)
K('flat_idx_f', flat_idx)
chan_ramp = np.arange(10, dtype=np.float32).reshape(1, 10, 1, 1)
K('color_idx', chan_ramp); K('chan_idx', chan_ramp)
K('shape1_1_900', [1, 1, 900], np.int64); K('shape900_1', [900, 1], np.int64); K('shape1_900', [1, 900], np.int64)
K('shape1_1_30_30', [1, 1, 30, 30], np.int64)
K('pv0', [0.0])
K('ax1v', [1], np.int64); K('ch0', [0], np.int64); K('ch1', [1], np.int64)  # unused leftovers, kept for exact parity

# 4-neighbor shift specs (start, end, pad begin/end) -- identical every time they're used,
# whether computing the static adjacency gates or shifting the label each round.
DIRS = [
    ([0, 0], [29, 30], [0, 0, 1, 0, 0, 0, 0, 0]),
    ([1, 0], [30, 30], [0, 0, 0, 0, 0, 0, 1, 0]),
    ([0, 0], [30, 29], [0, 0, 0, 1, 0, 0, 0, 0]),
    ([0, 1], [30, 30], [0, 0, 0, 0, 0, 0, 0, 1]),
]
for k, (s, e, p) in enumerate(DIRS):
    K(f's_g{k}f', s, np.int64); K(f'e_g{k}f', e, np.int64); K(f'pad_g{k}f', p, np.int64)

x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
nodes = []

# --- bounding box of the whole non-background grid, to blank the untouched frame at the end ---
nodes += [
    helper.make_node('ReduceMax', ['input'], ['anych'], axes=[1], keepdims=1),
    helper.make_node('Greater', ['anych', 'half'], ['in_grid_b']),
    helper.make_node('Cast', ['in_grid_b'], ['in_grid_f'], to=1),
    helper.make_node('ReduceMax', ['in_grid_f'], ['row_any_f'], axes=[3], keepdims=1),
    helper.make_node('Greater', ['row_any_f', 'half'], ['row_any_b']),
    helper.make_node('Where', ['row_any_b', 'row_idx_f', 'm1f'], ['r_hi_w']),
    helper.make_node('ReduceMax', ['r_hi_w'], ['r_max'], axes=[1, 2, 3], keepdims=1),
    helper.make_node('ReduceMax', ['in_grid_f'], ['col_any_f'], axes=[2], keepdims=1),
    helper.make_node('Greater', ['col_any_f', 'half'], ['col_any_b']),
    helper.make_node('Where', ['col_any_b', 'col_idx_f', 'm1f'], ['c_hi_w']),
    helper.make_node('ReduceMax', ['c_hi_w'], ['c_max'], axes=[1, 2, 3], keepdims=1),
    helper.make_node('LessOrEqual', ['row_idx_f', 'r_max'], ['row_ok_b']),
    helper.make_node('LessOrEqual', ['col_idx_f', 'c_max'], ['col_ok_b']),
    helper.make_node('And', ['row_ok_b', 'col_ok_b'], ['within_grid_b']),
    helper.make_node('Cast', ['within_grid_b'], ['within_grid_f'], to=1),
]

# --- foreground = color-9 cells; label0 = flat pixel index for fg cells, else "infinity" ---
nodes += [
    helper.make_node('Mul', ['input', 'color_idx'], ['cw']),
    helper.make_node('ReduceSum', ['cw', 'ax1'], ['colorcode'], keepdims=1),
    helper.make_node('Equal', ['colorcode', 'nine_f'], ['fg_b']),
    helper.make_node('Cast', ['fg_b'], ['fg_f'], to=1),
    helper.make_node('Where', ['fg_b', 'flat_idx_f', 'sentinel'], ['label0']),
]

# --- 4-connectivity adjacency gates (computed once, reused every relaxation round) ---
for k in range(4):
    nodes += [
        helper.make_node('Slice', ['fg_f', f's_g{k}f', f'e_g{k}f', 'ax23'], [f'cut_g{k}f']),
        helper.make_node('Pad', [f'cut_g{k}f', f'pad_g{k}f', 'pv0'], [f'shifted_g{k}f'], mode='constant'),
        helper.make_node('Greater', [f'shifted_g{k}f', 'half'], [f'fgb2_{k}']),
        helper.make_node('And', [f'fgb2_{k}', 'fg_b'], [f'gate_{k}']),
    ]

# --- 32 rounds of parallel min-relaxation connected-component labeling ---
label = 'label0'
for it in range(32):
    for k, (s, e, p) in enumerate(DIRS):
        K(f's_it{it}_{k}', s, np.int64); K(f'e_it{it}_{k}', e, np.int64); K(f'pad_it{it}_{k}', p, np.int64)
    running = label
    for k in range(4):
        nodes += [
            helper.make_node('Slice', [label, f's_it{it}_{k}', f'e_it{it}_{k}', 'ax23'], [f'cut_it{it}_{k}']),
            helper.make_node('Pad', [f'cut_it{it}_{k}', f'pad_it{it}_{k}', 'sentinel'], [f'shifted_it{it}_{k}'], mode='constant'),
            helper.make_node('Where', [f'gate_{k}', f'shifted_it{it}_{k}', 'sentinel'], [f'gated_it{it}_{k}']),
        ]
        min_out = f'min_{it}_{k}'
        nodes.append(helper.make_node('Min', [running, f'gated_it{it}_{k}'], [min_out]))
        running = min_out
    label = f'label_{it}'
    nodes.append(helper.make_node('Where', ['fg_b', running, 'sentinel'], [label]))

# --- per-pixel component bounding box (via a flattened [900,900] same-component matrix) ---
nodes += [
    helper.make_node('Reshape', [label, 'shape1_1_900'], ['label_flat']),
    helper.make_node('Reshape', ['label_flat', 'shape900_1'], ['label_col']),
    helper.make_node('Reshape', ['label_flat', 'shape1_900'], ['label_row']),
    helper.make_node('Equal', ['label_col', 'label_row'], ['eqlab_b']),
    helper.make_node('Cast', ['eqlab_b'], ['eqlab_f'], to=1),
    helper.make_node('Reshape', ['fg_f', 'shape900_1'], ['fg_col']),
    helper.make_node('Reshape', ['fg_f', 'shape1_900'], ['fg_row_f']),
    helper.make_node('Reshape', ['fg_b', 'shape1_900'], ['fg_row_b']),
    helper.make_node('Mul', ['eqlab_f', 'fg_col'], ['sc_a']),
    helper.make_node('Mul', ['sc_a', 'fg_row_f'], ['same_comp']),
    helper.make_node('Greater', ['same_comp', 'half'], ['same_comp_b']),
    helper.make_node('Expand', ['row_idx_f', 'shape1_1_30_30'], ['row_idx_full']),
    helper.make_node('Reshape', ['row_idx_full', 'shape1_900'], ['rowflat_row']),
    helper.make_node('Reshape', ['rowflat_row', 'shape900_1'], ['rowflat_col']),
    helper.make_node('Expand', ['col_idx_f', 'shape1_1_30_30'], ['col_idx_full']),
    helper.make_node('Reshape', ['col_idx_full', 'shape1_900'], ['colflat_row']),
    helper.make_node('Reshape', ['colflat_row', 'shape900_1'], ['colflat_col']),
    helper.make_node('Where', ['same_comp_b', 'rowflat_row', 'bigf'], ['fm_r0']),
    helper.make_node('ReduceMin', ['fm_r0'], ['r0_col'], axes=[1], keepdims=1),
    helper.make_node('Where', ['same_comp_b', 'rowflat_row', 'm1f'], ['fm_r1']),
    helper.make_node('ReduceMax', ['fm_r1'], ['r1_col'], axes=[1], keepdims=1),
    helper.make_node('Where', ['same_comp_b', 'colflat_row', 'bigf'], ['fm_c0']),
    helper.make_node('ReduceMin', ['fm_c0'], ['c0_col'], axes=[1], keepdims=1),
    helper.make_node('Where', ['same_comp_b', 'colflat_row', 'm1f'], ['fm_c1']),
    helper.make_node('ReduceMax', ['fm_c1'], ['c1_col'], axes=[1], keepdims=1),
    helper.make_node('Sub', ['c1_col', 'c0_col'], ['wm1']),
    helper.make_node('Add', ['wm1', 'one_f'], ['width_col']),
    helper.make_node('Div', ['width_col', 'two_f'], ['halfw_col']),
    helper.make_node('Floor', ['halfw_col'], ['thick_col']),
    helper.make_node('Reshape', ['r0_col', 'shape1_900'], ['r0_row']),
    helper.make_node('Reshape', ['r1_col', 'shape1_900'], ['r1_row']),
    helper.make_node('Reshape', ['c0_col', 'shape1_900'], ['c0_row']),
    helper.make_node('Reshape', ['c1_col', 'shape1_900'], ['c1_row']),
    helper.make_node('Reshape', ['thick_col', 'shape1_900'], ['thick_row']),
]

# --- classify each pixel as ring / ray / object / background, relative to its own component bbox ---
nodes += [
    helper.make_node('Sub', ['r0_row', 'thick_row'], ['ring_r0']),
    helper.make_node('Add', ['r1_row', 'thick_row'], ['ring_r1']),
    helper.make_node('Sub', ['c0_row', 'thick_row'], ['ring_c0']),
    helper.make_node('Add', ['c1_row', 'thick_row'], ['ring_c1']),
    helper.make_node('GreaterOrEqual', ['rowflat_col', 'ring_r0'], ['a1']),
    helper.make_node('LessOrEqual', ['rowflat_col', 'ring_r1'], ['a2']),
    helper.make_node('GreaterOrEqual', ['colflat_col', 'ring_c0'], ['a3']),
    helper.make_node('LessOrEqual', ['colflat_col', 'ring_c1'], ['a4']),
    helper.make_node('And', ['a1', 'a2'], ['a12']),
    helper.make_node('And', ['a3', 'a4'], ['a34']),
    helper.make_node('And', ['a12', 'a34'], ['in_expanded_b']),
    helper.make_node('GreaterOrEqual', ['rowflat_col', 'r0_row'], ['b1']),
    helper.make_node('LessOrEqual', ['rowflat_col', 'r1_row'], ['b2']),
    helper.make_node('GreaterOrEqual', ['colflat_col', 'c0_row'], ['b3']),
    helper.make_node('LessOrEqual', ['colflat_col', 'c1_row'], ['b4']),
    helper.make_node('And', ['b1', 'b2'], ['b12']),
    helper.make_node('And', ['b3', 'b4'], ['b34']),
    helper.make_node('And', ['b12', 'b34'], ['in_interior_b']),
    helper.make_node('Not', ['in_interior_b'], ['not_interior_b']),
    helper.make_node('And', ['in_expanded_b', 'not_interior_b'], ['ring_cand_b']),
    helper.make_node('And', ['ring_cand_b', 'fg_row_b'], ['ring_match_b']),
    helper.make_node('Cast', ['ring_match_b'], ['ring_match_f'], to=1),
    helper.make_node('ReduceMax', ['ring_match_f'], ['ring_hit_col'], axes=[1], keepdims=1),
    helper.make_node('Greater', ['ring_hit_col', 'half'], ['ring_b']),
    helper.make_node('Greater', ['rowflat_col', 'r1_row'], ['ray_below_b']),
    helper.make_node('And', ['ray_below_b', 'b3'], ['ray_c1']),
    helper.make_node('And', ['ray_c1', 'b4'], ['ray_cand_b']),
    helper.make_node('And', ['ray_cand_b', 'fg_row_b'], ['ray_match_b']),
    helper.make_node('Cast', ['ray_match_b'], ['ray_match_f'], to=1),
    helper.make_node('ReduceMax', ['ray_match_f'], ['ray_hit_col'], axes=[1], keepdims=1),
    helper.make_node('Greater', ['ray_hit_col', 'half'], ['ray_b']),
    helper.make_node('Reshape', ['ring_b', 'shape1_1_30_30'], ['ring_grid_b']),
    helper.make_node('Reshape', ['ray_b', 'shape1_1_30_30'], ['ray_grid_b']),
    helper.make_node('Where', ['ray_grid_b', 'one_f', 'zero_f'], ['c_ray']),
    helper.make_node('Where', ['ring_grid_b', 'three_f', 'c_ray'], ['c_ring_ray']),
    helper.make_node('Where', ['fg_b', 'nine_f', 'c_ring_ray'], ['final_color']),
    helper.make_node('Equal', ['final_color', 'chan_idx'], ['ch_match_b']),
    helper.make_node('Cast', ['ch_match_b'], ['ch_match_f'], to=1),
    helper.make_node('Mul', ['ch_match_f', 'within_grid_f'], ['output']),
]

graph = helper.make_graph(nodes, 'task349', [x], [y], inits)
model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
