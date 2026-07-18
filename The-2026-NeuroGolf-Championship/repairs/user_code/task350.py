# --- Available: onnx, helper, numpy_helper, np, TensorProto. Assign final model to `model`. ---
# task350: "connect the dots" rule.
# For every pair of color-1 cells that share a ROW, fill the cells strictly between them
# (same row, column between min/max) with color 8 -- and symmetric for cells sharing a COLUMN --
# but only where that row/column has >=2 color-1 cells (a lone dot draws nothing), and never
# overwriting an existing color-1 cell (the endpoints stay color 1).
#
# Implementation (all static [1,10,30,30] one-hot tensors):
#   is1_b   = channel 1 > 0                                  -- boolean "is a color-1 cell"
#   per row: rowcount = #color-1 cells in the row; rowmin/rowmax = their column extent
#            (via Where(is1_b, col_idx, +-sentinel) + ReduceMin/Max over the column axis)
#   fillrow = row has >=2 dots AND this column is strictly inside [rowmin,rowmax] AND not itself a dot
#   fillcol = symmetric, swapping rows/columns
#   fill    = fillrow OR fillcol
#   output  = input with all channels zeroed at fill positions, channel 8 set to 1 there instead
import numpy as np
F = TensorProto.FLOAT
inits = []
def K(name, arr, dtype=np.float32):
    inits.append(numpy_helper.from_array(np.array(arr, dtype=dtype), name)); return name

K('ch1', [1], np.int64); K('ch2', [2], np.int64); K('ax1', [1], np.int64)
K('ax2v', [2], np.int64); K('ax3v', [3], np.int64)
K('half', [0.5]); K('one_f', [1.0]); K('c1_5', [1.5]); K('m1f', [-1.0]); K('p999f', [999.0])
K('row_idx_f', np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1))
K('col_idx_f', np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30))
ch8sel = np.zeros((1, 10, 1, 1), np.float32); ch8sel[0, 8, 0, 0] = 1.0
K('ch8sel', ch8sel)

x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
nodes = [
    helper.make_node('Slice', ['input', 'ch1', 'ch2', 'ax1'], ['is1f']),
    helper.make_node('Greater', ['is1f', 'half'], ['is1_b']),
    helper.make_node('Cast', ['is1_b'], ['is1_fl'], to=1),
    # row-wise: count of dots, and their column min/max (sentinel +-999 where absent)
    helper.make_node('ReduceSum', ['is1_fl', 'ax3v'], ['rowcount'], keepdims=1),
    helper.make_node('Where', ['is1_b', 'col_idx_f', 'p999f'], ['rc_lo_w']),
    helper.make_node('ReduceMin', ['rc_lo_w'], ['rowmin'], axes=[3], keepdims=1),
    helper.make_node('Where', ['is1_b', 'col_idx_f', 'm1f'], ['rc_hi_w']),
    helper.make_node('ReduceMax', ['rc_hi_w'], ['rowmax'], axes=[3], keepdims=1),
    # column-wise: count of dots, and their row min/max
    helper.make_node('ReduceSum', ['is1_fl', 'ax2v'], ['colcount'], keepdims=1),
    helper.make_node('Where', ['is1_b', 'row_idx_f', 'p999f'], ['cr_lo_w']),
    helper.make_node('ReduceMin', ['cr_lo_w'], ['colmin'], axes=[2], keepdims=1),
    helper.make_node('Where', ['is1_b', 'row_idx_f', 'm1f'], ['cr_hi_w']),
    helper.make_node('ReduceMax', ['cr_hi_w'], ['colmax'], axes=[2], keepdims=1),
    # fill-between-row-dots mask
    helper.make_node('Greater', ['rowcount', 'c1_5'], ['row_ge2']),
    helper.make_node('Greater', ['col_idx_f', 'rowmin'], ['col_gt_min']),
    helper.make_node('Less', ['col_idx_f', 'rowmax'], ['col_lt_max']),
    helper.make_node('And', ['col_gt_min', 'col_lt_max'], ['col_between']),
    helper.make_node('And', ['row_ge2', 'col_between'], ['fillrow_pre']),
    helper.make_node('Not', ['is1_b'], ['not1_b']),
    helper.make_node('And', ['fillrow_pre', 'not1_b'], ['fillrow_b']),
    # fill-between-column-dots mask
    helper.make_node('Greater', ['colcount', 'c1_5'], ['col_ge2']),
    helper.make_node('Greater', ['row_idx_f', 'colmin'], ['row_gt_min']),
    helper.make_node('Less', ['row_idx_f', 'colmax'], ['row_lt_max']),
    helper.make_node('And', ['row_gt_min', 'row_lt_max'], ['row_between']),
    helper.make_node('And', ['col_ge2', 'row_between'], ['fillcol_pre']),
    helper.make_node('And', ['fillcol_pre', 'not1_b'], ['fillcol_b']),
    # combine + paint
    helper.make_node('Or', ['fillrow_b', 'fillcol_b'], ['fill_b']),
    helper.make_node('Cast', ['fill_b'], ['fill_f'], to=1),
    helper.make_node('Sub', ['one_f', 'fill_f'], ['keep_f']),
    helper.make_node('Mul', ['input', 'keep_f'], ['keep_part']),
    helper.make_node('Mul', ['fill_f', 'ch8sel'], ['add8']),
    helper.make_node('Add', ['keep_part', 'add8'], ['output']),
]
graph = helper.make_graph(nodes, 'task350', [x], [y], inits)
model = helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])
