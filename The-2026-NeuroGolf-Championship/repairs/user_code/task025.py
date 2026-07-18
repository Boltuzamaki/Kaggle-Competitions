# --- task025 (ARC 1a07d186): gravitate each stray dot to the frontier line of its own colour.
# Each colour may have one full-grid frontier line (a complete row or column). Every single dot
# whose colour matches a line slides until it is adjacent to that line; dots whose colour has NO
# line are removed. Lines stay put.
# YOUR PIPELINE WAS RIGHT (per-colour line detect -> scattered dots -> move +/-1 to the line).
# Two fixes:
#   FIX A (real logic bug): line detection used `sum > 2`, which misreads 3-4 ALIGNED stray dots
#     as a "line". A true frontier spans the WHOLE grid, so: h-line <=> row_sum == grid_W and
#     v-line <=> col_sum == grid_H. grid_W/grid_H come from the content mask. Without this,
#     aligned dots freeze in place (or a phantom line attracts/keeps dots). (fixed arc-gen 12 & 42)
#   FIX B (canvas trap): OneHot + c0=0.5 tie-break lights channel-0 across the whole 30x30, so the
#     decoder returned a 30x30 grid. Multiply by the grid mask  M = ReduceSum(input over channels).
# Verified 266/266. (Cost ~585569 -> 11.72 pts vs baseline 11664 -> 15.64 pts, so the submission
#  keeps baseline; you still own a correct build.)
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    BOOL = TensorProto.BOOL
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
    inits = [
        K('ax_1', [1]), K('ax_2', [2]), K('ax_3', [3]),
        K('c1_i', [1], dtype=np.int64),
        K('cols', np.arange(30).reshape(1, 1, 1, 30), dtype=np.int64),
        K('rows', np.arange(30).reshape(1, 1, 30, 1), dtype=np.int64),
        K('c0', 0.5 * np.ones((1, 1, 30, 30), dtype=np.float32), dtype=np.float32),
        K('depth10', 10), K('oh_vals', [0.0, 1.0], dtype=np.float32),
    ]
    nodes = []
    out_channels = ['c0']
    # grid width W and height H from the content mask -> a real frontier spans the WHOLE grid,
    # so line <=> count == W (row) / == H (col); the old ">2" misreads aligned dots as lines.
    nodes.append(helper.make_node('ReduceSum', ['input', 'ax_1'], ['gm'], keepdims=1))          # [1,1,30,30]
    nodes.append(helper.make_node('ReduceSum', ['gm', 'ax_3'], ['row_content'], keepdims=1))     # [1,1,30,1]
    nodes.append(helper.make_node('ReduceMax', ['row_content'], ['grid_W'], axes=[2], keepdims=1))
    nodes.append(helper.make_node('ReduceSum', ['gm', 'ax_2'], ['col_content'], keepdims=1))     # [1,1,1,30]
    nodes.append(helper.make_node('ReduceMax', ['col_content'], ['grid_H'], axes=[3], keepdims=1))
    for k in range(1, 10):
        inits.extend([K(f'k_s_{k}', [k]), K(f'k_e_{k}', [k + 1])])
        nodes.append(helper.make_node('Slice', ['input', f'k_s_{k}', f'k_e_{k}', 'ax_1'], [f'is_k_{k}']))
        nodes.append(helper.make_node('Cast', [f'is_k_{k}'], [f'is_k_bool_{k}'], to=BOOL))
        nodes.append(helper.make_node('ReduceSum', [f'is_k_{k}', 'ax_2'], [f'col_sum_{k}'], keepdims=1))
        nodes.append(helper.make_node('ReduceSum', [f'is_k_{k}', 'ax_3'], [f'row_sum_{k}'], keepdims=1))
        # FIX A: frontier <=> spans the whole grid dimension
        nodes.append(helper.make_node('Equal', [f'col_sum_{k}', 'grid_H'], [f'v_line_mask_{k}']))
        nodes.append(helper.make_node('Equal', [f'row_sum_{k}', 'grid_W'], [f'h_line_mask_{k}']))
        nodes.append(helper.make_node('Cast', [f'v_line_mask_{k}'], [f'v_line_mask_i_{k}'], to=I64))
        nodes.append(helper.make_node('Cast', [f'h_line_mask_{k}'], [f'h_line_mask_i_{k}'], to=I64))
        nodes.append(helper.make_node('ReduceMax', [f'v_line_mask_i_{k}'], [f'has_v_{k}'], axes=[3], keepdims=1))
        nodes.append(helper.make_node('ReduceMax', [f'h_line_mask_i_{k}'], [f'has_h_{k}'], axes=[2], keepdims=1))
        nodes.append(helper.make_node('Cast', [f'has_v_{k}'], [f'has_v_bool_{k}'], to=BOOL))
        nodes.append(helper.make_node('Cast', [f'has_h_{k}'], [f'has_h_bool_{k}'], to=BOOL))
        # scattered = this colour, not lying on any of its lines
        nodes.append(helper.make_node('Not', [f'v_line_mask_{k}'], [f'not_v_{k}']))
        nodes.append(helper.make_node('Not', [f'h_line_mask_{k}'], [f'not_h_{k}']))
        nodes.append(helper.make_node('And', [f'not_v_{k}', f'not_h_{k}'], [f'not_lines_{k}']))
        nodes.append(helper.make_node('And', [f'is_k_bool_{k}', f'not_lines_{k}'], [f'scat_k_{k}']))
        # vertical move: to column C+/-1 of the v-line
        nodes.append(helper.make_node('ArgMax', [f'v_line_mask_i_{k}'], [f'C_{k}'], axis=3, keepdims=1))
        nodes.append(helper.make_node('Less', ['cols', f'C_{k}'], [f'left_of_C_{k}']))
        nodes.append(helper.make_node('Greater', ['cols', f'C_{k}'], [f'right_of_C_{k}']))
        nodes.append(helper.make_node('And', [f'scat_k_{k}', f'left_of_C_{k}'], [f'scat_left_{k}']))
        nodes.append(helper.make_node('And', [f'scat_k_{k}', f'right_of_C_{k}'], [f'scat_right_{k}']))
        nodes.append(helper.make_node('Cast', [f'scat_left_{k}'], [f'scat_left_i_{k}'], to=I64))
        nodes.append(helper.make_node('Cast', [f'scat_right_{k}'], [f'scat_right_i_{k}'], to=I64))
        nodes.append(helper.make_node('ReduceMax', [f'scat_left_i_{k}'], [f'has_scat_left_{k}'], axes=[3], keepdims=1))
        nodes.append(helper.make_node('ReduceMax', [f'scat_right_i_{k}'], [f'has_scat_right_{k}'], axes=[3], keepdims=1))
        nodes.append(helper.make_node('Cast', [f'has_scat_left_{k}'], [f'has_scat_left_b_{k}'], to=BOOL))
        nodes.append(helper.make_node('Cast', [f'has_scat_right_{k}'], [f'has_scat_right_b_{k}'], to=BOOL))
        nodes.append(helper.make_node('Sub', [f'C_{k}', 'c1_i'], [f'C_m1_{k}']))
        nodes.append(helper.make_node('Add', [f'C_{k}', 'c1_i'], [f'C_p1_{k}']))
        nodes.append(helper.make_node('Equal', ['cols', f'C_m1_{k}'], [f'is_C_m1_{k}']))
        nodes.append(helper.make_node('Equal', ['cols', f'C_p1_{k}'], [f'is_C_p1_{k}']))
        nodes.append(helper.make_node('And', [f'has_scat_left_b_{k}', f'is_C_m1_{k}'], [f'moved_left_{k}']))
        nodes.append(helper.make_node('And', [f'has_scat_right_b_{k}', f'is_C_p1_{k}'], [f'moved_right_{k}']))
        nodes.append(helper.make_node('Or', [f'moved_left_{k}', f'moved_right_{k}'], [f'moved_v_any_{k}']))
        nodes.append(helper.make_node('And', [f'moved_v_any_{k}', f'has_v_bool_{k}'], [f'moved_v_{k}']))
        # horizontal move: to row R+/-1 of the h-line
        nodes.append(helper.make_node('ArgMax', [f'h_line_mask_i_{k}'], [f'R_{k}'], axis=2, keepdims=1))
        nodes.append(helper.make_node('Less', ['rows', f'R_{k}'], [f'above_R_{k}']))
        nodes.append(helper.make_node('Greater', ['rows', f'R_{k}'], [f'below_R_{k}']))
        nodes.append(helper.make_node('And', [f'scat_k_{k}', f'above_R_{k}'], [f'scat_above_{k}']))
        nodes.append(helper.make_node('And', [f'scat_k_{k}', f'below_R_{k}'], [f'scat_below_{k}']))
        nodes.append(helper.make_node('Cast', [f'scat_above_{k}'], [f'scat_above_i_{k}'], to=I64))
        nodes.append(helper.make_node('Cast', [f'scat_below_{k}'], [f'scat_below_i_{k}'], to=I64))
        nodes.append(helper.make_node('ReduceMax', [f'scat_above_i_{k}'], [f'has_scat_above_{k}'], axes=[2], keepdims=1))
        nodes.append(helper.make_node('ReduceMax', [f'scat_below_i_{k}'], [f'has_scat_below_{k}'], axes=[2], keepdims=1))
        nodes.append(helper.make_node('Cast', [f'has_scat_above_{k}'], [f'has_scat_above_b_{k}'], to=BOOL))
        nodes.append(helper.make_node('Cast', [f'has_scat_below_{k}'], [f'has_scat_below_b_{k}'], to=BOOL))
        nodes.append(helper.make_node('Sub', [f'R_{k}', 'c1_i'], [f'R_m1_{k}']))
        nodes.append(helper.make_node('Add', [f'R_{k}', 'c1_i'], [f'R_p1_{k}']))
        nodes.append(helper.make_node('Equal', ['rows', f'R_m1_{k}'], [f'is_R_m1_{k}']))
        nodes.append(helper.make_node('Equal', ['rows', f'R_p1_{k}'], [f'is_R_p1_{k}']))
        nodes.append(helper.make_node('And', [f'has_scat_above_b_{k}', f'is_R_m1_{k}'], [f'moved_above_{k}']))
        nodes.append(helper.make_node('And', [f'has_scat_below_b_{k}', f'is_R_p1_{k}'], [f'moved_below_{k}']))
        nodes.append(helper.make_node('Or', [f'moved_above_{k}', f'moved_below_{k}'], [f'moved_h_any_{k}']))
        nodes.append(helper.make_node('And', [f'moved_h_any_{k}', f'has_h_bool_{k}'], [f'moved_h_{k}']))
        # combine: keep the line cells + the moved dots
        nodes.append(helper.make_node('Or', [f'v_line_mask_{k}', f'h_line_mask_{k}'], [f'line_mask_{k}']))
        nodes.append(helper.make_node('And', [f'is_k_bool_{k}', f'line_mask_{k}'], [f'base_{k}']))
        nodes.append(helper.make_node('Or', [f'moved_v_{k}', f'moved_h_{k}'], [f'moved_{k}']))
        nodes.append(helper.make_node('Or', [f'base_{k}', f'moved_{k}'], [f'out_b_{k}']))
        nodes.append(helper.make_node('Cast', [f'out_b_{k}'], [f'out_{k}'], to=F))
        out_channels.append(f'out_{k}')
    nodes.append(helper.make_node('Concat', out_channels, ['out_logits'], axis=1))
    nodes.append(helper.make_node('ArgMax', ['out_logits'], ['pred_idx'], axis=1, keepdims=0))
    nodes.append(helper.make_node('OneHot', ['pred_idx', 'depth10', 'oh_vals'], ['oh_f'], axis=1))
    # FIX B (canvas trap): keep only the actual grid rectangle so padding decodes as empty
    nodes.append(helper.make_node('ReduceSum', ['input', 'ax_1'], ['content'], keepdims=1))
    nodes.append(helper.make_node('Mul', ['oh_f', 'content'], ['output']))
    graph = helper.make_graph(nodes, 'task025', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

model = create_model()
