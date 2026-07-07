# --- task008 (ARC 05f2a901): gravitate(colour-2 object, colour-8 object).
# Slide the 2-object toward the 8-object along their shared row/column band until the two
# bounding boxes are adjacent (touching, not overlapping), then repaint.
# YOUR LOGIC WAS 100% CORRECT (bbox gap -> dy/dx -> pad+dynamic-slice shift -> recompose).
# Two fixes, both from our playbook:
#   FIX 1 (grid size): OneHot(final_class) lit channel-0 across the whole 30x30 padding, so the
#     decoder never trimmed and returned a 30x30 grid instead of HxW. Multiply the output by the
#     grid-rectangle mask  M = ReduceSum(input over all 10 channels)  -> padding decodes as empty.
#   FIX 2 (cost=None): the shift Slice uses runtime-computed starts/ends, so ONNX can't infer a
#     static shape for its output -> mem=None -> task scores 0. Bake value_info [1,1,30,30] for it.
# Verified 266/266. (Cost ~204990 -> 12.77 pts vs baseline 5626 -> 16.36 pts, so the submission
#  keeps baseline; you still own a correct build.)
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_task008_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)

    inits = [
        K('c8_s', [8]), K('c8_e', [9]), K('c2_s', [2]), K('c2_e', [3]), K('ax_c', [1]),
        K('sq_01', [0, 1]), K('arange30', np.arange(30)), K('rev_arange30', np.arange(29, -1, -1)),
        K('c29', [29]), K('c1', [1]), K('c0', [0]), K('c30', [30]), K('c60', [60]), K('c2_scalar', [2]),
        K('pads_30', [0, 0, 30, 30, 0, 0, 30, 30]), K('depth10', 10),
        K('oh_vals', [0.0, 1.0], dtype=np.float32), K('ax_0_1_2_3', [0, 1, 2, 3]), K('rd_all', [1]),
    ]

    def get_bbox_nodes(prefix, mask_sq):
        # static bounding box via row/col projections (no NonZero):
        #   max index of a hit = max(proj * arange);  min index = 29 - max(proj * reverse_arange)
        return [
            helper.make_node('ReduceMax', [mask_sq], [f'{prefix}_r_proj'], axes=[1], keepdims=0),
            helper.make_node('Mul', [f'{prefix}_r_proj', 'arange30'], [f'{prefix}_r_mul1']),
            helper.make_node('ReduceMax', [f'{prefix}_r_mul1'], [f'{prefix}_r_max'], axes=[0], keepdims=0),
            helper.make_node('Mul', [f'{prefix}_r_proj', 'rev_arange30'], [f'{prefix}_r_mul2']),
            helper.make_node('ReduceMax', [f'{prefix}_r_mul2'], [f'{prefix}_r_min_inv'], axes=[0], keepdims=0),
            helper.make_node('Sub', ['c29', f'{prefix}_r_min_inv'], [f'{prefix}_r_min']),
            helper.make_node('ReduceMax', [mask_sq], [f'{prefix}_c_proj'], axes=[0], keepdims=0),
            helper.make_node('Mul', [f'{prefix}_c_proj', 'arange30'], [f'{prefix}_c_mul1']),
            helper.make_node('ReduceMax', [f'{prefix}_c_mul1'], [f'{prefix}_c_max'], axes=[0], keepdims=0),
            helper.make_node('Mul', [f'{prefix}_c_proj', 'rev_arange30'], [f'{prefix}_c_mul2']),
            helper.make_node('ReduceMax', [f'{prefix}_c_mul2'], [f'{prefix}_c_min_inv'], axes=[0], keepdims=0),
            helper.make_node('Sub', ['c29', f'{prefix}_c_min_inv'], [f'{prefix}_c_min']),
        ]

    nodes = [
        helper.make_node('Slice', ['input', 'c8_s', 'c8_e', 'ax_c'], ['ch8_f']),
        helper.make_node('Slice', ['input', 'c2_s', 'c2_e', 'ax_c'], ['ch2_f']),
        helper.make_node('Squeeze', ['ch8_f', 'sq_01'], ['ch8_f_sq']),
        helper.make_node('Squeeze', ['ch2_f', 'sq_01'], ['ch2_f_sq']),
        helper.make_node('Cast', ['ch8_f_sq'], ['ch8'], to=I64),
        helper.make_node('Cast', ['ch2_f_sq'], ['ch2'], to=I64),
    ]
    nodes += get_bbox_nodes('c8', 'ch8')
    nodes += get_bbox_nodes('c2', 'ch2')
    nodes += [
        # do the two boxes share a column band? a row band?
        helper.make_node('GreaterOrEqual', ['c2_c_max', 'c8_c_min'], ['col_ov1']),
        helper.make_node('LessOrEqual', ['c2_c_min', 'c8_c_max'], ['col_ov2']),
        helper.make_node('And', ['col_ov1', 'col_ov2'], ['col_ov']),
        helper.make_node('GreaterOrEqual', ['c2_r_max', 'c8_r_min'], ['row_ov1']),
        helper.make_node('LessOrEqual', ['c2_r_min', 'c8_r_max'], ['row_ov2']),
        helper.make_node('And', ['row_ov1', 'row_ov2'], ['row_ov']),
        # dy: vertical gap closure (only if columns overlap)
        helper.make_node('Less', ['c2_r_max', 'c8_r_min'], ['dy_up_cond']),
        helper.make_node('Sub', ['c8_r_min', 'c1'], ['t1']),
        helper.make_node('Sub', ['t1', 'c2_r_max'], ['dy_up_val']),
        helper.make_node('Cast', ['dy_up_cond'], ['dy_up_c'], to=I64),
        helper.make_node('Mul', ['dy_up_c', 'dy_up_val'], ['dy_up']),
        helper.make_node('Greater', ['c2_r_min', 'c8_r_max'], ['dy_dn_cond']),
        helper.make_node('Add', ['c8_r_max', 'c1'], ['t2']),
        helper.make_node('Sub', ['t2', 'c2_r_min'], ['dy_dn_val']),
        helper.make_node('Cast', ['dy_dn_cond'], ['dy_dn_c'], to=I64),
        helper.make_node('Mul', ['dy_dn_c', 'dy_dn_val'], ['dy_dn']),
        helper.make_node('Add', ['dy_up', 'dy_dn'], ['dy_raw']),
        helper.make_node('Cast', ['col_ov'], ['col_ov_c'], to=I64),
        helper.make_node('Mul', ['dy_raw', 'col_ov_c'], ['dy_cand']),
        # dx: horizontal gap closure (only if rows overlap)
        helper.make_node('Less', ['c2_c_max', 'c8_c_min'], ['dx_l_cond']),
        helper.make_node('Sub', ['c8_c_min', 'c1'], ['t3']),
        helper.make_node('Sub', ['t3', 'c2_c_max'], ['dx_l_val']),
        helper.make_node('Cast', ['dx_l_cond'], ['dx_l_c'], to=I64),
        helper.make_node('Mul', ['dx_l_c', 'dx_l_val'], ['dx_l']),
        helper.make_node('Greater', ['c2_c_min', 'c8_c_max'], ['dx_r_cond']),
        helper.make_node('Add', ['c8_c_max', 'c1'], ['t4']),
        helper.make_node('Sub', ['t4', 'c2_c_min'], ['dx_r_val']),
        helper.make_node('Cast', ['dx_r_cond'], ['dx_r_c'], to=I64),
        helper.make_node('Mul', ['dx_r_c', 'dx_r_val'], ['dx_r']),
        helper.make_node('Add', ['dx_l', 'dx_r'], ['dx_raw']),
        helper.make_node('Cast', ['row_ov'], ['row_ov_c'], to=I64),
        helper.make_node('Mul', ['dx_raw', 'row_ov_c'], ['dx_cand']),
        # only move if both colours are actually present
        helper.make_node('ReduceMax', ['ch8'], ['M8_max'], axes=[0, 1], keepdims=0),
        helper.make_node('ReduceMax', ['ch2'], ['M2_max'], axes=[0, 1], keepdims=0),
        helper.make_node('Greater', ['M8_max', 'c0'], ['p8']),
        helper.make_node('Greater', ['M2_max', 'c0'], ['p2']),
        helper.make_node('And', ['p8', 'p2'], ['both_p']),
        helper.make_node('Cast', ['both_p'], ['both_p_c'], to=I64),
        helper.make_node('Mul', ['dy_cand', 'both_p_c'], ['dy']),
        helper.make_node('Mul', ['dx_cand', 'both_p_c'], ['dx']),
        # translate the 2-mask by (dy,dx): pad by 30 then dynamic-slice back a 30x30 window
        helper.make_node('Pad', ['ch2_f', 'pads_30'], ['padded_ch2']),
        helper.make_node('Sub', ['c30', 'dy'], ['start_H']),
        helper.make_node('Sub', ['c60', 'dy'], ['end_H']),
        helper.make_node('Sub', ['c30', 'dx'], ['start_W']),
        helper.make_node('Sub', ['c60', 'dx'], ['end_W']),
        helper.make_node('Concat', ['c0', 'c0', 'start_H', 'start_W'], ['starts'], axis=0),
        helper.make_node('Concat', ['c1', 'c1', 'end_H', 'end_W'], ['ends'], axis=0),
        helper.make_node('Slice', ['padded_ch2', 'starts', 'ends', 'ax_0_1_2_3'], ['ch2_shifted_f']),
        # recompose: erase old 2s, paint shifted 2s over everything else
        helper.make_node('ArgMax', ['input'], ['class_grid_0'], axis=1, keepdims=0),
        helper.make_node('Sub', ['c1', 'ch2'], ['inv_ch2']),
        helper.make_node('Mul', ['class_grid_0', 'inv_ch2'], ['class_grid_1']),
        helper.make_node('Squeeze', ['ch2_shifted_f', 'sq_01'], ['ch2_shifted_f_sq']),
        helper.make_node('Cast', ['ch2_shifted_f_sq'], ['ch2_shifted'], to=I64),
        helper.make_node('Sub', ['c1', 'ch2_shifted'], ['inv_ch2_shifted']),
        helper.make_node('Mul', ['class_grid_1', 'inv_ch2_shifted'], ['class_grid_2']),
        helper.make_node('Mul', ['ch2_shifted', 'c2_scalar'], ['new_2s']),
        helper.make_node('Add', ['class_grid_2', 'new_2s'], ['final_class']),
        helper.make_node('OneHot', ['final_class', 'depth10', 'oh_vals'], ['output_raw'], axis=1),
        helper.make_node('Cast', ['output_raw'], ['oh_f'], to=F),
        # FIX 1 (grid size): keep only the actual grid rectangle so padding decodes as empty
        helper.make_node('ReduceSum', ['input', 'rd_all'], ['content'], keepdims=1),
        helper.make_node('Mul', ['oh_f', 'content'], ['output']),
    ]
    graph = helper.make_graph(nodes, 'task008', [x], [y], inits)
    # FIX 2 (cost=None): bake a static shape for the dynamically-sliced tensor
    graph.value_info.append(helper.make_tensor_value_info('ch2_shifted_f', F, [1, 1, 30, 30]))
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

model = create_task008_model()
