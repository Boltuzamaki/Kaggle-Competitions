# Source: predicted/test_onnx_task119.py — ONNX graph construction code
# Verified model: repairs/task119.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    F_type = TensorProto.FLOAT
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', F_type, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info('output', F_type, [1, 10, 30, 30])
    
    nodes = []
    inits = []
    
    def K(name, val, dtype=I64):
        val_arr = np.array(val)
        inits.append(helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist()))
        nodes.append(helper.make_node('Constant', [], [name], value=inits[-1]))
    
    K('c0', 0)
    K('c1', 1)
    K('c2', 2)
    K('c3', 3)
    K('c8', 8)
    K('c1000', 1000)
    K('cm1000', -1000)
    K('c0_f', 0.0, dtype=F_type)
    
    K('axes_0', [0])
    K('axes_1', [1])
    K('axes_2', [2])
    K('axes_3', [3])
    
    K('c0_1d', [0])
    K('c1_1d', [1])
    K('c29_1d', [29])
    K('c30_1d', [30])
    
    # ArgMax
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax_raw'], axis=1, keepdims=0)) # [1, 30, 30]
    nodes.append(helper.make_node('Squeeze', ['argmax_raw', 'axes_0'], ['argmax_grid'])) # [30, 30]
    
    # Presence
    nodes.append(helper.make_node('ReduceMax', ['input'], ['presence'], axes=[1], keepdims=1)) # [1, 1, 30, 30]
    nodes.append(helper.make_node('Squeeze', ['presence', 'axes_0'], ['presence_sq1'])) # [1, 30, 30]
    nodes.append(helper.make_node('Squeeze', ['presence_sq1', 'axes_0'], ['presence_sq'])) # [30, 30]
    
    # H and W
    nodes.append(helper.make_node('Slice', ['presence_sq', 'c0_1d', 'c1_1d', 'axes_0'], ['row0_f_2d']))
    nodes.append(helper.make_node('Slice', ['presence_sq', 'c0_1d', 'c1_1d', 'axes_1'], ['col0_f_2d']))
    nodes.append(helper.make_node('ReduceSum', ['row0_f_2d'], ['W_f'], keepdims=0))
    nodes.append(helper.make_node('ReduceSum', ['col0_f_2d'], ['H_f'], keepdims=0))
    nodes.append(helper.make_node('Cast', ['W_f'], ['W'], to=I64))
    nodes.append(helper.make_node('Cast', ['H_f'], ['H'], to=I64))
    
    nodes.append(helper.make_node('Sub', ['H', 'c1'], ['H_minus_1']))
    nodes.append(helper.make_node('Sub', ['W', 'c1'], ['W_minus_1']))
    
    K('c30', 30)
    # Indices
    nodes.append(helper.make_node('Range', ['c0', 'c30', 'c1'], ['r_indices']))
    nodes.append(helper.make_node('Range', ['c0', 'c30', 'c1'], ['c_indices']))
    
    K('shape_30_1', [30, 1])
    K('shape_1_30', [1, 30])
    nodes.append(helper.make_node('Reshape', ['r_indices', 'shape_30_1'], ['r_indices_col']))
    nodes.append(helper.make_node('Reshape', ['c_indices', 'shape_1_30'], ['c_indices_row']))
    
    # Endpoints of 8s
    nodes.append(helper.make_node('Equal', ['argmax_grid', 'c8'], ['mask8']))
    
    nodes.append(helper.make_node('Where', ['mask8', 'r_indices_col', 'c1000'], ['mask8_r_min']))
    nodes.append(helper.make_node('ReduceMin', ['mask8_r_min'], ['r8_min'], axes=[0, 1], keepdims=0))
    
    nodes.append(helper.make_node('Where', ['mask8', 'r_indices_col', 'cm1000'], ['mask8_r_max']))
    nodes.append(helper.make_node('ReduceMax', ['mask8_r_max'], ['r8_max'], axes=[0, 1], keepdims=0))
    
    nodes.append(helper.make_node('Equal', ['r_indices_col', 'r8_min'], ['is_rmin']))
    nodes.append(helper.make_node('And', ['is_rmin', 'mask8'], ['mask8_at_rmin']))
    nodes.append(helper.make_node('Where', ['mask8_at_rmin', 'c_indices_row', 'cm1000'], ['c_at_rmin_raw']))
    nodes.append(helper.make_node('ReduceMax', ['c_at_rmin_raw'], ['c_at_rmin'], axes=[0, 1], keepdims=0))
    
    nodes.append(helper.make_node('Equal', ['r_indices_col', 'r8_max'], ['is_rmax']))
    nodes.append(helper.make_node('And', ['is_rmax', 'mask8'], ['mask8_at_rmax']))
    nodes.append(helper.make_node('Where', ['mask8_at_rmax', 'c_indices_row', 'cm1000'], ['c_at_rmax_raw']))
    nodes.append(helper.make_node('ReduceMax', ['c_at_rmax_raw'], ['c_at_rmax'], axes=[0, 1], keepdims=0))
    
    # Distances
    nodes.append(helper.make_node('Sub', ['H_minus_1', 'r8_min'], ['d_rmin_top']))
    nodes.append(helper.make_node('Min', ['r8_min', 'd_rmin_top'], ['d_min_r']))
    nodes.append(helper.make_node('Sub', ['W_minus_1', 'c_at_rmin'], ['d_cmin_top']))
    nodes.append(helper.make_node('Min', ['c_at_rmin', 'd_cmin_top'], ['d_min_c']))
    nodes.append(helper.make_node('Min', ['d_min_r', 'd_min_c'], ['d_min']))
    
    nodes.append(helper.make_node('Sub', ['H_minus_1', 'r8_max'], ['d_rmax_bot']))
    nodes.append(helper.make_node('Min', ['r8_max', 'd_rmax_bot'], ['d_max_r']))
    nodes.append(helper.make_node('Sub', ['W_minus_1', 'c_at_rmax'], ['d_cmax_bot']))
    nodes.append(helper.make_node('Min', ['c_at_rmax', 'd_cmax_bot'], ['d_max_c']))
    nodes.append(helper.make_node('Min', ['d_max_r', 'd_max_c'], ['d_max']))
    
    nodes.append(helper.make_node('Less', ['d_min', 'd_max'], ['d_min_less']))
    
    # P_edge and P_start
    nodes.append(helper.make_node('Where', ['d_min_less', 'r8_min', 'r8_max'], ['P_edge_r']))
    nodes.append(helper.make_node('Where', ['d_min_less', 'c_at_rmin', 'c_at_rmax'], ['P_edge_c']))
    nodes.append(helper.make_node('Where', ['d_min_less', 'r8_max', 'r8_min'], ['P_start_r']))
    nodes.append(helper.make_node('Where', ['d_min_less', 'c_at_rmax', 'c_at_rmin'], ['P_start_c']))
    
    # Direction
    nodes.append(helper.make_node('Sub', ['P_start_r', 'P_edge_r'], ['dr_raw']))
    nodes.append(helper.make_node('Sub', ['P_start_c', 'P_edge_c'], ['dc_raw']))
    nodes.append(helper.make_node('Sign', ['dr_raw'], ['dr']))
    nodes.append(helper.make_node('Sign', ['dc_raw'], ['dc']))
    
    # Ray generation
    K('c41', 41)
    nodes.append(helper.make_node('Range', ['c1', 'c41', 'c1'], ['t_raw'])) # [40]
    
    nodes.append(helper.make_node('Mul', ['t_raw', 'dr'], ['t_dr']))
    nodes.append(helper.make_node('Add', ['P_start_r', 't_dr'], ['straight_r'])) # [40]
    
    nodes.append(helper.make_node('Mul', ['t_raw', 'dc'], ['t_dc']))
    nodes.append(helper.make_node('Add', ['P_start_c', 't_dc'], ['straight_c'])) # [40]
    
    # Mirrors
    nodes.append(helper.make_node('Equal', ['argmax_grid', 'c2'], ['is_2']))
    nodes.append(helper.make_node('Not', ['is_2'], ['mask_free']))
    nodes.append(helper.make_node('Greater', ['presence_sq', 'c0_f'], ['is_valid_cell']))
    nodes.append(helper.make_node('And', ['mask_free', 'is_valid_cell'], ['mask_free_true']))
    
    nodes.append(helper.make_node('Where', ['mask_free_true', 'r_indices_col', 'c1000'], ['free_r_min_raw']))
    nodes.append(helper.make_node('ReduceMin', ['free_r_min_raw'], ['free_r_min'], axes=[0, 1], keepdims=0))
    nodes.append(helper.make_node('Where', ['mask_free_true', 'r_indices_col', 'cm1000'], ['free_r_max_raw']))
    nodes.append(helper.make_node('ReduceMax', ['free_r_max_raw'], ['free_r_max'], axes=[0, 1], keepdims=0))
    
    nodes.append(helper.make_node('Where', ['mask_free_true', 'c_indices_row', 'c1000'], ['free_c_min_raw']))
    nodes.append(helper.make_node('ReduceMin', ['free_c_min_raw'], ['free_c_min'], axes=[0, 1], keepdims=0))
    nodes.append(helper.make_node('Where', ['mask_free_true', 'c_indices_row', 'cm1000'], ['free_c_max_raw']))
    nodes.append(helper.make_node('ReduceMax', ['free_c_max_raw'], ['free_c_max'], axes=[0, 1], keepdims=0))
    
    nodes.append(helper.make_node('Greater', ['free_r_min', 'c0'], ['r_min_gt_0']))
    nodes.append(helper.make_node('Where', ['r_min_gt_0', 'free_r_min', 'cm1000'], ['r_mirror_min']))
    nodes.append(helper.make_node('Less', ['free_r_max', 'H_minus_1'], ['r_max_lt_H']))
    nodes.append(helper.make_node('Where', ['r_max_lt_H', 'free_r_max', 'c1000'], ['r_mirror_max']))
    
    nodes.append(helper.make_node('Greater', ['free_c_min', 'c0'], ['c_min_gt_0']))
    nodes.append(helper.make_node('Where', ['c_min_gt_0', 'free_c_min', 'cm1000'], ['c_mirror_min']))
    nodes.append(helper.make_node('Less', ['free_c_max', 'W_minus_1'], ['c_max_lt_W']))
    nodes.append(helper.make_node('Where', ['c_max_lt_W', 'free_c_max', 'c1000'], ['c_mirror_max']))
    
    # Reflection
    nodes.append(helper.make_node('Greater', ['straight_r', 'r_mirror_max'], ['r_gt_max']))
    nodes.append(helper.make_node('Mul', ['r_mirror_max', 'c2'], ['r_mirror_max_2']))
    nodes.append(helper.make_node('Sub', ['r_mirror_max_2', 'straight_r'], ['refl_r_1']))
    nodes.append(helper.make_node('Where', ['r_gt_max', 'refl_r_1', 'straight_r'], ['refl_r_step1']))
    
    nodes.append(helper.make_node('Less', ['refl_r_step1', 'r_mirror_min'], ['r_lt_min']))
    nodes.append(helper.make_node('Mul', ['r_mirror_min', 'c2'], ['r_mirror_min_2']))
    nodes.append(helper.make_node('Sub', ['r_mirror_min_2', 'refl_r_step1'], ['refl_r_2']))
    nodes.append(helper.make_node('Where', ['r_lt_min', 'refl_r_2', 'refl_r_step1'], ['refl_r'])) # [40]
    
    nodes.append(helper.make_node('Greater', ['straight_c', 'c_mirror_max'], ['c_gt_max']))
    nodes.append(helper.make_node('Mul', ['c_mirror_max', 'c2'], ['c_mirror_max_2']))
    nodes.append(helper.make_node('Sub', ['c_mirror_max_2', 'straight_c'], ['refl_c_1']))
    nodes.append(helper.make_node('Where', ['c_gt_max', 'refl_c_1', 'straight_c'], ['refl_c_step1']))
    
    nodes.append(helper.make_node('Less', ['refl_c_step1', 'c_mirror_min'], ['c_lt_min']))
    nodes.append(helper.make_node('Mul', ['c_mirror_min', 'c2'], ['c_mirror_min_2']))
    nodes.append(helper.make_node('Sub', ['c_mirror_min_2', 'refl_c_step1'], ['refl_c_2']))
    nodes.append(helper.make_node('Where', ['c_lt_min', 'refl_c_2', 'refl_c_step1'], ['refl_c'])) # [40]
    
    # Validity
    nodes.append(helper.make_node('GreaterOrEqual', ['refl_r', 'c0'], ['r_ge_0']))
    nodes.append(helper.make_node('Less', ['refl_r', 'H'], ['r_lt_H']))
    nodes.append(helper.make_node('And', ['r_ge_0', 'r_lt_H'], ['valid_r']))
    
    nodes.append(helper.make_node('GreaterOrEqual', ['refl_c', 'c0'], ['c_ge_0']))
    nodes.append(helper.make_node('Less', ['refl_c', 'W'], ['c_lt_W']))
    nodes.append(helper.make_node('And', ['c_ge_0', 'c_lt_W'], ['valid_c']))
    
    nodes.append(helper.make_node('And', ['valid_r', 'valid_c'], ['valid_pt'])) # [40]
    
    # Grid broadcast to draw
    # Let's use Unsqueeze!
    nodes.append(helper.make_node('Unsqueeze', ['refl_r', 'axes_0'], ['refl_r_u1']))
    nodes.append(helper.make_node('Unsqueeze', ['refl_r_u1', 'axes_0'], ['refl_r_u2'])) # [1, 1, 40]
    
    nodes.append(helper.make_node('Unsqueeze', ['refl_c', 'axes_0'], ['refl_c_u1']))
    nodes.append(helper.make_node('Unsqueeze', ['refl_c_u1', 'axes_0'], ['refl_c_u2'])) # [1, 1, 40]
    
    nodes.append(helper.make_node('Unsqueeze', ['valid_pt', 'axes_0'], ['valid_pt_1']))
    nodes.append(helper.make_node('Unsqueeze', ['valid_pt_1', 'axes_0'], ['valid_pt_2'])) # [1, 1, 40]
    
    nodes.append(helper.make_node('Unsqueeze', ['r_indices_col', 'axes_2'], ['r_grid_3d'])) # [30, 1, 1]
    nodes.append(helper.make_node('Unsqueeze', ['c_indices_row', 'axes_2'], ['c_grid_3d'])) # [1, 30, 1]
    
    nodes.append(helper.make_node('Equal', ['refl_r_u2', 'r_grid_3d'], ['match_r'])) # [30, 1, 40]
    nodes.append(helper.make_node('Equal', ['refl_c_u2', 'c_grid_3d'], ['match_c'])) # [1, 30, 40]
    
    nodes.append(helper.make_node('And', ['match_r', 'match_c'], ['match_pt'])) # [30, 30, 40]
    nodes.append(helper.make_node('And', ['match_pt', 'valid_pt_2'], ['match_pt_valid']))
    nodes.append(helper.make_node('Cast', ['match_pt_valid'], ['match_pt_i64'], to=I64))
    
    nodes.append(helper.make_node('ReduceMax', ['match_pt_i64'], ['ray_mask_raw'], axes=[2], keepdims=0)) # [30, 30]
    nodes.append(helper.make_node('Greater', ['ray_mask_raw', 'c0'], ['ray_mask_bool']))
    nodes.append(helper.make_node('Where', ['ray_mask_bool', 'c3', 'c0'], ['ray_img']))
    
    # Canvas
    nodes.append(helper.make_node('Max', ['argmax_grid', 'ray_img'], ['out_grid']))
    
    nodes.append(helper.make_node('Unsqueeze', ['out_grid', 'axes_0'], ['out_grid_1x30x30']))
    K('depth10', [10])
    K('oh_vals', [0.0, 1.0], dtype=F_type)
    nodes.append(helper.make_node('OneHot', ['out_grid_1x30x30', 'depth10', 'oh_vals'], ['oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['oh'], ['raw_output'], perm=[0, 3, 1, 2]))
    
    # Mask padding
    nodes.append(helper.make_node('Mul', ['raw_output', 'presence'], ['output']))
    
    graph = helper.make_graph(nodes, 'task119_graph', [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name='task119_model', opset_imports=[helper.make_opsetid('', 13)])
    onnx.save(model, 'task119.onnx')


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task119.onnx")
