# Source: predicted/test_onnx_task107.py — ONNX graph construction code
# Verified model: repairs/task107.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

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

    K('c0', [0])
    K('c1', [1])
    K('c2', [2])
    K('c4', [4])
    K('c5', [5])
    K('cm1', [-1])
    K('axes_0', [0])
    K('axes_1', [1])
    K('axes_2', [2])
    
    # ArgMax over input channels
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=0))
    # Slice out the 5x5 region
    nodes.append(helper.make_node('Slice', ['argmax', 'c0', 'c5', 'axes_1'], ['argmax_5_r']))
    nodes.append(helper.make_node('Slice', ['argmax_5_r', 'c0', 'c5', 'axes_2'], ['in_5x5']))
    
    # Calculate F = number of unique non-zero colors in in_5x5
    nodes.append(helper.make_node('Equal', ['in_5x5', 'c1'], ['eq_1']))
    nodes.append(helper.make_node('Cast', ['eq_1'], ['eq_1_i64'], to=I64))
    nodes.append(helper.make_node('ReduceMax', ['eq_1_i64'], ['has_1'], keepdims=1))
    
    curr_sum = 'has_1'
    for c in range(2, 10):
        K(f'color_{c}', [c])
        nodes.append(helper.make_node('Equal', ['in_5x5', f'color_{c}'], [f'eq_{c}']))
        nodes.append(helper.make_node('Cast', [f'eq_{c}'], [f'eq_{c}_i64'], to=I64))
        nodes.append(helper.make_node('ReduceMax', [f'eq_{c}_i64'], [f'has_{c}'], keepdims=1))
        nodes.append(helper.make_node('Add', [curr_sum, f'has_{c}'], [f'sum_{c}']))
        curr_sum = f'sum_{c}'
        
    nodes.append(helper.make_node('Identity', [curr_sum], ['F'])) # shape [1, 1, 1]
    nodes.append(helper.make_node('Squeeze', ['F', 'axes_1'], ['F_1'])) # shape [1, 1]
    nodes.append(helper.make_node('Squeeze', ['F_1', 'axes_1'], ['F_0'])) # shape [1]
    
    # Scale input statically using Div and Gather
    nodes.append(helper.make_node('Mul', ['F_0', 'c5'], ['out_dim'])) # shape [1]
    
    # Bounding box in original 4x4
    nodes.append(helper.make_node('Slice', ['in_5x5', 'c0', 'c4', 'axes_1'], ['in_4_r']))
    nodes.append(helper.make_node('Slice', ['in_4_r', 'c0', 'c4', 'axes_2'], ['in_4x4']))
    
    nodes.append(helper.make_node('Greater', ['in_4x4', 'c0'], ['is_shape_bool']))
    nodes.append(helper.make_node('Cast', ['is_shape_bool'], ['is_shape'], to=I64))
    
    nodes.append(helper.make_node('ReduceMax', ['is_shape'], ['r_has_shape'], axes=[2], keepdims=1))
    nodes.append(helper.make_node('ReduceMax', ['is_shape'], ['c_has_shape'], axes=[1], keepdims=1))
    
    nodes.append(helper.make_node('Greater', ['r_has_shape', 'c0'], ['r_has_shape_bool']))
    nodes.append(helper.make_node('Greater', ['c_has_shape', 'c0'], ['c_has_shape_bool']))
    
    K('indices_4', np.arange(4, dtype=np.int64).reshape(1, 4, 1))
    K('indices_4_c', np.arange(4, dtype=np.int64).reshape(1, 1, 4))
    
    nodes.append(helper.make_node('Where', ['r_has_shape_bool', 'indices_4', 'c4'], ['r_valid_min']))
    nodes.append(helper.make_node('ReduceMin', ['r_valid_min'], ['r_min_orig'], axes=[1, 2], keepdims=1))
    nodes.append(helper.make_node('Where', ['r_has_shape_bool', 'indices_4', 'cm1'], ['r_valid_max']))
    nodes.append(helper.make_node('ReduceMax', ['r_valid_max'], ['r_max_orig'], axes=[1, 2], keepdims=1))
    
    nodes.append(helper.make_node('Where', ['c_has_shape_bool', 'indices_4_c', 'c4'], ['c_valid_min']))
    nodes.append(helper.make_node('ReduceMin', ['c_valid_min'], ['c_min_orig'], axes=[1, 2], keepdims=1))
    nodes.append(helper.make_node('Where', ['c_has_shape_bool', 'indices_4_c', 'cm1'], ['c_valid_max']))
    nodes.append(helper.make_node('ReduceMax', ['c_valid_max'], ['c_max_orig'], axes=[1, 2], keepdims=1))
    
    # Scale bounding box
    nodes.append(helper.make_node('Mul', ['r_min_orig', 'F_0'], ['r_min']))
    nodes.append(helper.make_node('Add', ['r_max_orig', 'c1'], ['r_max_orig_1']))
    nodes.append(helper.make_node('Mul', ['r_max_orig_1', 'F_0'], ['r_max_1']))
    nodes.append(helper.make_node('Sub', ['r_max_1', 'c1'], ['r_max']))
    
    nodes.append(helper.make_node('Mul', ['c_min_orig', 'F_0'], ['c_min']))
    nodes.append(helper.make_node('Add', ['c_max_orig', 'c1'], ['c_max_orig_1']))
    nodes.append(helper.make_node('Mul', ['c_max_orig_1', 'F_0'], ['c_max_1']))
    nodes.append(helper.make_node('Sub', ['c_max_1', 'c1'], ['c_max']))
    
    # Static 30x30 grids
    K('R_30x30', np.arange(30).reshape(1, 30, 1))
    K('C_30x30', np.arange(30).reshape(1, 1, 30))
    
    # Generate scaled_padded using Div and Gather
    nodes.append(helper.make_node('Div', ['R_30x30', 'F_0'], ['R_idx']))
    nodes.append(helper.make_node('Div', ['C_30x30', 'F_0'], ['C_idx']))
    
    nodes.append(helper.make_node('Clip', ['R_idx', 'c0', 'c4'], ['R_idx_clamped']))
    nodes.append(helper.make_node('Clip', ['C_idx', 'c0', 'c4'], ['C_idx_clamped']))
    
    nodes.append(helper.make_node('Squeeze', ['in_5x5', 'axes_0'], ['in_5x5_2d']))
    nodes.append(helper.make_node('Squeeze', ['R_idx_clamped', 'axes_0'], ['R_idx_2d']))
    nodes.append(helper.make_node('Squeeze', ['C_idx_clamped', 'axes_0'], ['C_idx_2d']))
    
    nodes.append(helper.make_node('Gather', ['in_5x5_2d', 'R_idx_2d'], ['gather_R'], axis=0))
    nodes.append(helper.make_node('Squeeze', ['gather_R', 'axes_1'], ['gather_R_2d']))
    nodes.append(helper.make_node('Gather', ['gather_R_2d', 'C_idx_2d'], ['gather_RC'], axis=1))
    nodes.append(helper.make_node('Squeeze', ['gather_RC', 'axes_1'], ['scaled_padded_2d']))
    nodes.append(helper.make_node('Unsqueeze', ['scaled_padded_2d', 'axes_0'], ['scaled_padded']))
    
    # Bounds for the valid output dimension
    nodes.append(helper.make_node('Less', ['R_30x30', 'out_dim'], ['in_h']))
    nodes.append(helper.make_node('Less', ['C_30x30', 'out_dim'], ['in_w']))
    nodes.append(helper.make_node('And', ['in_h', 'in_w'], ['in_bounds_dim']))
    
    # Apply in_bounds_dim mask
    nodes.append(helper.make_node('Where', ['in_bounds_dim', 'scaled_padded', 'c0'], ['scaled_padded_masked']))
    
    # Now draw lines on the 30x30 grid!
    nodes.append(helper.make_node('Sub', ['R_30x30', 'r_min'], ['R_sub_rmin']))
    nodes.append(helper.make_node('Sub', ['C_30x30', 'c_min'], ['C_sub_cmin']))
    nodes.append(helper.make_node('Sub', ['R_30x30', 'r_max'], ['R_sub_rmax']))
    nodes.append(helper.make_node('Sub', ['C_30x30', 'c_max'], ['C_sub_cmax']))
    
    nodes.append(helper.make_node('Neg', ['C_sub_cmax'], ['neg_C_sub_cmax']))
    nodes.append(helper.make_node('Neg', ['C_sub_cmin'], ['neg_C_sub_cmin']))
    
    # Line 1: (R - r_min) == (C - c_min) and R < r_min and C < c_min
    nodes.append(helper.make_node('Equal', ['R_sub_rmin', 'C_sub_cmin'], ['eq1']))
    nodes.append(helper.make_node('Less', ['R_30x30', 'r_min'], ['lt_r1']))
    nodes.append(helper.make_node('Less', ['C_30x30', 'c_min'], ['lt_c1']))
    nodes.append(helper.make_node('And', ['lt_r1', 'lt_c1'], ['cond1']))
    nodes.append(helper.make_node('And', ['eq1', 'cond1'], ['L1']))
    
    # Line 2: (R - r_min) == -(C - c_max) and R < r_min and C > c_max
    nodes.append(helper.make_node('Equal', ['R_sub_rmin', 'neg_C_sub_cmax'], ['eq2']))
    nodes.append(helper.make_node('Greater', ['C_30x30', 'c_max'], ['gt_c2']))
    nodes.append(helper.make_node('And', ['lt_r1', 'gt_c2'], ['cond2']))
    nodes.append(helper.make_node('And', ['eq2', 'cond2'], ['L2']))
    
    # Line 3: (R - r_max) == -(C - c_min) and R > r_max and C < c_min
    nodes.append(helper.make_node('Equal', ['R_sub_rmax', 'neg_C_sub_cmin'], ['eq3']))
    nodes.append(helper.make_node('Greater', ['R_30x30', 'r_max'], ['gt_r3']))
    nodes.append(helper.make_node('And', ['gt_r3', 'lt_c1'], ['cond3']))
    nodes.append(helper.make_node('And', ['eq3', 'cond3'], ['L3']))
    
    # Line 4: (R - r_max) == (C - c_max) and R > r_max and C > c_max
    nodes.append(helper.make_node('Equal', ['R_sub_rmax', 'C_sub_cmax'], ['eq4']))
    nodes.append(helper.make_node('And', ['gt_r3', 'gt_c2'], ['cond4']))
    nodes.append(helper.make_node('And', ['eq4', 'cond4'], ['L4']))
    
    nodes.append(helper.make_node('Or', ['L1', 'L2'], ['L12']))
    nodes.append(helper.make_node('Or', ['L3', 'L4'], ['L34']))
    nodes.append(helper.make_node('Or', ['L12', 'L34'], ['is_line']))
    
    # In bounds (4F x 4F) -> wait, the original logic uses 4F * max_bounds.
    nodes.append(helper.make_node('Mul', ['c4', 'F_0'], ['max_bounds']))
    nodes.append(helper.make_node('Less', ['R_30x30', 'max_bounds'], ['in_r']))
    nodes.append(helper.make_node('Less', ['C_30x30', 'max_bounds'], ['in_c']))
    nodes.append(helper.make_node('And', ['in_r', 'in_c'], ['in_bounds_4F']))
    
    nodes.append(helper.make_node('And', ['is_line', 'in_bounds_4F'], ['to_draw']))
    nodes.append(helper.make_node('Equal', ['scaled_padded_masked', 'c0'], ['scaled_is_0']))
    nodes.append(helper.make_node('And', ['to_draw', 'scaled_is_0'], ['draw_mask']))
    
    # We must also AND it with in_bounds_dim (the 5F x 5F limit)!
    nodes.append(helper.make_node('And', ['draw_mask', 'in_bounds_dim'], ['final_draw_mask']))
    
    nodes.append(helper.make_node('Where', ['final_draw_mask', 'c2', 'scaled_padded_masked'], ['output_int']))
    
    # OneHot and pad output
    K('depth10', [10])
    K('oh_vals', [0.0, 1.0], dtype=F_type)
    nodes.append(helper.make_node('OneHot', ['output_int', 'depth10', 'oh_vals'], ['oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['oh'], ['raw_output'], perm=[0, 3, 1, 2]))
    
    nodes.append(helper.make_node('Cast', ['in_bounds_dim'], ['presence'], to=F_type))
    # in_bounds_dim is [1, 30, 30]. We need [1, 1, 30, 30] to broadcast correctly over [1, 10, 30, 30].
    nodes.append(helper.make_node('Unsqueeze', ['presence', 'axes_1'], ['presence_4d']))
    nodes.append(helper.make_node('Mul', ['raw_output', 'presence_4d'], ['output']))
    
    graph = helper.make_graph(nodes, 'task107_graph', [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name='task107_model', opset_imports=[helper.make_opsetid('', 13)])
    onnx.save(model, 'task107.onnx')


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task107.onnx")
