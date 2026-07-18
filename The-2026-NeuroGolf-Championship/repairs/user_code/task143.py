# Source: predicted/test_onnx_task143.py — ONNX graph construction code
# Verified model: repairs/task143.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 143: Template matching using shifts.
    """
    I64 = TensorProto.INT64
    F32 = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', F32, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info('output', F32, [1, 10, 30, 30])
    
    nodes = []
    inits = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        inits.append(helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist()))
        nodes.append(helper.make_node('Constant', [], [name], value=inits[-1]))
    
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('c3', 3)
    make_const('c5', 5)
    make_const('f0', 0.0, F32)
    make_const('f1', 1.0, F32)
    make_const('f0_5', 0.5, F32)
    make_const('axes_1_2', [1, 2])
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    
    # 1. ArgMax
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax_raw'], axis=1, keepdims=0)) # [1, 30, 30]
    
    # 2. Presence
    nodes.append(helper.make_node('ReduceMax', ['input'], ['presence'], axes=[1], keepdims=1)) # [1, 1, 30, 30]
    
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('c30', 30)
    
    nodes.append(helper.make_node('Range', ['c0', 'c30', 'c1'], ['range_H']))
    nodes.append(helper.make_node('Range', ['c0', 'c30', 'c1'], ['range_W']))
    make_const('shape_1H1', [1, -1, 1])
    make_const('shape_11W', [1, 1, -1])
    nodes.append(helper.make_node('Reshape', ['range_H', 'shape_1H1'], ['grid_r']))
    nodes.append(helper.make_node('Reshape', ['range_W', 'shape_11W'], ['grid_c']))
    
    # M = (argmax_raw > 0) & (argmax_raw != 5)
    nodes.append(helper.make_node('Greater', ['argmax_raw', 'c0'], ['gt_0']))
    make_const('c5_bcast', 5) # scalar
    nodes.append(helper.make_node('Equal', ['argmax_raw', 'c5_bcast'], ['is_5']))
    nodes.append(helper.make_node('Not', ['is_5'], ['not_5']))
    nodes.append(helper.make_node('And', ['gt_0', 'not_5'], ['M_bool']))
    nodes.append(helper.make_node('Cast', ['M_bool'], ['M'], to=int(F32)))
    
    # T = M[:, :3, :3]
    make_const('starts_00', [0, 0])
    make_const('ends_33', [3, 3])
    nodes.append(helper.make_node('Slice', ['M', 'starts_00', 'ends_33', 'axes_1_2'], ['T']))
    
    nodes.append(helper.make_node('ReduceSum', ['T', 'axes_1_2'], ['T_sum'], keepdims=1))
    
    make_const('shape_1x1x1x1', [1, 1, 1, 1])
    nodes.append(helper.make_node('Reshape', ['T_sum', 'shape_1x1x1x1'], ['T_sum_4d']))

    # Iterate over 3x3 to extract T_val
    for i in range(3):
        for j in range(3):
            suffix = f'_{i}_{j}'
            make_const(f'starts_T{suffix}', [i, j])
            make_const(f'ends_T{suffix}', [i+1, j+1])
            nodes.append(helper.make_node('Slice', ['T', f'starts_T{suffix}', f'ends_T{suffix}', 'axes_1_2'], [f't_val_3d{suffix}']))
            nodes.append(helper.make_node('Reshape', [f't_val_3d{suffix}', 'shape_1x1x1x1'], [f't_val{suffix}']))
            nodes.append(helper.make_node('Sub', ['f1', f't_val{suffix}'], [f'not_T{suffix}']))

    make_const('axes_2_3', [2, 3])
    
    conv_T_nodes = []
    conv_not_T_nodes = []
    for i in range(3):
        for j in range(3):
            suffix = f'_{i}_{j}'
            make_const(f'starts_in{suffix}', [i, j])
            make_const(f'ends_in{suffix}', [i+28, j+28])
            nodes.append(helper.make_node('Slice', ['input', f'starts_in{suffix}', f'ends_in{suffix}', 'axes_2_3'], [f'in_slice{suffix}']))
            nodes.append(helper.make_node('Mul', [f'in_slice{suffix}', f't_val{suffix}'], [f'term_T{suffix}']))
            conv_T_nodes.append(f'term_T{suffix}')
            nodes.append(helper.make_node('Mul', [f'in_slice{suffix}', f'not_T{suffix}'], [f'term_not_T{suffix}']))
            conv_not_T_nodes.append(f'term_not_T{suffix}')
            
    nodes.append(helper.make_node('Sum', conv_T_nodes, ['Conv_T']))
    nodes.append(helper.make_node('Sum', conv_not_T_nodes, ['Conv_not_T']))

    # 5x5 sum on input
    make_const('pads_4d', [0, 0, 2, 2, 0, 0, 2, 2])
    nodes.append(helper.make_node('Pad', ['input', 'pads_4d'], ['input_padded']))
    m_5x5_nodes = []
    for di in range(5):
        for dj in range(5):
            suffix = f'_{di}_{dj}'
            make_const(f'starts_in5{suffix}', [di+1, dj+1])
            make_const(f'ends_in5{suffix}', [di+29, dj+29])
            nodes.append(helper.make_node('Slice', ['input_padded', f'starts_in5{suffix}', f'ends_in5{suffix}', 'axes_2_3'], [f'in5_slice{suffix}']))
            m_5x5_nodes.append(f'in5_slice{suffix}')
            
    nodes.append(helper.make_node('Sum', m_5x5_nodes, ['M_5x5_sum']))

    # Match_c
    nodes.append(helper.make_node('Equal', ['Conv_T', 'T_sum_4d'], ['match_T']))
    nodes.append(helper.make_node('Equal', ['Conv_not_T', 'f0'], ['match_not_T']))
    nodes.append(helper.make_node('Equal', ['M_5x5_sum', 'T_sum_4d'], ['match_iso']))
    nodes.append(helper.make_node('And', ['match_T', 'match_not_T'], ['Match_bool_1']))
    nodes.append(helper.make_node('And', ['Match_bool_1', 'match_iso'], ['Match_bool_c']))
    
    # Or across channels
    nodes.append(helper.make_node('Cast', ['Match_bool_c'], ['Match_int_c'], to=int(F32)))

    nodes.append(helper.make_node('ReduceSum', ['Match_int_c', 'axes_1'], ['Match_any_sum'], keepdims=1))
    nodes.append(helper.make_node('Greater', ['Match_any_sum', 'f0'], ['Match_bool']))
    nodes.append(helper.make_node('Cast', ['Match_bool'], ['Match'], to=int(F32)))
    nodes.append(helper.make_node('Squeeze', ['Match', 'axes_1'], ['Match_3d']))

    # Reconstruct To_Change
    to_change_nodes = []
    for i in range(3):
        for j in range(3):
            suffix = f'_{i}_{j}'
            nodes.append(helper.make_node('Mul', ['Match_3d', f't_val_3d{suffix}'], [f'Match_T{suffix}']))
            pad_top = i
            pad_bottom = 2 - i
            pad_left = j
            pad_right = 2 - j
            make_const(f'pads{suffix}', [0, pad_top, pad_left, 0, pad_bottom, pad_right])
            nodes.append(helper.make_node('Pad', [f'Match_T{suffix}', f'pads{suffix}'], [f'Padded{suffix}']))
            to_change_nodes.append(f'Padded{suffix}')
            
    nodes.append(helper.make_node('Sum', to_change_nodes, ['To_Change']))
    
    # not_top_left = (grid_r >= 3) | (grid_c >= 3)
    make_const('c3_bcast', 3)
    nodes.append(helper.make_node('GreaterOrEqual', ['grid_r', 'c3_bcast'], ['r_ge_3']))
    nodes.append(helper.make_node('GreaterOrEqual', ['grid_c', 'c3_bcast'], ['c_ge_3']))
    nodes.append(helper.make_node('Or', ['r_ge_3', 'c_ge_3'], ['not_top_left']))
    
    nodes.append(helper.make_node('Greater', ['To_Change', 'f0_5'], ['change_mask']))
    nodes.append(helper.make_node('And', ['change_mask', 'not_top_left'], ['to_fill']))
    
    nodes.append(helper.make_node('Where', ['to_fill', 'c5_bcast', 'argmax_raw'], ['out_grid']))
    
    make_const('depth10', [10])
    make_const('oh_vals', [0.0, 1.0], dtype=F32)
    nodes.append(helper.make_node('OneHot', ['out_grid', 'depth10', 'oh_vals'], ['oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['oh'], ['raw_output'], perm=[0, 3, 1, 2]))
    
    # Mask padding
    nodes.append(helper.make_node('Mul', ['raw_output', 'presence'], ['output']))
    
    graph = helper.make_graph(nodes, 'task143_graph', [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name='task143_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task143.onnx')


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task143.onnx")
