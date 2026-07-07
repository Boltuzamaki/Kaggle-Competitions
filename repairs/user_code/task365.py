# Source: predicted/test_onnx_task365.py — ONNX graph construction code
# Verified model: repairs/task365.onnx
import onnx
from onnx import helper, TensorProto
import numpy as np

def create_onnx_model():
    F32 = TensorProto.FLOAT
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', F32, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info('output', F32, [1, 10, 30, 30])
    
    nodes = []
    
    def make_const(name, val, dtype=F32):
        val_arr = np.array(val, dtype=np.float32 if dtype == F32 else np.int64)
        node = helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.tolist()))
        nodes.append(node)
        
    make_const('starts_1', [1], I64)
    make_const('ends_10', [10], I64)
    make_const('axes_1_i64', [1], I64)
    nodes.append(helper.make_node('Slice', ['input', 'starts_1', 'ends_10', 'axes_1_i64'], ['input_1_to_9']))
    nodes.append(helper.make_node('ReduceSum', ['input_1_to_9', 'axes_1_i64'], ['mask_fg'], keepdims=1))    
    make_const('one_f32', [1.0])
    make_const('shape_900', [900], I64)
    make_const('shape_1_900', [1, 900], I64)
    make_const('shape_900_1', [900, 1], I64)
    make_const('shape_1_1_30_30', [1, 1, 30, 30], I64)
    
    nodes.append(helper.make_node('Reshape', ['mask_fg', 'shape_1_900'], ['M']))
    nodes.append(helper.make_node('Reshape', ['mask_fg', 'shape_900_1'], ['M_col']))
    nodes.append(helper.make_node('Mul', ['M_col', 'M'], ['valid_pairs']))
    
    B = np.zeros((900, 900), dtype=np.float32)
    for r in range(30):
        for c in range(30):
            i = r * 30 + c
            B[i, i] = 1.0
            if r > 0: B[i, i - 30] = 1.0
            if r < 29: B[i, i + 30] = 1.0
            if c > 0: B[i, i - 1] = 1.0
            if c < 29: B[i, i + 1] = 1.0
    make_const('B', B)
    
    nodes.append(helper.make_node('Mul', ['B', 'valid_pairs'], ['A0']))
    make_const('zero_f', [0.0])
    
    curr_A = 'A0'
    for i in range(10):
        next_A = f'A{i+1}'
        nodes.append(helper.make_node('MatMul', [curr_A, curr_A], [next_A+'_raw']))
        nodes.append(helper.make_node('Clip', [next_A+'_raw', 'zero_f', 'one_f32'], [next_A]))
        curr_A = next_A
        
    make_const('c2', [2], I64)
    nodes.append(helper.make_node('Gather', ['input', 'c2'], ['mask_2_unreshaped'], axis=1))
    nodes.append(helper.make_node('Reshape', ['mask_2_unreshaped', 'shape_900_1'], ['mask_2_col']))
    
    nodes.append(helper.make_node('MatMul', [curr_A, 'mask_2_col'], ['num_2s']))
    
    nodes.append(helper.make_node('ReduceMax', ['num_2s'], ['max_2s'], axes=[0], keepdims=1))
    nodes.append(helper.make_node('Equal', ['num_2s', 'max_2s'], ['is_best_comp_raw']))
    nodes.append(helper.make_node('Cast', ['is_best_comp_raw'], ['is_best_comp_f'], to=F32))
    nodes.append(helper.make_node('Mul', ['is_best_comp_f', 'M_col'], ['best_mask_col']))
    nodes.append(helper.make_node('Reshape', ['best_mask_col', 'shape_1_1_30_30'], ['best_mask']))
    
    r_indices = np.array([[[[r] * 30 for r in range(30)]]], dtype=np.float32)
    c_indices = np.array([[[[c for c in range(30)] for r in range(30)]]], dtype=np.float32)
    make_const('r_indices', r_indices)
    make_const('c_indices', c_indices)
    
    make_const('inf', [99.0])
    
    nodes.append(helper.make_node('Equal', ['best_mask', 'zero_f'], ['best_mask_0']))
    nodes.append(helper.make_node('Where', ['best_mask_0', 'inf', 'r_indices'], ['best_rows']))
    nodes.append(helper.make_node('Where', ['best_mask_0', 'inf', 'c_indices'], ['best_cols']))
    
    nodes.append(helper.make_node('ReduceMin', ['best_rows'], ['min_r'], axes=[2,3], keepdims=1))
    nodes.append(helper.make_node('ReduceMin', ['best_cols'], ['min_c'], axes=[2,3], keepdims=1))
    
    r_vec = np.arange(30, dtype=np.float32).reshape(30, 1)
    c_vec = np.arange(30, dtype=np.float32).reshape(1, 30)
    r_diff = c_vec - r_vec
    make_const('r_diff', r_diff.reshape(1, 1, 30, 30))
    
    nodes.append(helper.make_node('Equal', ['r_diff', 'min_r'], ['R_eq']))
    nodes.append(helper.make_node('Cast', ['R_eq'], ['R'], to=F32))
    
    nodes.append(helper.make_node('Equal', ['r_diff', 'min_c'], ['C_eq']))
    nodes.append(helper.make_node('Cast', ['C_eq'], ['C_mat'], to=F32))
    nodes.append(helper.make_node('Transpose', ['C_mat'], ['C_mat_T'], perm=[0,1,3,2]))
    
    nodes.append(helper.make_node('Mul', ['input', 'best_mask'], ['input_best']))
    
    nodes.append(helper.make_node('MatMul', ['input_best', 'C_mat_T'], ['shifted_cols']))
    nodes.append(helper.make_node('MatMul', ['R', 'shifted_cols'], ['shifted_all']))
    
    nodes.append(helper.make_node('Identity', ['shifted_all'], ['output']))
    
    graph = helper.make_graph(nodes, 'task365_graph', [input_info], [output_info])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task365.onnx')

create_onnx_model()


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task365.onnx")
