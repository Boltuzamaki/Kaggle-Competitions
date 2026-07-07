# Source: predicted/test_onnx_task115.py — ONNX graph construction code
# Verified model: repairs/task115.onnx
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
    K('c29', [29])
    K('c30', [30])
    K('axes_0', [0])
    K('axes_1', [1])
    K('c0_s', 0)
    
    # 1. ArgMax
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=0)) # [1, 30, 30]
    nodes.append(helper.make_node('Squeeze', ['argmax', 'axes_0'], ['in_30x30'])) # [30, 30]
    
    # 2. W and H masks
    nodes.append(helper.make_node('Slice', ['in_30x30', 'c0', 'c1', 'axes_0'], ['row0_2d'])) # [1, 30]
    nodes.append(helper.make_node('Squeeze', ['row0_2d', 'axes_0'], ['row0'])) # [30]
    
    nodes.append(helper.make_node('Slice', ['in_30x30', 'c0', 'c1', 'axes_1'], ['col0_2d'])) # [30, 1]
    nodes.append(helper.make_node('Squeeze', ['col0_2d', 'axes_1'], ['col0'])) # [30]
    
    # 3. size_X and size_Y
    nodes.append(helper.make_node('Slice', ['row0', 'c0', 'c29', 'axes_0'], ['row0_slice']))
    nodes.append(helper.make_node('Concat', ['c0', 'row0_slice'], ['shifted_row0'], axis=0))
    nodes.append(helper.make_node('Equal', ['row0', 'shifted_row0'], ['eq_row0']))
    nodes.append(helper.make_node('Not', ['eq_row0'], ['neq_row0']))
    nodes.append(helper.make_node('Greater', ['row0', 'c0_s'], ['gt0_row0']))
    nodes.append(helper.make_node('And', ['neq_row0', 'gt0_row0'], ['diff_row0']))
    nodes.append(helper.make_node('Cast', ['diff_row0'], ['diff_row0_i64'], to=I64))
    nodes.append(helper.make_node('ReduceSum', ['diff_row0_i64'], ['size_X'], keepdims=1)) # [1]
    
    nodes.append(helper.make_node('Slice', ['col0', 'c0', 'c29', 'axes_0'], ['col0_slice']))
    nodes.append(helper.make_node('Concat', ['c0', 'col0_slice'], ['shifted_col0'], axis=0))
    nodes.append(helper.make_node('Equal', ['col0', 'shifted_col0'], ['eq_col0']))
    nodes.append(helper.make_node('Not', ['eq_col0'], ['neq_col0']))
    nodes.append(helper.make_node('Greater', ['col0', 'c0_s'], ['gt0_col0']))
    nodes.append(helper.make_node('And', ['neq_col0', 'gt0_col0'], ['diff_col0']))
    nodes.append(helper.make_node('Cast', ['diff_col0'], ['diff_col0_i64'], to=I64))
    nodes.append(helper.make_node('ReduceSum', ['diff_col0_i64'], ['size_Y'], keepdims=1)) # [1]
    
    nodes.append(helper.make_node('Greater', ['size_X', 'size_Y'], ['is_horizontal_1d'])) # [1]
    nodes.append(helper.make_node('Squeeze', ['is_horizontal_1d', 'axes_0'], ['is_horizontal'])) # scalar bool
    
    # 3. L = Where(is_horizontal, row0, col0)
    nodes.append(helper.make_node('Where', ['is_horizontal', 'row0', 'col0'], ['L'])) # [30]
    
    # 4. diff = (L != shifted_L) & (L != 0)
    nodes.append(helper.make_node('Slice', ['L', 'c0', 'c29', 'axes_0'], ['L_slice'])) # [29]
    nodes.append(helper.make_node('Concat', ['c0', 'L_slice'], ['shifted_L'], axis=0)) # [30]
    
    nodes.append(helper.make_node('Equal', ['L', 'shifted_L'], ['eq_L']))
    nodes.append(helper.make_node('Not', ['eq_L'], ['neq_L']))
    nodes.append(helper.make_node('Greater', ['L', 'c0_s'], ['gt0_L']))
    nodes.append(helper.make_node('And', ['neq_L', 'gt0_L'], ['diff'])) # [30] bool
    nodes.append(helper.make_node('Cast', ['diff'], ['diff_f'], to=F_type)) # [30] float
    
    # 5. CumSum using MatMul
    L_tri = np.triu(np.ones((30, 30), dtype=np.float32))
    K('L_tri', L_tri, dtype=F_type)
    nodes.append(helper.make_node('Unsqueeze', ['diff_f', 'axes_0'], ['diff_f_2d'])) # [1, 30]
    nodes.append(helper.make_node('MatMul', ['diff_f_2d', 'L_tri'], ['cumsum_f_2d'])) # [1, 30]
    nodes.append(helper.make_node('Squeeze', ['cumsum_f_2d', 'axes_0'], ['cumsum_f'])) # [30]
    nodes.append(helper.make_node('Cast', ['cumsum_f'], ['cumsum'], to=I64)) # [30]
    
    # 6. Extract c_k for k=1..9
    c_k_nodes = []
    for k in range(1, 10):
        K(f'k_{k}', [k])
        # is_k = (cumsum == k) & diff
        nodes.append(helper.make_node('Equal', ['cumsum', f'k_{k}'], [f'eq_k_{k}']))
        nodes.append(helper.make_node('And', [f'eq_k_{k}', 'diff'], [f'is_k_{k}']))
        nodes.append(helper.make_node('Cast', [f'is_k_{k}'], [f'is_k_i64_{k}'], to=I64))
        
        # c_k = Sum(L * is_k)
        nodes.append(helper.make_node('Mul', ['L', f'is_k_i64_{k}'], [f'L_masked_{k}']))
        nodes.append(helper.make_node('ReduceSum', [f'L_masked_{k}'], [f'c_k_{k}'], keepdims=1)) # [1]
        c_k_nodes.append(f'c_k_{k}')
        
    # Pad with zeros to size 30
    K('zeros_21', np.zeros(21, dtype=np.int64))
    nodes.append(helper.make_node('Concat', c_k_nodes + ['zeros_21'], ['out_array'], axis=0)) # [30]
    
    # 7. Place in grid
    nodes.append(helper.make_node('Unsqueeze', ['out_array', 'axes_0'], ['out_row_1x30'])) # [1, 30]
    K('pads_row', [0, 0, 29, 0]) # Pad 29 rows at bottom (axes are H, W so [pad_top, pad_left, pad_bottom, pad_right])
    nodes.append(helper.make_node('Pad', ['out_row_1x30', 'pads_row'], ['out_row'], mode='constant')) # [30, 30]
    
    nodes.append(helper.make_node('Unsqueeze', ['out_array', 'axes_1'], ['out_col_30x1'])) # [30, 1]
    K('pads_col', [0, 0, 0, 29]) # Pad 29 cols at right
    nodes.append(helper.make_node('Pad', ['out_col_30x1', 'pads_col'], ['out_col'], mode='constant')) # [30, 30]
    
    nodes.append(helper.make_node('Where', ['is_horizontal', 'out_row', 'out_col'], ['out_grid'])) # [30, 30]
    nodes.append(helper.make_node('Unsqueeze', ['out_grid', 'axes_0'], ['out_grid_1x30x30'])) # [1, 30, 30]
    
    # 8. OneHot
    K('depth10', [10])
    K('oh_vals', [0.0, 1.0], dtype=F_type)
    nodes.append(helper.make_node('OneHot', ['out_grid_1x30x30', 'depth10', 'oh_vals'], ['oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['oh'], ['raw_output'], perm=[0, 3, 1, 2]))
    
    nodes.append(helper.make_node('Greater', ['out_grid_1x30x30', 'c0_s'], ['presence_bool']))
    nodes.append(helper.make_node('Cast', ['presence_bool'], ['presence_f'], to=F_type))
    nodes.append(helper.make_node('Unsqueeze', ['presence_f', 'axes_1'], ['presence_f_4d']))
    nodes.append(helper.make_node('Mul', ['raw_output', 'presence_f_4d'], ['output']))
    
    graph = helper.make_graph(nodes, 'task115_graph', [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name='task115_model', opset_imports=[helper.make_opsetid('', 13)])
    onnx.save(model, 'task115.onnx')


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task115.onnx")
