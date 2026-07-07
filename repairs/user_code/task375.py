# Source: predicted/test_onnx_task375.py — ONNX graph construction code
# Verified model: repairs/task375.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 375: Draw an X with color 0, crossing at the 0 in the input.
    """
    F32 = TensorProto.FLOAT
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', F32, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info('output', F32, [1, 10, 30, 30])
    nodes = []
    inits = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val, dtype=np.float32 if dtype == F32 else np.int64)
        inits.append(helper.make_tensor(name, dtype, val_arr.shape, val_arr.flatten().tolist()))
        
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    make_const('axes_3', [3])
    make_const('axes_2_3', [2, 3])
    make_const('c0', [0])
    make_const('c1', [1])
    make_const('f1', 1.0, F32)
    
    # 1. Presence mask
    nodes.append(helper.make_node('ReduceSum', ['input', 'axes_1'], ['presence'], keepdims=1))
    
    # 2. Extract is_0
    nodes.append(helper.make_node('Slice', ['input', 'c0', 'c1', 'axes_1'], ['is_0']))
    
    # 3. Grids
    make_const('range_30', np.arange(30).astype(np.float32), F32)
    make_const('shape_r', [1, 1, 30, 1])
    make_const('shape_c', [1, 1, 1, 30])
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_r'], ['r_grid']))
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_c'], ['c_grid']))
    
    # 4. center_r, center_c
    nodes.append(helper.make_node('Mul', ['r_grid', 'is_0'], ['r_masked']))
    nodes.append(helper.make_node('Mul', ['c_grid', 'is_0'], ['c_masked']))
    nodes.append(helper.make_node('ReduceSum', ['r_masked', 'axes_2_3'], ['center_r'], keepdims=1))
    nodes.append(helper.make_node('ReduceSum', ['c_masked', 'axes_2_3'], ['center_c'], keepdims=1))
    
    # 5. abs_H and abs_W
    nodes.append(helper.make_node('Sub', ['r_grid', 'center_r'], ['diff_r']))
    nodes.append(helper.make_node('Sub', ['c_grid', 'center_c'], ['diff_c']))
    nodes.append(helper.make_node('Abs', ['diff_r'], ['abs_r']))
    nodes.append(helper.make_node('Abs', ['diff_c'], ['abs_c']))
    
    # 6. is_X
    nodes.append(helper.make_node('Equal', ['abs_r', 'abs_c'], ['is_X_bool']))
    nodes.append(helper.make_node('Cast', ['is_X_bool'], ['is_X_f32'], to=int(F32)))
    nodes.append(helper.make_node('Sub', ['f1', 'is_X_f32'], ['not_X_f32']))
    
    # 7. bg_color_oh
    nodes.append(helper.make_node('Slice', ['input', 'c0', 'c1', 'axes_2'], ['slice_0']))
    nodes.append(helper.make_node('Slice', ['slice_0', 'c0', 'c1', 'axes_3'], ['bg_color_oh']))
    
    # 8. Output computation
    make_const('c0_oh', np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape(1, 10, 1, 1), F32)
    nodes.append(helper.make_node('Mul', ['bg_color_oh', 'not_X_f32'], ['bg_part']))
    nodes.append(helper.make_node('Mul', ['c0_oh', 'is_X_f32'], ['x_part']))
    nodes.append(helper.make_node('Add', ['bg_part', 'x_part'], ['output_raw']))
    nodes.append(helper.make_node('Mul', ['output_raw', 'presence'], ['output']))
    
    graph = helper.make_graph(nodes, 'task375_graph', [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name='task375_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task375.onnx')

def check_task():
    import onnxruntime as ort
    create_onnx_model()
    print('ONNX model generated!')


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task375.onnx")
