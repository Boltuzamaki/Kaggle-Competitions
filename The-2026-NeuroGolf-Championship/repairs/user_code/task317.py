# Source: predicted/test_onnx_task317.py — ONNX graph construction code
# Verified model: repairs/task317.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    
    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
    
    inits = [
        K('c0', [0]),
        K('c1', [1]),
        K('c5', [5]),
        K('shape_5d', [1, 10, 3, 10, 3]),
        K('shape_3d', [1, 30, 30]),
        K('depth10', [10]),
        K('oh_vals', [0.0, 1.0], dtype=np.float32),
    ]
    
    nodes = [
        # Presence mask
        helper.make_node('ReduceMax', ['input'], ['presence'], axes=[1], keepdims=1),
        
        # ArgMax to get int grid [1, 30, 30]
        helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=0),
        
        # Check where value == 5
        helper.make_node('Equal', ['argmax', 'c5'], ['is_5']),
        helper.make_node('Cast', ['is_5'], ['is_5_int'], to=I64),
        
        # Reshape to [1, 10, 3, 10, 3] (blocks of 3x3 in a 30x30 grid)
        helper.make_node('Reshape', ['is_5_int', 'shape_5d'], ['is_5_5d']),
        
        # ReduceMax over axes 2, 4 (within each 3x3 block) -> [1, 10, 1, 10, 1]
        helper.make_node('ReduceMax', ['is_5_5d'], ['max_5d'], axes=[2, 4], keepdims=1),
        
        # Expand back to [1, 10, 3, 10, 3]
        helper.make_node('Expand', ['max_5d', 'shape_5d'], ['expanded']),
        
        # Reshape back to [1, 30, 30]
        helper.make_node('Reshape', ['expanded', 'shape_3d'], ['output_mask']),
        
        # Where mask==1, output 1, else 0
        helper.make_node('Equal', ['output_mask', 'c1'], ['is_1']),
        helper.make_node('Where', ['is_1', 'c1', 'c0'], ['result']),
        
        # OneHot to [1, 30, 30, 10] -> transpose to [1, 10, 30, 30]
        helper.make_node('OneHot', ['result', 'depth10', 'oh_vals'], ['oh'], axis=-1),
        helper.make_node('Transpose', ['oh'], ['raw_output'], perm=[0, 3, 1, 2]),
        
        # Apply presence mask
        helper.make_node('Mul', ['raw_output', 'presence'], ['output']),
    ]
    
    graph = helper.make_graph(nodes, 'task317', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])


# Build the model
model = create_model()
