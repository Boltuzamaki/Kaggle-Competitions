# Source: predicted/test_onnx_task074.py — ONNX graph construction code
# Verified model: repairs/task074.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, ['batch', 10, 30, 30])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    indices_h_val = [29, 29, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    
    inits = [
        K('c9_i64', [9]),
        K('c_depth_10', [10]), K('c_oh_vals', [0.0, 1.0], dtype=np.float32),
        K('indices_h', indices_h_val),
    ]
    
    nodes = []
    
    nodes.append(helper.make_node('ArgMax', ['input'], ['pred_0'], axis=1, keepdims=1))
    
    curr_pred = 'pred_0'
    
    for i in range(4):
        # Transpose
        trans = f'trans_{i}'
        is_9_t = f'is_9_t_{i}'
        pred_t = f'pred_t_{i}'
        nodes.append(helper.make_node('Transpose', [curr_pred], [trans], perm=[0, 1, 3, 2]))
        nodes.append(helper.make_node('Equal', [curr_pred, 'c9_i64'], [is_9_t]))
        nodes.append(helper.make_node('Where', [is_9_t, trans, curr_pred], [pred_t]))
        
        # Horizontal
        horiz = f'horiz_{i}'
        is_9_h = f'is_9_h_{i}'
        pred_h = f'pred_h_{i}'
        nodes.append(helper.make_node('Gather', [pred_t, 'indices_h'], [horiz], axis=3))
        nodes.append(helper.make_node('Equal', [pred_t, 'c9_i64'], [is_9_h]))
        nodes.append(helper.make_node('Where', [is_9_h, horiz, pred_t], [pred_h]))
        
        # Vertical
        vert = f'vert_{i}'
        is_9_v = f'is_9_v_{i}'
        pred_v = f'pred_v_{i}'
        nodes.append(helper.make_node('Gather', [pred_h, 'indices_h'], [vert], axis=2))
        nodes.append(helper.make_node('Equal', [pred_h, 'c9_i64'], [is_9_v]))
        nodes.append(helper.make_node('Where', [is_9_v, vert, pred_h], [pred_v]))
        
        curr_pred = pred_v
        
    nodes.append(helper.make_node('OneHot', [curr_pred, 'c_depth_10', 'c_oh_vals'], ['pred_oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['pred_oh'], ['pred_trans'], perm=[0, 4, 1, 2, 3]))
    nodes.append(helper.make_node('Squeeze', ['pred_trans'], ['output'], axes=[2]))
    
    graph = helper.make_graph(nodes, 'task074', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 11)])

def check_task():
    with open('task074.json', 'r') as f:
        task = json.load(f)
        
    model = create_model()
    onnx.save(model, 'task074.onnx')
    session = onnxruntime.InferenceSession('task074.onnx')
    
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array(ex['input'], dtype=np.int64)
            out = np.array(ex['output'], dtype=np.int64)
            
            padded_inp = np.zeros((30, 30), dtype=np.int64)
            padded_inp[:inp.shape[0], :inp.shape[1]] = inp
            
            oh_inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for r in range(30):
                for c in range(30):
                    oh_inp[0, padded_inp[r, c], r, c] = 1.0
                    
            res = session.run(['output'], {'input': oh_inp})[0]
            pred = np.argmax(res[0, :, :inp.shape[0], :inp.shape[1]], axis=0)
            
            if np.array_equal(pred, out):
                print(f"{split} {i}: ONNX MATCH")
            else:
                print(f"{split} {i}: ONNX FAIL")
                print("Expected:\n", out)
                print("Pred:\n", pred)


# Build the model
model = create_model()
