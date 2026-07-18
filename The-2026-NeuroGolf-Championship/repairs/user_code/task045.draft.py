# Source: predicted/test_onnx_task045.py — ONNX graph construction code
# Verified model: repairs/task045.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    inits = [
        K('ax_1', [1]), K('ax_3', [3]),
        K('c0', [0]), K('c1', [1]),
        K('c0_i64', [0], dtype=np.int64),
        K('col_indices', np.arange(30).reshape(1, 1, 1, 30), dtype=np.int64),
        K('c_oh_vals', [0.0, 1.0], dtype=np.float32),
        K('c_depth_10', [10], dtype=np.int64),
    ]
    
    nodes = []
    
    # argmax
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=1))
    
    # non_zeros
    nodes.append(helper.make_node('Greater', ['argmax', 'c0_i64'], ['non_zeros_b']))
    nodes.append(helper.make_node('Cast', ['non_zeros_b'], ['non_zeros_i64'], to=I64))
    
    # max_col
    nodes.append(helper.make_node('Mul', ['col_indices', 'non_zeros_i64'], ['col_nz']))
    nodes.append(helper.make_node('ReduceMax', ['col_nz'], ['max_col'], axes=[2, 3], keepdims=1))
    
    # valid_cols
    nodes.append(helper.make_node('LessOrEqual', ['col_indices', 'max_col'], ['valid_cols']))
    
    # c0
    nodes.append(helper.make_node('Slice', ['argmax', 'c0', 'c1', 'ax_3'], ['c0_val']))
    
    # c_right
    nodes.append(helper.make_node('Equal', ['col_indices', 'max_col'], ['is_max_col']))
    nodes.append(helper.make_node('Cast', ['is_max_col'], ['is_max_col_i64'], to=I64))
    nodes.append(helper.make_node('Mul', ['argmax', 'is_max_col_i64'], ['right_vals']))
    nodes.append(helper.make_node('ReduceMax', ['right_vals'], ['c_right'], axes=[3], keepdims=1))
    
    # is_match
    nodes.append(helper.make_node('Equal', ['c0_val', 'c_right'], ['c_match']))
    nodes.append(helper.make_node('Greater', ['c0_val', 'c0_i64'], ['c0_gt_0']))
    nodes.append(helper.make_node('And', ['c_match', 'c0_gt_0'], ['is_match']))
    
    # fill_mask
    nodes.append(helper.make_node('And', ['is_match', 'valid_cols'], ['fill_mask']))
    
    # output_argmax
    nodes.append(helper.make_node('Where', ['fill_mask', 'c0_val', 'argmax'], ['output_argmax']))
    
    # onehot
    nodes.append(helper.make_node('OneHot', ['output_argmax', 'c_depth_10', 'c_oh_vals'], ['pred_oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['pred_oh'], ['pred_trans'], perm=[0, 4, 1, 2, 3]))
    inits.append(K('ax_2', [2]))
    nodes.append(helper.make_node('Squeeze', ['pred_trans', 'ax_2'], ['output']))
    
    graph = helper.make_graph(nodes, 'task045', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

def check_task():
    with open('task045.json', 'r') as f:
        task = json.load(f)
        
    model = create_model()
    onnx.save(model, 'task045.onnx')
    session = onnxruntime.InferenceSession('task045.onnx')
    
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
