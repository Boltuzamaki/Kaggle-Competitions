# Source: predicted/test_onnx_task084.py — ONNX graph construction code
# Verified model: repairs/task084.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def K(name, val, dtype=np.int64):
    return helper.make_tensor_value_info(name, dtype, np.array(val).shape)

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    # Constants
    nodes.append(helper.make_node('Constant', [], ['axes_0'], value=helper.make_tensor('axes_0_v', I64, [1], [0])))
    nodes.append(helper.make_node('Constant', [], ['c0'], value=helper.make_tensor('c0_v', I64, [1], [0])))
    nodes.append(helper.make_node('Constant', [], ['c1'], value=helper.make_tensor('c1_v', I64, [1], [1])))
    nodes.append(helper.make_node('Constant', [], ['c2'], value=helper.make_tensor('c2_v', I64, [1], [2])))
    nodes.append(helper.make_node('Constant', [], ['c3'], value=helper.make_tensor('c3_v', I64, [1], [3])))
    nodes.append(helper.make_node('Constant', [], ['c2_val'], value=helper.make_tensor('c2_val_v', I64, [], [2])))
    nodes.append(helper.make_node('Constant', [], ['c4_val'], value=helper.make_tensor('c4_val_v', I64, [], [4])))
    
    # Shape, H, W
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1', 'c2', 'axes_0'], ['H_1d']))
    nodes.append(helper.make_node('Slice', ['shape', 'c2', 'c3', 'axes_0'], ['W_1d']))
    
    nodes.append(helper.make_node('Sub', ['H_1d', 'c1'], ['Hm1_1d']))
    nodes.append(helper.make_node('Sub', ['W_1d', 'c1'], ['Wm1_1d']))
    
    # Generate x and y grids using CumSum
    # ones = (input == input) cast to INT64
    nodes.append(helper.make_node('Equal', ['input', 'input'], ['ones_bool']))
    nodes.append(helper.make_node('Cast', ['ones_bool'], ['ones'], to=I64))
    
    nodes.append(helper.make_node('CumSum', ['ones', 'c2'], ['x_plus_1']))
    nodes.append(helper.make_node('Sub', ['x_plus_1', 'c1'], ['x']))
    
    nodes.append(helper.make_node('CumSum', ['ones', 'c1'], ['y_plus_1']))
    nodes.append(helper.make_node('Sub', ['y_plus_1', 'c1'], ['y']))
    
    nodes.append(helper.make_node('Add', ['x', 'y'], ['x_plus_y']))
    
    # Conditions
    nodes.append(helper.make_node('Equal', ['x_plus_y', 'Wm1_1d'], ['mask_diag']))
    nodes.append(helper.make_node('Equal', ['y', 'Hm1_1d'], ['mask_bottom']))
    nodes.append(helper.make_node('Greater', ['x', 'c0'], ['mask_x_gt_0']))
    
    nodes.append(helper.make_node('And', ['mask_diag', 'mask_x_gt_0'], ['diag_cond']))
    nodes.append(helper.make_node('And', ['mask_bottom', 'mask_x_gt_0'], ['bottom_cond']))
    
    # Where
    nodes.append(helper.make_node('Where', ['bottom_cond', 'c4_val', 'input'], ['out1']))
    nodes.append(helper.make_node('Where', ['diag_cond', 'c2_val', 'out1'], ['output']))
    
    graph = helper.make_graph(
        nodes,
        'task084_graph',
        [input_info],
        [output_info]
    )
    
    model = helper.make_model(graph, producer_name='task084_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task084.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task084.onnx')
    
    with open('task084.json', 'r') as f:
        task = json.load(f)
        
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array(ex['input'], dtype=np.int64)[np.newaxis, ...]
            if 'output' in ex:
                out = np.array(ex['output'], dtype=np.int64)[np.newaxis, ...]
            else:
                out = None
                
            res = session.run(['output'], {'input': inp})[0]
            
            if out is not None:
                if np.array_equal(res, out):
                    print(f"{split} {i}: ONNX MATCH")
                else:
                    print(f"{split} {i}: ONNX FAIL")
                    print("RES:")
                    for r in res[0]: print(''.join(str(x) for x in r))
                    print("OUT:")
                    for r in out[0]: print(''.join(str(x) for x in r))


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task084.onnx")
