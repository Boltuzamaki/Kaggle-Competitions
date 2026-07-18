# Source: predicted/test_onnx_task085.py — ONNX graph construction code
# Verified model: repairs/task085.onnx
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
    nodes.append(helper.make_node('Constant', [], ['axes_1'], value=helper.make_tensor('axes_1_v', I64, [1], [1])))
    nodes.append(helper.make_node('Constant', [], ['c0'], value=helper.make_tensor('c0_v', I64, [], [0])))
    nodes.append(helper.make_node('Constant', [], ['c1_1d'], value=helper.make_tensor('c1_1d_v', I64, [1], [1])))
    nodes.append(helper.make_node('Constant', [], ['c2_1d'], value=helper.make_tensor('c2_1d_v', I64, [1], [2])))
    nodes.append(helper.make_node('Constant', [], ['c2'], value=helper.make_tensor('c2_v', I64, [], [2])))
    
    # CumSum logic
    nodes.append(helper.make_node('Greater', ['input', 'c0'], ['is_pos']))
    nodes.append(helper.make_node('Cast', ['is_pos'], ['is_pos_int'], to=I64))
    nodes.append(helper.make_node('CumSum', ['is_pos_int', 'c2_1d'], ['cumsum']))
    nodes.append(helper.make_node('Mod', ['cumsum', 'c2'], ['mod2']))
    nodes.append(helper.make_node('Equal', ['mod2', 'c0'], ['is_even']))
    
    # Pad logic
    nodes.append(helper.make_node('Constant', [], ['pads'], value=helper.make_tensor('pads_v', I64, [6], [0, 1, 0, 0, 1, 0])))
    nodes.append(helper.make_node('Pad', ['input', 'pads'], ['padded']))
    
    # Get H
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H']))
    nodes.append(helper.make_node('Add', ['H', 'c2_1d'], ['Hp2']))
    nodes.append(helper.make_node('Constant', [], ['c0_1d'], value=helper.make_tensor('c0_1d_v', I64, [1], [0])))
    
    # up and down
    nodes.append(helper.make_node('Slice', ['padded', 'c2_1d', 'Hp2', 'axes_1'], ['up']))
    nodes.append(helper.make_node('Slice', ['padded', 'c0_1d', 'H', 'axes_1'], ['down']))
    
    # Middle condition
    nodes.append(helper.make_node('Equal', ['input', 'up'], ['eq_up']))
    nodes.append(helper.make_node('Equal', ['input', 'down'], ['eq_down']))
    nodes.append(helper.make_node('And', ['eq_up', 'eq_down'], ['is_middle']))
    nodes.append(helper.make_node('And', ['is_middle', 'is_pos'], ['is_valid_middle']))
    
    # Final condition
    nodes.append(helper.make_node('And', ['is_valid_middle', 'is_even'], ['to_zero']))
    
    # Output
    nodes.append(helper.make_node('Where', ['to_zero', 'c0', 'input'], ['output']))
    
    graph = helper.make_graph(
        nodes,
        'task085_graph',
        [input_info],
        [output_info]
    )
    
    model = helper.make_model(graph, producer_name='task085_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task085.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task085.onnx')
    
    with open('task085.json', 'r') as f:
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
model = onnx.load("/project/repairs/task085.onnx")
