# Source: predicted/test_onnx_task082.py — ONNX graph construction code
# Verified model: repairs/task082.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def K(name, val, dtype=np.int64):
    return helper.make_tensor_value_info(name, dtype, np.array(val).shape)

def create_onnx_model():
    I64 = TensorProto.INT64
    F = TensorProto.FLOAT
    BOOL = TensorProto.BOOL
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    # Constants
    nodes.append(helper.make_node('Constant', [], ['axes_0'], value=helper.make_tensor('axes_0_v', I64, [1], [0])))
    nodes.append(helper.make_node('Constant', [], ['axes_1'], value=helper.make_tensor('axes_1_v', I64, [1], [1])))
    nodes.append(helper.make_node('Constant', [], ['axes_2'], value=helper.make_tensor('axes_2_v', I64, [1], [2])))
    nodes.append(helper.make_node('Constant', [], ['axes_3'], value=helper.make_tensor('axes_3_v', I64, [1], [3])))
    nodes.append(helper.make_node('Constant', [], ['c0'], value=helper.make_tensor('c0_v', I64, [1], [0])))
    nodes.append(helper.make_node('Constant', [], ['c1'], value=helper.make_tensor('c1_v', I64, [1], [1])))
    
    # Get Shape
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1', 'axes_2', 'axes_0'], ['H']))
    nodes.append(helper.make_node('Slice', ['shape', 'axes_2', 'axes_3', 'axes_0'], ['W']))
    nodes.append(helper.make_node('Sub', ['W', 'c1'], ['Wm1']))
    
    # Extract row 0
    nodes.append(helper.make_node('Slice', ['input', 'c0', 'c1', 'axes_1'], ['row0']))
    
    # Shift Left
    nodes.append(helper.make_node('Slice', ['row0', 'c1', 'W', 'axes_2'], ['row0_sliced_left']))
    nodes.append(helper.make_node('Constant', [], ['pads_left'], value=helper.make_tensor('pads_left_v', I64, [6], [0, 0, 0, 0, 0, 1])))
    nodes.append(helper.make_node('Pad', ['row0_sliced_left', 'pads_left'], ['shifted_left']))
    
    # Shift Right
    nodes.append(helper.make_node('Slice', ['row0', 'c0', 'Wm1', 'axes_2'], ['row0_sliced_right']))
    nodes.append(helper.make_node('Constant', [], ['pads_right'], value=helper.make_tensor('pads_right_v', I64, [6], [0, 0, 1, 0, 0, 0])))
    nodes.append(helper.make_node('Pad', ['row0_sliced_right', 'pads_right'], ['shifted_right']))
    
    # odd_rows_1d = Max(shifted_left, shifted_right)
    nodes.append(helper.make_node('Max', ['shifted_left', 'shifted_right'], ['odd_rows_1d']))
    
    # Expand to full shape
    nodes.append(helper.make_node('Expand', ['row0', 'shape'], ['even_rows']))
    nodes.append(helper.make_node('Expand', ['odd_rows_1d', 'shape'], ['odd_rows']))
    
    # Create even_mask
    mask_list = [1, 0] * 25 # length 50
    nodes.append(helper.make_node('Constant', [], ['mask_large'], value=helper.make_tensor('mask_large_v', BOOL, [50], mask_list)))
    nodes.append(helper.make_node('Slice', ['mask_large', 'c0', 'H', 'axes_0'], ['mask_sliced']))
    
    # Reshape mask to [1, H, 1]
    nodes.append(helper.make_node('Concat', ['c1', 'H', 'c1'], ['mask_shape'], axis=0))
    nodes.append(helper.make_node('Reshape', ['mask_sliced', 'mask_shape'], ['even_mask']))
    
    # Where
    nodes.append(helper.make_node('Where', ['even_mask', 'even_rows', 'odd_rows'], ['output']))
    
    graph = helper.make_graph(
        nodes,
        'task082_graph',
        [input_info],
        [output_info]
    )
    
    model = helper.make_model(graph, producer_name='task082_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task082.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task082.onnx')
    
    with open('task082.json', 'r') as f:
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
model = onnx.load("/project/repairs/task082.onnx")
