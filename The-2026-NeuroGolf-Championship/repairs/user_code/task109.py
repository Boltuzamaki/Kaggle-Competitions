# Source: predicted/test_onnx_task109.py — ONNX graph construction code
# Verified model: repairs/task109.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H2', 'W2'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', [0])
    make_const('c1', [1])
    make_const('c2', [2])
    make_const('cm1', [-1])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    make_const('axes_0', [0])
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1', 'c2', 'axes_0'], ['H_1d']))
    nodes.append(helper.make_node('Div', ['H_1d', 'c2'], ['mid']))
    
    # Extract top_left
    nodes.append(helper.make_node('Slice', ['input', 'c0', 'mid', 'axes_1'], ['top_half_in']))
    nodes.append(helper.make_node('Slice', ['top_half_in', 'c0', 'mid', 'axes_2'], ['top_left']))
    
    # Extract cross color (at mid, mid)
    nodes.append(helper.make_node('Add', ['mid', 'c1'], ['mid_plus_1']))
    nodes.append(helper.make_node('Slice', ['input', 'mid', 'mid_plus_1', 'axes_1'], ['cross_row']))
    nodes.append(helper.make_node('Slice', ['cross_row', 'mid', 'mid_plus_1', 'axes_2'], ['cross_color_111']))
    nodes.append(helper.make_node('Squeeze', ['cross_color_111', 'axes_1'], ['cross_color_11']))
    nodes.append(helper.make_node('Squeeze', ['cross_color_11', 'axes_1'], ['cross_color'])) # shape [batch, 1, 1] - actually it's [batch] but since mid is [1] it might be [batch, 1, 1]
    
    # Process top_left
    nodes.append(helper.make_node('Greater', ['top_left', 'c0'], ['mask_bool']))
    nodes.append(helper.make_node('Cast', ['mask_bool'], ['mask'], to=I64))
    
    nodes.append(helper.make_node('Mul', ['mask', 'cross_color_111'], ['colored']))
    
    # Reverses
    make_const('starts_rev', [-1])
    make_const('ends_rev', [-1000])
    make_const('steps_rev', [-1])
    
    nodes.append(helper.make_node('Slice', ['colored', 'starts_rev', 'ends_rev', 'axes_2', 'steps_rev'], ['top_right']))
    nodes.append(helper.make_node('Slice', ['colored', 'starts_rev', 'ends_rev', 'axes_1', 'steps_rev'], ['bot_left']))
    nodes.append(helper.make_node('Slice', ['top_right', 'starts_rev', 'ends_rev', 'axes_1', 'steps_rev'], ['bot_right']))
    
    # Concat
    nodes.append(helper.make_node('Concat', ['colored', 'top_right'], ['top_half_out'], axis=2))
    nodes.append(helper.make_node('Concat', ['bot_left', 'bot_right'], ['bot_half_out'], axis=2))
    nodes.append(helper.make_node('Concat', ['top_half_out', 'bot_half_out'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task109_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task109_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task109.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task109.onnx')
    
    with open('task109.json', 'r') as f:
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
model = onnx.load("/project/repairs/task109.onnx")
