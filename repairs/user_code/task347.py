# Source: predicted/test_onnx_task347.py — ONNX graph construction code
# Verified model: repairs/task347.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 347: Bitwise OR of left and right halves, cast to 6.
    """
    I64 = TensorProto.INT64
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H_out', 'W_out'])
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
        
    make_const('c0', [0])
    make_const('c1', [1])
    make_const('c2', [2])
    make_const('c3', [3])
    make_const('axes_0', [0])
    make_const('axes_2', [2])
    make_const('c0_s', 0)
    make_const('c6_s', 6)
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    # W = shape[2]
    nodes.append(helper.make_node('Slice', ['shape', 'c2', 'c3', 'axes_0'], ['W']))
    nodes.append(helper.make_node('Div', ['W', 'c2'], ['mid']))
    
    # left = input[:, :, :mid]
    nodes.append(helper.make_node('Slice', ['input', 'c0', 'mid', 'axes_2'], ['left']))
    # right = input[:, :, mid:]
    nodes.append(helper.make_node('Slice', ['input', 'mid', 'W', 'axes_2'], ['right']))
    
    nodes.append(helper.make_node('Greater', ['left', 'c0_s'], ['left_bool']))
    nodes.append(helper.make_node('Greater', ['right', 'c0_s'], ['right_bool']))
    
    nodes.append(helper.make_node('Or', ['left_bool', 'right_bool'], ['out_bool']))
    nodes.append(helper.make_node('Cast', ['out_bool'], ['out_i64'], to=I64))
    
    nodes.append(helper.make_node('Mul', ['out_i64', 'c6_s'], ['output']))
    
    graph = helper.make_graph(nodes, 'task347_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task347_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task347.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task347.onnx')
    
    with open('task347.json', 'r') as f:
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
                    print(str(split)+' '+str(i)+': ONNX MATCH')
                else:
                    print(str(split)+' '+str(i)+': ONNX FAIL')
                    print("RES:")
                    for r in res[0]: print(''.join(str(x) for x in r))
                    print("OUT:")
                    for r in out[0]: print(''.join(str(x) for x in r))


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task347.onnx")
