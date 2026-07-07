# Source: predicted/test_onnx_task395.py — ONNX graph construction code
# Verified model: repairs/task395.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H_out', 'W_out'])
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
        
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('c2', 2)
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H']))
    nodes.append(helper.make_node('Div', ['H', 'c2_1d'], ['H_half']))
    
    nodes.append(helper.make_node('Slice', ['input', 'c0_1d', 'H_half', 'axes_1'], ['top']))
    nodes.append(helper.make_node('Slice', ['input', 'H_half', 'H', 'axes_1'], ['bottom']))
    
    nodes.append(helper.make_node('Equal', ['top', 'c0'], ['is_0_top']))
    nodes.append(helper.make_node('Equal', ['bottom', 'c0'], ['is_0_bottom']))
    
    nodes.append(helper.make_node('And', ['is_0_top', 'is_0_bottom'], ['both_0']))
    nodes.append(helper.make_node('Where', ['both_0', 'c2', 'c0'], ['output']))
    
    graph = helper.make_graph(nodes, 'task395_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task395_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task395.onnx')

def check_task():
    import onnxruntime as ort
    create_onnx_model()
    session = ort.InferenceSession('task395.onnx')
    with open('task395.json', 'r') as f:
        task = json.load(f)
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array(ex['input'], dtype=np.int64)[np.newaxis, ...]
            out = np.array(ex['output'], dtype=np.int64)[np.newaxis, ...] if 'output' in ex else None
            res = session.run(['output'], {'input': inp})[0]
            if out is not None:
                if np.array_equal(res, out):
                    print(f'{split} {i}: ONNX MATCH')
                else:
                    print(f'{split} {i}: ONNX FAIL')
                    print("RES:\n", res[0])
                    print("OUT:\n", out[0])


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task395.onnx")
