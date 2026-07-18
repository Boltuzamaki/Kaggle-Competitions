# Source: predicted/test_onnx_task389.py — ONNX graph construction code
# Verified model: repairs/task389.onnx
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
    make_const('c5', 5)
    
    nodes.append(helper.make_node('Equal', ['input', 'c5'], ['is_5']))
    nodes.append(helper.make_node('Where', ['is_5', 'c0', 'input'], ['masked_input']))
    
    nodes.append(helper.make_node('ReduceMax', ['masked_input'], ['c_val'], axes=[0, 1, 2], keepdims=1))
    
    nodes.append(helper.make_node('Where', ['is_5', 'c_val', 'c0'], ['output']))
    
    graph = helper.make_graph(nodes, 'task389_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task389_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task389.onnx')

def check_task():
    import onnxruntime as ort
    create_onnx_model()
    session = ort.InferenceSession('task389.onnx')
    with open('task389.json', 'r') as f:
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
model = onnx.load("/project/repairs/task389.onnx")
