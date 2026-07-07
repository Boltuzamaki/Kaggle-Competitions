# Source: predicted/test_onnx_task095.py — ONNX graph construction code
# Verified model: repairs/task095.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    F32 = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c1', [1])
    make_const('c5', [5])
    make_const('cm1_1d', [-1])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('axes_0', [0])
    make_const('c0_5', [0.5], dtype=F32)
    
    kernel = np.ones((1, 1, 3, 3), dtype=np.float32)
    make_const('kernel', kernel, dtype=F32)
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_1d']))
    nodes.append(helper.make_node('Slice', ['shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_1d']))
    nodes.append(helper.make_node('Concat', ['cm1_1d', 'c1_1d', 'H_1d', 'W_1d'], ['shape_4d'], axis=0))
    
    nodes.append(helper.make_node('Equal', ['input', 'c5'], ['mask5']))
    nodes.append(helper.make_node('Cast', ['mask5'], ['mask5_f32'], to=F32))
    
    nodes.append(helper.make_node('Reshape', ['mask5_f32', 'shape_4d'], ['mask5_4d']))
    
    nodes.append(helper.make_node('Conv', ['mask5_4d', 'kernel'], ['conv_4d'], pads=[1, 1, 1, 1]))
    
    nodes.append(helper.make_node('Greater', ['conv_4d', 'c0_5'], ['mask_3x3_4d']))
    
    nodes.append(helper.make_node('Reshape', ['mask_3x3_4d', 'shape'], ['mask_3x3']))
    
    nodes.append(helper.make_node('Where', ['mask_3x3', 'c1', 'input'], ['out1']))
    nodes.append(helper.make_node('Where', ['mask5', 'c5', 'out1'], ['output']))
    
    graph = helper.make_graph(nodes, 'task095_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task095_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task095.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task095.onnx')
    
    with open('task095.json', 'r') as f:
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
model = onnx.load("/project/repairs/task095.onnx")
