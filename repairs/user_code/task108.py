# Source: predicted/test_onnx_task108.py — ONNX graph construction code
# Verified model: repairs/task108.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 10, 10])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 20, 20])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('starts', [1, 1])
    make_const('ends', [10, 10])
    make_const('axes', [1, 2])
    make_const('steps', [2, 2])
    
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes', 'steps'], ['sliced']))
    
    # Get batch size dynamically
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('axes_0', [0])
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    
    make_const('c5_1d', [5])
    make_const('c1_1d_val', [1])
    make_const('c4_1d', [4])
    make_const('c20_1d', [20])
    
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c5_1d', 'c1_1d_val', 'c5_1d', 'c1_1d_val'], ['reshape1_shape'], axis=0))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c5_1d', 'c4_1d', 'c5_1d', 'c4_1d'], ['expand_shape'], axis=0))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c20_1d', 'c20_1d'], ['reshape2_shape'], axis=0))
    
    nodes.append(helper.make_node('Reshape', ['sliced', 'reshape1_shape'], ['reshaped1']))
    nodes.append(helper.make_node('Expand', ['reshaped1', 'expand_shape'], ['expanded']))
    nodes.append(helper.make_node('Reshape', ['expanded', 'reshape2_shape'], ['output']))
    
    graph = helper.make_graph(nodes, 'task108_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task108_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task108.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task108.onnx')
    
    with open('task108.json', 'r') as f:
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
model = onnx.load("/project/repairs/task108.onnx")
