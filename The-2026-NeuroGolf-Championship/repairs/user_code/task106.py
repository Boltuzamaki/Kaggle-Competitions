# Source: predicted/test_onnx_task106.py — ONNX graph construction code
# Verified model: repairs/task106.onnx
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
    
    make_const('starts', [-1])
    make_const('ends', [-1000])
    make_const('steps', [-1])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    
    # I_T
    nodes.append(helper.make_node('Transpose', ['input'], ['I_T'], perm=[0, 2, 1]))
    
    # I_CW (reverse axis 2 of I_T)
    nodes.append(helper.make_node('Slice', ['I_T', 'starts', 'ends', 'axes_2', 'steps'], ['I_CW']))
    
    # I_CCW (reverse axis 1 of I_T)
    nodes.append(helper.make_node('Slice', ['I_T', 'starts', 'ends', 'axes_1', 'steps'], ['I_CCW']))
    
    # I_180 (reverse axis 1 and 2 of input)
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes_1', 'steps'], ['I_180_1']))
    nodes.append(helper.make_node('Slice', ['I_180_1', 'starts', 'ends', 'axes_2', 'steps'], ['I_180']))
    
    # Concat
    nodes.append(helper.make_node('Concat', ['input', 'I_CW'], ['top_half'], axis=2))
    nodes.append(helper.make_node('Concat', ['I_CCW', 'I_180'], ['bot_half'], axis=2))
    nodes.append(helper.make_node('Concat', ['top_half', 'bot_half'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task106_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task106_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task106.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task106.onnx')
    
    with open('task106.json', 'r') as f:
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
model = onnx.load("/project/repairs/task106.onnx")
