# Source: predicted/test_onnx_task152.py — ONNX graph construction code
# Verified model: repairs/task152.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 152: Mirror horizontally and then mirror vertically to double size (like 142).
    """
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H2', 'W2'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('starts', [1000])
    make_const('ends', [-1000])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    make_const('steps', [-1])
    
    # Reverse width
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes_2', 'steps'], ['rev_w']))
    # Concat width
    nodes.append(helper.make_node('Concat', ['input', 'rev_w'], ['top'], axis=2))
    
    # Reverse height
    nodes.append(helper.make_node('Slice', ['top', 'starts', 'ends', 'axes_1', 'steps'], ['rev_h']))
    # Concat height
    nodes.append(helper.make_node('Concat', ['top', 'rev_h'], ['output'], axis=1))
    
    graph = helper.make_graph(nodes, 'task152_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task152_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task152.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task152.onnx')
    
    with open('task152.json', 'r') as f:
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
model = onnx.load("/project/repairs/task152.onnx")
