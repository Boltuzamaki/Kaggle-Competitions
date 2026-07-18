# Source: predicted/test_onnx_task303.py — ONNX graph construction code
# Verified model: repairs/task303.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 303: Replace fully-0 rows and fully-0 cols with 2.
    """
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H_out', 'W_out'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    
    nodes.append(helper.make_node('Equal', ['input', 'c0'], ['is_zero_bool']))
    nodes.append(helper.make_node('Cast', ['is_zero_bool'], ['is_zero'], to=I64))
    
    nodes.append(helper.make_node('ReduceMin', ['is_zero'], ['row_min'], axes=[2], keepdims=1))
    nodes.append(helper.make_node('ReduceMin', ['is_zero'], ['col_min'], axes=[1], keepdims=1))
    
    nodes.append(helper.make_node('Max', ['row_min', 'col_min'], ['is_line']))
    
    make_const('c2', 2)
    nodes.append(helper.make_node('Mul', ['is_line', 'c2'], ['added_val']))
    
    nodes.append(helper.make_node('Add', ['input', 'added_val'], ['output']))
    
    graph = helper.make_graph(nodes, 'task303_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task303_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task303.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task303.onnx')
    
    with open('task303.json', 'r') as f:
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
model = onnx.load("/project/repairs/task303.onnx")
