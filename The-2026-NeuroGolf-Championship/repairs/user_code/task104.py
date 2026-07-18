# Source: predicted/test_onnx_task104.py — ONNX graph construction code
# Verified model: repairs/task104.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', '9', '9'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', [0])
    make_const('c1', [1])
    make_const('c2', [2])
    make_const('c3', [3])
    
    # Get color_C
    nodes.append(helper.make_node('Equal', ['input', 'c2'], ['is_2']))
    nodes.append(helper.make_node('Not', ['is_2'], ['not_2']))
    nodes.append(helper.make_node('Equal', ['input', 'c0'], ['is_0']))
    nodes.append(helper.make_node('Not', ['is_0'], ['not_0']))
    nodes.append(helper.make_node('And', ['not_2', 'not_0'], ['is_C_bool']))
    nodes.append(helper.make_node('Cast', ['is_C_bool'], ['is_C'], to=I64))
    nodes.append(helper.make_node('Mul', ['input', 'is_C'], ['input_C_only']))
    nodes.append(helper.make_node('ReduceMax', ['input_C_only'], ['color_C'], axes=[1, 2], keepdims=1))
    
    make_const('axes_12', [1, 2])
    
    # Extract corners
    make_const('st_tl', [0, 0])
    make_const('en_tl', [1, 1])
    nodes.append(helper.make_node('Slice', ['input', 'st_tl', 'en_tl', 'axes_12'], ['tl']))
    
    make_const('st_br', [2, 2])
    make_const('en_br', [3, 3])
    nodes.append(helper.make_node('Slice', ['input', 'st_br', 'en_br', 'axes_12'], ['br']))
    
    make_const('st_tr', [0, 2])
    make_const('en_tr', [1, 3])
    nodes.append(helper.make_node('Slice', ['input', 'st_tr', 'en_tr', 'axes_12'], ['tr']))
    
    make_const('st_bl', [2, 0])
    make_const('en_bl', [3, 1])
    nodes.append(helper.make_node('Slice', ['input', 'st_bl', 'en_bl', 'axes_12'], ['bl']))
    
    # Check matches
    nodes.append(helper.make_node('Equal', ['tl', 'color_C'], ['is_tl']))
    nodes.append(helper.make_node('Equal', ['br', 'color_C'], ['is_br']))
    nodes.append(helper.make_node('Equal', ['tr', 'color_C'], ['is_tr']))
    nodes.append(helper.make_node('Equal', ['bl', 'color_C'], ['is_bl']))
    
    nodes.append(helper.make_node('Cast', ['is_tl'], ['w_tl'], to=I64))
    nodes.append(helper.make_node('Cast', ['is_br'], ['w_br'], to=I64))
    nodes.append(helper.make_node('Cast', ['is_tr'], ['w_tr'], to=I64))
    nodes.append(helper.make_node('Cast', ['is_bl'], ['w_bl'], to=I64))
    
    # Patterns
    pat_tl = np.zeros((9, 9), dtype=np.int64)
    pat_tl[0:4, 0:4] = 1
    pat_tl[4:8, 4:8] = 1
    make_const('pat_tl', pat_tl)
    
    pat_br = np.zeros((9, 9), dtype=np.int64)
    pat_br[1:5, 1:5] = 1
    pat_br[5:9, 5:9] = 1
    make_const('pat_br', pat_br)
    
    pat_tr = np.zeros((9, 9), dtype=np.int64)
    pat_tr[0:4, 5:9] = 1
    pat_tr[4:8, 1:5] = 1
    make_const('pat_tr', pat_tr)
    
    pat_bl = np.zeros((9, 9), dtype=np.int64)
    pat_bl[1:5, 4:8] = 1
    pat_bl[5:9, 0:4] = 1
    make_const('pat_bl', pat_bl)
    
    # Multiply and add
    nodes.append(helper.make_node('Mul', ['w_tl', 'pat_tl'], ['out_tl']))
    nodes.append(helper.make_node('Mul', ['w_br', 'pat_br'], ['out_br']))
    nodes.append(helper.make_node('Mul', ['w_tr', 'pat_tr'], ['out_tr']))
    nodes.append(helper.make_node('Mul', ['w_bl', 'pat_bl'], ['out_bl']))
    
    nodes.append(helper.make_node('Add', ['out_tl', 'out_br'], ['out_12']))
    nodes.append(helper.make_node('Add', ['out_12', 'out_tr'], ['out_123']))
    nodes.append(helper.make_node('Add', ['out_123', 'out_bl'], ['out_sum']))
    
    nodes.append(helper.make_node('Mul', ['out_sum', 'color_C'], ['output']))
    
    graph = helper.make_graph(nodes, 'task104_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task104_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task104.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task104.onnx')
    
    with open('task104.json', 'r') as f:
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
model = onnx.load("/project/repairs/task104.onnx")
