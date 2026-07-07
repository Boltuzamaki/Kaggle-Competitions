# Source: predicted/test_onnx_task188.py — ONNX graph construction code
# Verified model: repairs/task188.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 188: Slice half of height or half of width, depending on which one has identical halves.
    """
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
    make_const('c1_0d', 1)
    
    # Get H and W
    make_const('axes_0', [0])
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1', 'c2', 'axes_0'], ['H']))
    make_const('c3', [3])
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2', 'c3', 'axes_0'], ['W']))
    
    # H_div_2 = H / 2
    nodes.append(helper.make_node('Div', ['H', 'c2'], ['H_div_2']))
    nodes.append(helper.make_node('Div', ['W', 'c2'], ['W_div_2']))
    
    # top = input[:, 0:H/2, :]
    # bottom = input[:, H/2:H/2*2, :]
    make_const('axes_1', [1])
    nodes.append(helper.make_node('Slice', ['input', 'c0', 'H_div_2', 'axes_1'], ['top']))
    
    nodes.append(helper.make_node('Mul', ['H_div_2', 'c2'], ['H_div_2_mul_2']))
    nodes.append(helper.make_node('Slice', ['input', 'H_div_2', 'H_div_2_mul_2', 'axes_1'], ['bottom']))
    
    nodes.append(helper.make_node('Equal', ['top', 'bottom'], ['eq']))
    nodes.append(helper.make_node('Cast', ['eq'], ['eq_i64'], to=int(I64)))
    
    # ReduceMin(eq_i64)
    nodes.append(helper.make_node('ReduceMin', ['eq_i64'], ['min_eq'], keepdims=0))
    
    nodes.append(helper.make_node('Equal', ['min_eq', 'c1_0d'], ['is_vert_0d']))
    make_const('axes_new', [0])
    nodes.append(helper.make_node('Unsqueeze', ['is_vert_0d', 'axes_new'], ['is_vert']))
    
    nodes.append(helper.make_node('Where', ['is_vert', 'H_div_2', 'H'], ['end_h']))
    nodes.append(helper.make_node('Where', ['is_vert', 'W', 'W_div_2'], ['end_w']))
    
    # Concat ends
    nodes.append(helper.make_node('Concat', ['end_h', 'end_w'], ['ends'], axis=0))
    make_const('starts', [0, 0])
    make_const('axes_1_2', [1, 2])
    
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes_1_2'], ['output']))
    
    graph = helper.make_graph(nodes, 'task188_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task188_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task188.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task188.onnx')
    
    with open('task188.json', 'r') as f:
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
model = onnx.load("/project/repairs/task188.onnx")
