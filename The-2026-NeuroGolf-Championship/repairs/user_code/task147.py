# Source: predicted/test_onnx_task147.py — ONNX graph construction code
# Verified model: repairs/task147.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 147: '3's with orthogonal '3' neighbors become '8'.
    """
    I64 = TensorProto.INT64
    F32 = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c3', 3)
    make_const('c8', 8)
    make_const('f0_5', 0.5, F32)
    
    nodes.append(helper.make_node('Equal', ['input', 'c3'], ['is_3']))
    nodes.append(helper.make_node('Cast', ['is_3'], ['is_3_f32'], to=int(F32)))
    
    make_const('axes_1', [1])
    nodes.append(helper.make_node('Unsqueeze', ['is_3_f32', 'axes_1'], ['mask3']))
    
    # Conv2D with cross kernel
    weight = np.array([[[[0, 1, 0],
                         [1, 0, 1],
                         [0, 1, 0]]]], dtype=np.float32)
    nodes.append(helper.make_node('Constant', [], ['conv_w'], value=helper.make_tensor('conv_w_v', F32, weight.shape, weight.flatten().tolist())))
    
    nodes.append(helper.make_node('Conv', ['mask3', 'conv_w'], ['neighbor_count'], pads=[1, 1, 1, 1]))
    
    nodes.append(helper.make_node('Squeeze', ['neighbor_count', 'axes_1'], ['neighbor_count_sq']))
    
    nodes.append(helper.make_node('Greater', ['neighbor_count_sq', 'f0_5'], ['has_neighbor']))
    nodes.append(helper.make_node('And', ['is_3', 'has_neighbor'], ['to_change']))
    
    nodes.append(helper.make_node('Where', ['to_change', 'c8', 'input'], ['output']))
    
    graph = helper.make_graph(nodes, 'task147_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task147_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task147.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task147.onnx')
    
    with open('task147.json', 'r') as f:
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
model = onnx.load("/project/repairs/task147.onnx")
