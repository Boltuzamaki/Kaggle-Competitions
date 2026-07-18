# Source: predicted/test_onnx_task160.py — ONNX graph construction code
# Verified model: repairs/task160.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 160: Hollow squares of 1 become crosses of 2. 
    """
    I64 = TensorProto.INT64
    F32 = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('c2', 2)
    make_const('c8_f', 8.0, F32)
    make_const('f0_5', 0.5, F32)
    
    nodes.append(helper.make_node('Equal', ['input', 'c1'], ['is_1']))
    nodes.append(helper.make_node('Cast', ['is_1'], ['M'], to=int(F32)))
    
    nodes.append(helper.make_node('Equal', ['input', 'c0'], ['is_0']))
    
    make_const('axes_1', [1])
    nodes.append(helper.make_node('Unsqueeze', ['M', 'axes_1'], ['M_4d']))
    
    # Kernel for hollow square:
    w_hollow = np.array([[[[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]]]], dtype=np.float32)
    nodes.append(helper.make_node('Constant', [], ['w_hollow'], value=helper.make_tensor('w_hollow_v', F32, w_hollow.shape, w_hollow.flatten().tolist())))
    
    nodes.append(helper.make_node('Conv', ['M_4d', 'w_hollow'], ['C'], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node('Squeeze', ['C', 'axes_1'], ['C_sq']))
    
    nodes.append(helper.make_node('Equal', ['C_sq', 'c8_f'], ['is_8']))
    nodes.append(helper.make_node('And', ['is_8', 'is_0'], ['Match']))
    nodes.append(helper.make_node('Cast', ['Match'], ['Match_f'], to=int(F32)))
    nodes.append(helper.make_node('Unsqueeze', ['Match_f', 'axes_1'], ['Match_4d']))
    
    # Kernel for cross of 2s
    w_cross = np.array([[[[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]]], dtype=np.float32)
    nodes.append(helper.make_node('Constant', [], ['w_cross'], value=helper.make_tensor('w_cross_v', F32, w_cross.shape, w_cross.flatten().tolist())))
    
    nodes.append(helper.make_node('Conv', ['Match_4d', 'w_cross'], ['To_2'], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node('Squeeze', ['To_2', 'axes_1'], ['To_2_sq']))
    nodes.append(helper.make_node('Greater', ['To_2_sq', 'f0_5'], ['to_2_mask']))
    
    nodes.append(helper.make_node('Conv', ['Match_4d', 'w_hollow'], ['To_0'], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node('Squeeze', ['To_0', 'axes_1'], ['To_0_sq']))
    nodes.append(helper.make_node('Greater', ['To_0_sq', 'f0_5'], ['to_0_mask']))
    
    nodes.append(helper.make_node('Where', ['to_0_mask', 'c0', 'input'], ['out_0']))
    nodes.append(helper.make_node('Where', ['to_2_mask', 'c2', 'out_0'], ['output']))
    
    graph = helper.make_graph(nodes, 'task160_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task160_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task160.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task160.onnx')
    
    with open('task160.json', 'r') as f:
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
model = onnx.load("/project/repairs/task160.onnx")
