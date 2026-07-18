# Source: predicted/test_onnx_task083.py — ONNX graph construction code
# Verified model: repairs/task083.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def K(name, val, dtype=np.int64):
    return helper.make_tensor_value_info(name, dtype, np.array(val).shape)

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H_out', 'W_out'])
    
    nodes = []
    
    # Constants
    nodes.append(helper.make_node('Constant', [], ['starts'], value=helper.make_tensor('starts_v', I64, [1], [-1])))
    # -INT_MAX for ends to go all the way to index 0
    nodes.append(helper.make_node('Constant', [], ['ends'], value=helper.make_tensor('ends_v', I64, [1], [-9223372036854775808])))
    nodes.append(helper.make_node('Constant', [], ['axes_1'], value=helper.make_tensor('axes_1_v', I64, [1], [1])))
    nodes.append(helper.make_node('Constant', [], ['axes_2'], value=helper.make_tensor('axes_2_v', I64, [1], [2])))
    nodes.append(helper.make_node('Constant', [], ['steps'], value=helper.make_tensor('steps_v', I64, [1], [-1])))
    
    # Reverses
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes_2', 'steps'], ['rev_W']))
    nodes.append(helper.make_node('Slice', ['input', 'starts', 'ends', 'axes_1', 'steps'], ['rev_H']))
    nodes.append(helper.make_node('Slice', ['rev_H', 'starts', 'ends', 'axes_2', 'steps'], ['rev_both']))
    
    # Concats
    nodes.append(helper.make_node('Concat', ['input', 'rev_W'], ['top'], axis=2))
    nodes.append(helper.make_node('Concat', ['rev_H', 'rev_both'], ['bottom'], axis=2))
    nodes.append(helper.make_node('Concat', ['top', 'bottom'], ['output'], axis=1))
    
    graph = helper.make_graph(
        nodes,
        'task083_graph',
        [input_info],
        [output_info]
    )
    
    model = helper.make_model(graph, producer_name='task083_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task083.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task083.onnx')
    
    with open('task083.json', 'r') as f:
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
model = onnx.load("/project/repairs/task083.onnx")
