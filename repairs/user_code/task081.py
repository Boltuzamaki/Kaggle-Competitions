# Source: predicted/test_onnx_task081.py — ONNX graph construction code
# Verified model: repairs/task081.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def K(name, val, dtype=np.int64):
    return helper.make_tensor_value_info(name, dtype, np.array(val).shape)

def create_onnx_model():
    I64 = TensorProto.INT64
    F = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    # Constants
    nodes.append(helper.make_node('Constant', [], ['c8'], value=helper.make_tensor('c8_v', I64, [], [8])))
    nodes.append(helper.make_node('Constant', [], ['c0'], value=helper.make_tensor('c0_v', I64, [], [0])))
    nodes.append(helper.make_node('Constant', [], ['c1'], value=helper.make_tensor('c1_v', I64, [], [1])))
    nodes.append(helper.make_node('Constant', [], ['c3_f'], value=helper.make_tensor('c3_f_v', F, [], [3.0])))
    nodes.append(helper.make_node('Constant', [], ['axes_1'], value=helper.make_tensor('axes_1_v', I64, [1], [1])))
    
    # Filter 2x2 of 1.0s
    W_2x2 = np.ones((1, 1, 2, 2), dtype=np.float32)
    nodes.append(helper.make_node('Constant', [], ['W_2x2'], value=helper.make_tensor('W_2x2_v', F, [1, 1, 2, 2], W_2x2.flatten().tolist())))
    
    # is_8 = (input == 8)
    nodes.append(helper.make_node('Equal', ['input', 'c8'], ['is_8']))
    nodes.append(helper.make_node('Cast', ['is_8'], ['is_8_f'], to=F))
    
    # Unsqueeze to 4D
    nodes.append(helper.make_node('Unsqueeze', ['is_8_f', 'axes_1'], ['is_8_4d']))
    
    # count = Conv(is_8_4d, W_2x2)  # output shape (batch, 1, H-1, W-1)
    nodes.append(helper.make_node('Conv', ['is_8_4d', 'W_2x2'], ['count_4d']))
    
    # is_3 = (count == 3.0)
    nodes.append(helper.make_node('Equal', ['count_4d', 'c3_f'], ['is_3']))
    nodes.append(helper.make_node('Cast', ['is_3'], ['is_3_f'], to=F))
    
    # ConvTranspose to spread the '1' to the 2x2 area
    # If is_3 is true at (y,x), we want to mark (y,x), (y,x+1), (y+1,x), (y+1,x+1)
    nodes.append(helper.make_node('ConvTranspose', ['is_3_f', 'W_2x2'], ['spread_4d']))
    
    # > 0.5
    nodes.append(helper.make_node('Constant', [], ['c05_f'], value=helper.make_tensor('c05_f_v', F, [], [0.5])))
    nodes.append(helper.make_node('Greater', ['spread_4d', 'c05_f'], ['spread_mask_4d']))
    nodes.append(helper.make_node('Squeeze', ['spread_mask_4d', 'axes_1'], ['spread_mask']))
    
    # is_0 = (input == 0)
    nodes.append(helper.make_node('Equal', ['input', 'c0'], ['is_0']))
    
    # to_fill = spread_mask & is_0
    nodes.append(helper.make_node('And', ['spread_mask', 'is_0'], ['to_fill']))
    
    # output = Where(to_fill, 1, input)
    nodes.append(helper.make_node('Where', ['to_fill', 'c1', 'input'], ['output']))
    
    graph = helper.make_graph(
        nodes,
        'task081_graph',
        [input_info],
        [output_info]
    )
    
    model = helper.make_model(graph, producer_name='task081_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task081.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task081.onnx')
    
    with open('task081.json', 'r') as f:
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
model = onnx.load("/project/repairs/task081.onnx")
