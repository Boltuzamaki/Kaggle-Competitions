# Source: predicted/test_onnx_task139.py — ONNX graph construction code
# Verified model: repairs/task139.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 139: Find 3x3 bounding boxes of '4's and fill the 0s inside them with '7'.
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
    make_const('c4', 4)
    make_const('c7', 7)
    make_const('f0_5', 0.5, F32)
    
    # mask4: [batch, 1, H, W]
    nodes.append(helper.make_node('Equal', ['input', 'c4'], ['is_4']))
    nodes.append(helper.make_node('Cast', ['is_4'], ['is_4_f32'], to=int(F32)))
    
    make_const('shape_b1HW', [-1, 1, -1, -1])
    nodes.append(helper.make_node('Shape', ['is_4_f32'], ['in_shape']))
    
    # We can just reshape to [batch, 1, H, W] using dynamic shape?
    # Actually Reshape with [-1, 1, H, W] is hard if H, W are unknown.
    # But wait, Reshape to [batch, 1, H, W] is easy:
    # Just Unsqueeze axis 1!
    make_const('axes_1', [1])
    nodes.append(helper.make_node('Unsqueeze', ['is_4_f32', 'axes_1'], ['mask4']))
    
    # Conv2D
    weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    nodes.append(helper.make_node('Constant', [], ['conv_w'], value=helper.make_tensor('conv_w_v', F32, weight.shape, weight.flatten().tolist())))
    
    nodes.append(helper.make_node('Conv', ['mask4', 'conv_w'], ['C'], pads=[1, 1, 1, 1]))
    
    # MaxPool2D
    nodes.append(helper.make_node('MaxPool', ['C'], ['M'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    
    # Centers = (C == M) & (C > 0.5)
    nodes.append(helper.make_node('Equal', ['C', 'M'], ['C_eq_M']))
    nodes.append(helper.make_node('Greater', ['C', 'f0_5'], ['C_gt_05']))
    nodes.append(helper.make_node('And', ['C_eq_M', 'C_gt_05'], ['centers']))
    
    # Boxes_f = MaxPool(Centers_f)
    nodes.append(helper.make_node('Cast', ['centers'], ['centers_f32'], to=int(F32)))
    nodes.append(helper.make_node('MaxPool', ['centers_f32'], ['boxes_f32'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]))
    
    nodes.append(helper.make_node('Greater', ['boxes_f32', 'f0_5'], ['boxes_1ch']))
    
    # Squeeze back to [batch, H, W]
    nodes.append(helper.make_node('Squeeze', ['boxes_1ch', 'axes_1'], ['boxes']))
    
    # output = Where(boxes & (input == 0), 7, input)
    nodes.append(helper.make_node('Equal', ['input', 'c0'], ['is_0']))
    nodes.append(helper.make_node('And', ['boxes', 'is_0'], ['to_fill']))
    
    nodes.append(helper.make_node('Where', ['to_fill', 'c7', 'input'], ['output']))
    
    graph = helper.make_graph(nodes, 'task139_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task139_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task139.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task139.onnx')
    
    with open('task139.json', 'r') as f:
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
model = onnx.load("/project/repairs/task139.onnx")
