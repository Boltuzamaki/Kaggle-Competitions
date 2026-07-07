# Source: predicted/test_onnx_task099.py — ONNX graph construction code
# Verified model: repairs/task099.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    F32 = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', [0])
    make_const('c1', [1])
    make_const('c0_5', [0.5], dtype=F32)
    make_const('c0_f32', [0.0], dtype=F32)
    make_const('cm1_1d', [-1])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('axes_0', [0])
    make_const('axes_2_1d', [2])
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('Slice', ['shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_1d']))
    nodes.append(helper.make_node('Slice', ['shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_1d']))
    nodes.append(helper.make_node('Concat', ['cm1_1d', 'c1_1d', 'H_1d', 'W_1d'], ['shape_4d'], axis=0))
    
    nodes.append(helper.make_node('Equal', ['input', 'c1'], ['mask1']))
    nodes.append(helper.make_node('Cast', ['mask1'], ['mask1_i64'], to=I64))
    
    nodes.append(helper.make_node('CumSum', ['mask1_i64', 'axes_2_1d'], ['left_sum']))
    nodes.append(helper.make_node('Greater', ['left_sum', 'c0'], ['left_1']))
    
    nodes.append(helper.make_node('CumSum', ['mask1_i64', 'axes_2_1d'], ['right_sum'], reverse=1))
    nodes.append(helper.make_node('Greater', ['right_sum', 'c0'], ['right_1']))
    
    nodes.append(helper.make_node('And', ['left_1', 'right_1'], ['horiz_inside']))
    
    nodes.append(helper.make_node('Cast', ['horiz_inside'], ['horiz_inside_f32'], to=F32))
    nodes.append(helper.make_node('Reshape', ['horiz_inside_f32', 'shape_4d'], ['horiz_inside_4d']))
    
    kernel_up = np.zeros((1, 1, 3, 3), dtype=np.float32)
    kernel_up[0, 0, 2, 1] = 1.0
    make_const('kernel_up', kernel_up, dtype=F32)
    
    nodes.append(helper.make_node('Conv', ['horiz_inside_4d', 'kernel_up'], ['shifted_4d'], pads=[1, 1, 1, 1]))
    nodes.append(helper.make_node('Greater', ['shifted_4d', 'c0_5'], ['shifted_up_4d']))
    nodes.append(helper.make_node('Reshape', ['shifted_up_4d', 'shape'], ['shifted_up']))
    
    nodes.append(helper.make_node('Or', ['horiz_inside', 'shifted_up'], ['ext_bb']))
    
    nodes.append(helper.make_node('Not', ['mask1'], ['not_1']))
    nodes.append(helper.make_node('And', ['ext_bb', 'not_1'], ['flood_mask']))
    nodes.append(helper.make_node('Reshape', ['flood_mask', 'shape_4d'], ['flood_mask_4d']))
    
    nodes.append(helper.make_node('Greater', ['input', 'c1'], ['is_dot']))
    nodes.append(helper.make_node('Where', ['is_dot', 'input', 'c0'], ['dots']))
    nodes.append(helper.make_node('Cast', ['dots'], ['dots_f32'], to=F32))
    nodes.append(helper.make_node('Reshape', ['dots_f32', 'shape_4d'], ['dots_4d']))
    
    curr = 'dots_4d'
    for i in range(15):
        pool_out = f'pool_{i}'
        nodes.append(helper.make_node('MaxPool', [curr], [pool_out], kernel_shape=[3, 3], pads=[1,1,1,1]))
        next_curr = f'curr_{i+1}'
        nodes.append(helper.make_node('Where', ['flood_mask_4d', pool_out, 'c0_f32'], [next_curr]))
        curr = next_curr
        
    nodes.append(helper.make_node('Cast', [curr], ['filled_4d'], to=I64))
    nodes.append(helper.make_node('Reshape', ['filled_4d', 'shape'], ['filled']))
    
    nodes.append(helper.make_node('Where', ['flood_mask', 'filled', 'input'], ['output']))
    
    graph = helper.make_graph(nodes, 'task099_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task099_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task099.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task099.onnx')
    
    with open('task099.json', 'r') as f:
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
model = onnx.load("/project/repairs/task099.onnx")
