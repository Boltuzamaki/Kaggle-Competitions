# Source: predicted/test_onnx_task127.py — ONNX graph construction code
# Verified model: repairs/task127.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 127: Regions separated by 5s. Each region contains a colored dot (1, 2, 3, 4).
    Fill the region with (dot_color + 5). 5s remain 5s.
    Use MaxPool + Masking to dilate the dots within the regions.
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
    make_const('c5', 5)
    make_const('axes_0', [0])
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_dim']))
    
    # 4D shape for MaxPool: [batch, 1, H, W]
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c1_1d', 'H_dim', 'W_dim'], ['shape_4d'], axis=0))
    
    # Initial state: input, but replace 5 with 0
    nodes.append(helper.make_node('Equal', ['input', 'c5'], ['is_5']))
    nodes.append(helper.make_node('Where', ['is_5', 'c0', 'input'], ['state0']))
    
    # Convert state to FLOAT for MaxPool
    nodes.append(helper.make_node('Cast', ['state0'], ['state0_f'], to=int(F32)))
    nodes.append(helper.make_node('Reshape', ['state0_f', 'shape_4d'], ['state_4d']))
    
    # Unroll 15 times to ensure full coverage
    curr_state = 'state_4d'
    for i in range(15):
        next_state_pool = f'state_pool_{i}'
        nodes.append(helper.make_node('MaxPool', [curr_state], [next_state_pool], kernel_shape=[3,3], pads=[1,1,1,1]))
        
        # Mask out 5s
        # We need is_5 as 4D boolean
        if i == 0:
            nodes.append(helper.make_node('Reshape', ['is_5', 'shape_4d'], ['is_5_4d']))
            make_const('c0_f', 0.0, dtype=F32)
        
        next_state_masked = f'state_masked_{i}'
        nodes.append(helper.make_node('Where', ['is_5_4d', 'c0_f', next_state_pool], [next_state_masked]))
        curr_state = next_state_masked
    
    # Reshape back to 3D and cast to I64
    nodes.append(helper.make_node('Reshape', [curr_state, 'in_shape'], ['state_final_f']))
    nodes.append(helper.make_node('Cast', ['state_final_f'], ['state_final'], to=int(I64)))
    
    # output where state > 0 is state + 5.
    # Where input == 5, it is 5.
    # Where state == 0 (no color), keep 0.
    nodes.append(helper.make_node('Add', ['state_final', 'c5'], ['state_plus_5']))
    nodes.append(helper.make_node('Greater', ['state_final', 'c0'], ['is_colored']))
    nodes.append(helper.make_node('Where', ['is_colored', 'state_plus_5', 'c0'], ['colored_out']))
    nodes.append(helper.make_node('Where', ['is_5', 'c5', 'colored_out'], ['output']))
    
    graph = helper.make_graph(nodes, 'task127_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task127_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task127.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task127.onnx')
    
    with open('task127.json', 'r') as f:
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
model = onnx.load("/project/repairs/task127.onnx")
