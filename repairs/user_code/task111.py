# Source: predicted/test_onnx_task111.py — ONNX graph construction code
# Verified model: repairs/task111.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    F32 = TensorProto.FLOAT
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'h_out', 'w_out'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c0_1d', [0])
    make_const('c5', 5)
    make_const('c1_1d', [1])
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_dim']))
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    
    # Add batch and channel dims -> [batch, 1, H, W]
    make_const('shape_4d', [1, 1, -1, -1])
    # Wait, shape_4d with -1 will flatten. Better to use Concat.
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c1_1d', 'H_dim', 'W_dim'], ['shape_4d_dyn'], axis=0))
    nodes.append(helper.make_node('Reshape', ['input', 'shape_4d_dyn'], ['input_4d']))
    
    nodes.append(helper.make_node('Greater', ['input_4d', 'c0'], ['gt_0']))
    nodes.append(helper.make_node('Equal', ['input_4d', 'c5'], ['eq_5']))
    nodes.append(helper.make_node('Not', ['eq_5'], ['neq_5']))
    nodes.append(helper.make_node('And', ['gt_0', 'neq_5'], ['mask_bool']))
    
    nodes.append(helper.make_node('Cast', ['mask_bool'], ['mask'], to=F32))
    nodes.append(helper.make_node('Cast', ['eq_5'], ['active'], to=F32))
    
    # Conv weights
    make_const('w_conv', np.ones((1, 1, 3, 3), dtype=np.float32), dtype=F32)
    make_const('half', 0.5, dtype=F32)
    
    curr_active = 'active'
    for i in range(8):
        next_conv = f'conv_{i}'
        nodes.append(helper.make_node('Conv', [curr_active, 'w_conv'], [next_conv], pads=[1, 1, 1, 1]))
        next_act = f'act_{i}'
        nodes.append(helper.make_node('Greater', [next_conv, 'half'], [next_act + '_bool']))
        nodes.append(helper.make_node('Cast', [next_act + '_bool'], [next_act + '_f'], to=F32))
        curr_active = f'active_{i}'
        nodes.append(helper.make_node('Mul', [next_act + '_f', 'mask'], [curr_active]))
        
    nodes.append(helper.make_node('Cast', [curr_active], ['active_int_4d'], to=I64))
    nodes.append(helper.make_node('Squeeze', ['active_int_4d', 'c1_1d'], ['active_int'])) # [batch, H, W]
    
    nodes.append(helper.make_node('ReduceMax', ['active_int'], ['r_mask'], axes=[2], keepdims=0)) # [batch, H]
    nodes.append(helper.make_node('ReduceMax', ['active_int'], ['c_mask'], axes=[1], keepdims=0)) # [batch, W]
    
    nodes.append(helper.make_node('Squeeze', ['H_dim', 'axes_0'], ['H_scalar']))
    nodes.append(helper.make_node('Squeeze', ['W_dim', 'axes_0'], ['W_scalar']))
    nodes.append(helper.make_node('Range', ['c0', 'H_scalar', 'c1_1d'], ['r_indices'])) # [H]
    nodes.append(helper.make_node('Range', ['c0', 'W_scalar', 'c1_1d'], ['c_indices'])) # [W]
    
    nodes.append(helper.make_node('Greater', ['r_mask', 'c0'], ['r_mask_bool']))
    nodes.append(helper.make_node('Greater', ['c_mask', 'c0'], ['c_mask_bool']))
    
    make_const('cm1', -1)
    nodes.append(helper.make_node('Where', ['r_mask_bool', 'r_indices', 'H_scalar'], ['r_valid_min']))
    nodes.append(helper.make_node('ReduceMin', ['r_valid_min'], ['r_min'], axes=[1], keepdims=0)) # [batch]
    nodes.append(helper.make_node('Where', ['r_mask_bool', 'r_indices', 'cm1'], ['r_valid_max']))
    nodes.append(helper.make_node('ReduceMax', ['r_valid_max'], ['r_max'], axes=[1], keepdims=0)) # [batch]
    
    nodes.append(helper.make_node('Where', ['c_mask_bool', 'c_indices', 'W_scalar'], ['c_valid_min']))
    nodes.append(helper.make_node('ReduceMin', ['c_valid_min'], ['c_min'], axes=[1], keepdims=0)) # [batch]
    nodes.append(helper.make_node('Where', ['c_mask_bool', 'c_indices', 'cm1'], ['c_valid_max']))
    nodes.append(helper.make_node('ReduceMax', ['c_valid_max'], ['c_max'], axes=[1], keepdims=0)) # [batch]
    
    nodes.append(helper.make_node('Add', ['r_max', 'c1_1d'], ['r_max_1']))
    nodes.append(helper.make_node('Add', ['c_max', 'c1_1d'], ['c_max_1']))
    
    nodes.append(helper.make_node('Concat', ['r_min', 'c_min'], ['starts'], axis=0))
    nodes.append(helper.make_node('Concat', ['r_max_1', 'c_max_1'], ['ends'], axis=0))
    
    nodes.append(helper.make_node('Mul', ['input', 'active_int'], ['masked_input']))
    
    make_const('axes_1_2', [1, 2])
    nodes.append(helper.make_node('Slice', ['masked_input', 'starts', 'ends', 'axes_1_2'], ['output']))
    
    graph = helper.make_graph(nodes, 'task111_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task111_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task111.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task111.onnx')
    
    with open('task111.json', 'r') as f:
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
model = onnx.load("/project/repairs/task111.onnx")
