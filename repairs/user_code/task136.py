# Source: predicted/test_onnx_task136.py — ONNX graph construction code
# Verified model: repairs/task136.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 136: '1' shoots diagonal beam up-left. '2' shoots diagonal beam down-right.
    """
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('c2', 2)
    make_const('c1000', 1000)
    make_const('cm1', -1)
    make_const('axes_0', [0])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_dim']))
    nodes.append(helper.make_node('Squeeze', ['H_dim', 'axes_0'], ['H']))
    nodes.append(helper.make_node('Squeeze', ['W_dim', 'axes_0'], ['W']))
    
    nodes.append(helper.make_node('Range', ['c0', 'H', 'c1'], ['range_H']))
    nodes.append(helper.make_node('Range', ['c0', 'W', 'c1'], ['range_W']))
    make_const('shape_1H1', [1, -1, 1])
    make_const('shape_11W', [1, 1, -1])
    make_const('shape_1H', [1, -1])
    make_const('shape_1W', [1, -1])
    make_const('shape_b11', [-1, 1, 1])
    
    nodes.append(helper.make_node('Reshape', ['range_H', 'shape_1H1'], ['grid_r'])) # [1, H, 1]
    nodes.append(helper.make_node('Reshape', ['range_W', 'shape_11W'], ['grid_c'])) # [1, 1, W]
    nodes.append(helper.make_node('Reshape', ['range_H', 'shape_1H'], ['range_H_2d'])) # [1, H]
    nodes.append(helper.make_node('Reshape', ['range_W', 'shape_1W'], ['range_W_2d'])) # [1, W]
    
    nodes.append(helper.make_node('Sub', ['grid_r', 'grid_c'], ['r_minus_c'])) # [1, H, W]
    
    # 1 bounds
    nodes.append(helper.make_node('Equal', ['input', 'c1'], ['is_1']))
    nodes.append(helper.make_node('Cast', ['is_1'], ['is_1_i64'], to=int(I64)))
    nodes.append(helper.make_node('ReduceSum', ['is_1_i64', 'axes_2'], ['sum_r1'], keepdims=0))
    nodes.append(helper.make_node('Greater', ['sum_r1', 'c0'], ['has_r1']))
    nodes.append(helper.make_node('ReduceSum', ['is_1_i64', 'axes_1'], ['sum_c1'], keepdims=0))
    nodes.append(helper.make_node('Greater', ['sum_c1', 'c0'], ['has_c1']))
    
    nodes.append(helper.make_node('Where', ['has_r1', 'range_H_2d', 'c1000'], ['r1_inv']))
    nodes.append(helper.make_node('ReduceMin', ['r1_inv'], ['min_r1'], axes=[1], keepdims=1))
    nodes.append(helper.make_node('Where', ['has_c1', 'range_W_2d', 'c1000'], ['c1_inv']))
    nodes.append(helper.make_node('ReduceMin', ['c1_inv'], ['min_c1'], axes=[1], keepdims=1))
    
    nodes.append(helper.make_node('Reshape', ['min_r1', 'shape_b11'], ['min_r1_3d']))
    nodes.append(helper.make_node('Reshape', ['min_c1', 'shape_b11'], ['min_c1_3d']))
    
    # 2 bounds
    nodes.append(helper.make_node('Equal', ['input', 'c2'], ['is_2']))
    nodes.append(helper.make_node('Cast', ['is_2'], ['is_2_i64'], to=int(I64)))
    nodes.append(helper.make_node('ReduceSum', ['is_2_i64', 'axes_2'], ['sum_r2'], keepdims=0))
    nodes.append(helper.make_node('Greater', ['sum_r2', 'c0'], ['has_r2']))
    nodes.append(helper.make_node('ReduceSum', ['is_2_i64', 'axes_1'], ['sum_c2'], keepdims=0))
    nodes.append(helper.make_node('Greater', ['sum_c2', 'c0'], ['has_c2']))
    
    nodes.append(helper.make_node('Where', ['has_r2', 'range_H_2d', 'cm1'], ['r2_inv']))
    nodes.append(helper.make_node('ReduceMax', ['r2_inv'], ['max_r2'], axes=[1], keepdims=1))
    nodes.append(helper.make_node('Where', ['has_c2', 'range_W_2d', 'cm1'], ['c2_inv']))
    nodes.append(helper.make_node('ReduceMax', ['c2_inv'], ['max_c2'], axes=[1], keepdims=1))
    
    nodes.append(helper.make_node('Reshape', ['max_r2', 'shape_b11'], ['max_r2_3d']))
    nodes.append(helper.make_node('Reshape', ['max_c2', 'shape_b11'], ['max_c2_3d']))
    
    # presence
    nodes.append(helper.make_node('Cast', ['has_r1'], ['has_r1_i64'], to=int(I64)))
    nodes.append(helper.make_node('ReduceSum', ['has_r1_i64', 'axes_1'], ['count_1'], keepdims=1))
    nodes.append(helper.make_node('Greater', ['count_1', 'c0'], ['pres_1']))
    nodes.append(helper.make_node('Reshape', ['pres_1', 'shape_b11'], ['pres_1_3d']))
    
    nodes.append(helper.make_node('Cast', ['has_r2'], ['has_r2_i64'], to=int(I64)))
    nodes.append(helper.make_node('ReduceSum', ['has_r2_i64', 'axes_1'], ['count_2'], keepdims=1))
    nodes.append(helper.make_node('Greater', ['count_2', 'c0'], ['pres_2']))
    nodes.append(helper.make_node('Reshape', ['pres_2', 'shape_b11'], ['pres_2_3d']))
    
    # beam 1: r < min_r1 & c < min_c1 & r - c == min_r1 - min_c1
    nodes.append(helper.make_node('Less', ['grid_r', 'min_r1_3d'], ['r_lt_1']))
    nodes.append(helper.make_node('Less', ['grid_c', 'min_r1_3d'], ['c_lt_1'])) # Wait, min_r1_3d used for c! Bug!
    nodes.append(helper.make_node('Less', ['grid_c', 'min_c1_3d'], ['c_lt_1_fix']))
    nodes.append(helper.make_node('Sub', ['min_r1_3d', 'min_c1_3d'], ['diff1']))
    nodes.append(helper.make_node('Equal', ['r_minus_c', 'diff1'], ['eq1']))
    nodes.append(helper.make_node('And', ['r_lt_1', 'c_lt_1_fix'], ['lt_1']))
    nodes.append(helper.make_node('And', ['lt_1', 'eq1'], ['beam1_raw']))
    nodes.append(helper.make_node('And', ['beam1_raw', 'pres_1_3d'], ['beam1']))
    
    # beam 2: r > max_r2 & c > max_c2 & r - c == max_r2 - max_c2
    nodes.append(helper.make_node('Greater', ['grid_r', 'max_r2_3d'], ['r_gt_2']))
    nodes.append(helper.make_node('Greater', ['grid_c', 'max_c2_3d'], ['c_gt_2']))
    nodes.append(helper.make_node('Sub', ['max_r2_3d', 'max_c2_3d'], ['diff2']))
    nodes.append(helper.make_node('Equal', ['r_minus_c', 'diff2'], ['eq2']))
    nodes.append(helper.make_node('And', ['r_gt_2', 'c_gt_2'], ['gt_2']))
    nodes.append(helper.make_node('And', ['gt_2', 'eq2'], ['beam2_raw']))
    nodes.append(helper.make_node('And', ['beam2_raw', 'pres_2_3d'], ['beam2']))
    
    nodes.append(helper.make_node('Where', ['beam2', 'c2', 'input'], ['out_2']))
    nodes.append(helper.make_node('Where', ['beam1', 'c1', 'out_2'], ['output']))
    
    graph = helper.make_graph(nodes, 'task136_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task136_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task136.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task136.onnx')
    
    with open('task136.json', 'r') as f:
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
model = onnx.load("/project/repairs/task136.onnx")
