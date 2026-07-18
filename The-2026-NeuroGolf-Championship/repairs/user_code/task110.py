# Source: predicted/test_onnx_task110.py — ONNX graph construction code
# Verified model: repairs/task110.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('c30_1d', [30])
    make_const('axes_0', [0])
    make_const('c0', 0)
    
    # Dimensions
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_dim']))
    
    # Pad input to 30x30
    nodes.append(helper.make_node('Sub', ['c30_1d', 'H_dim'], ['pad_H']))
    nodes.append(helper.make_node('Sub', ['c30_1d', 'W_dim'], ['pad_W']))
    nodes.append(helper.make_node('Concat', ['c0_1d', 'c0_1d', 'c0_1d', 'c0_1d', 'pad_H', 'pad_W'], ['pads'], axis=0))
    nodes.append(helper.make_node('Pad', ['input', 'pads'], ['pad_in']))
    
    # Pad to 60x30 and 30x60
    make_const('pads_60_h', [0, 0, 0, 0, 30, 0])
    make_const('pads_60_w', [0, 0, 0, 0, 0, 30])
    nodes.append(helper.make_node('Pad', ['pad_in', 'pads_60_h'], ['pad_in_60_h']))
    nodes.append(helper.make_node('Pad', ['pad_in', 'pads_60_w'], ['pad_in_60_w']))
    
    # shift_indices
    make_const('c0_scalar', 0)
    make_const('c30_scalar', 30)
    make_const('c1_scalar', 1)
    nodes.append(helper.make_node('Range', ['c0_scalar', 'c30_scalar', 'c1_scalar'], ['range_30']))
    make_const('shape_30_1', [30, 1])
    make_const('shape_1_30', [1, 30])
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_30_1'], ['H_range']))
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_1_30'], ['R_range']))
    nodes.append(helper.make_node('Add', ['H_range', 'R_range'], ['shift_indices'])) # [30, 30]
    
    # V_h
    nodes.append(helper.make_node('Gather', ['pad_in_60_h', 'shift_indices'], ['V_h'], axis=1)) # [batch, 30, 30, 30]
    make_const('shape_exp', [-1, 1, 30, 30])
    nodes.append(helper.make_node('Reshape', ['pad_in', 'shape_exp'], ['pad_in_exp']))
    
    # Metrics H
    nodes.append(helper.make_node('Equal', ['V_h', 'pad_in_exp'], ['eq_h']))
    nodes.append(helper.make_node('Not', ['eq_h'], ['neq_h']))
    nodes.append(helper.make_node('Equal', ['V_h', 'c0_1d'], ['v_eq_0_h']))
    nodes.append(helper.make_node('Not', ['v_eq_0_h'], ['v_neq_0_h']))
    nodes.append(helper.make_node('Equal', ['pad_in_exp', 'c0_1d'], ['p_eq_0_h']))
    nodes.append(helper.make_node('Not', ['p_eq_0_h'], ['p_neq_0_h']))
    
    nodes.append(helper.make_node('And', ['neq_h', 'v_neq_0_h'], ['c1_h']))
    nodes.append(helper.make_node('And', ['c1_h', 'p_neq_0_h'], ['c_h_bool']))
    nodes.append(helper.make_node('Cast', ['c_h_bool'], ['c_h'], to=I64))
    
    nodes.append(helper.make_node('And', ['v_neq_0_h', 'p_neq_0_h'], ['o_h_bool']))
    nodes.append(helper.make_node('Cast', ['o_h_bool'], ['o_h'], to=I64))
    
    make_const('axes_23', [2, 3])
    nodes.append(helper.make_node('ReduceSum', ['c_h', 'axes_23'], ['c_count_h'], keepdims=0))
    nodes.append(helper.make_node('ReduceSum', ['o_h', 'axes_23'], ['o_count_h'], keepdims=0))
    
    nodes.append(helper.make_node('Equal', ['c_count_h', 'c0_1d'], ['h_no_conflict']))
    nodes.append(helper.make_node('Greater', ['o_count_h', 'c0_1d'], ['h_has_overlap']))
    nodes.append(helper.make_node('And', ['h_no_conflict', 'h_has_overlap'], ['valid_h_raw']))
    nodes.append(helper.make_node('Greater', ['range_30', 'c0_1d'], ['range_gt_0']))
    nodes.append(helper.make_node('And', ['valid_h_raw', 'range_gt_0'], ['valid_h']))
    
    nodes.append(helper.make_node('Cast', ['valid_h'], ['valid_h_i64'], to=I64))
    nodes.append(helper.make_node('ArgMax', ['valid_h_i64'], ['best_h_idx'], axis=1, keepdims=0))
    
    nodes.append(helper.make_node('Squeeze', ['H_dim', 'c0_1d'], ['H_dim_sq']))
    nodes.append(helper.make_node('Equal', ['best_h_idx', 'c0_1d'], ['best_h_is_0']))
    nodes.append(helper.make_node('Where', ['best_h_is_0', 'H_dim_sq', 'best_h_idx'], ['best_h']))
    
    # Metrics W
    nodes.append(helper.make_node('Gather', ['pad_in_60_w', 'shift_indices'], ['V_w_raw'], axis=2))
    nodes.append(helper.make_node('Transpose', ['V_w_raw'], ['V_w'], perm=[0, 2, 1, 3]))
    
    nodes.append(helper.make_node('Equal', ['V_w', 'pad_in_exp'], ['eq_w']))
    nodes.append(helper.make_node('Not', ['eq_w'], ['neq_w']))
    nodes.append(helper.make_node('Equal', ['V_w', 'c0_1d'], ['v_eq_0_w']))
    nodes.append(helper.make_node('Not', ['v_eq_0_w'], ['v_neq_0_w']))
    nodes.append(helper.make_node('Equal', ['pad_in_exp', 'c0_1d'], ['p_eq_0_w']))
    nodes.append(helper.make_node('Not', ['p_eq_0_w'], ['p_neq_0_w']))
    
    nodes.append(helper.make_node('And', ['neq_w', 'v_neq_0_w'], ['c1_w']))
    nodes.append(helper.make_node('And', ['c1_w', 'p_neq_0_w'], ['c_w_bool']))
    nodes.append(helper.make_node('Cast', ['c_w_bool'], ['c_w'], to=I64))
    
    nodes.append(helper.make_node('And', ['v_neq_0_w', 'p_neq_0_w'], ['o_w_bool']))
    nodes.append(helper.make_node('Cast', ['o_w_bool'], ['o_w'], to=I64))
    
    nodes.append(helper.make_node('ReduceSum', ['c_w', 'axes_23'], ['c_count_w'], keepdims=0))
    nodes.append(helper.make_node('ReduceSum', ['o_w', 'axes_23'], ['o_count_w'], keepdims=0))
    
    nodes.append(helper.make_node('Equal', ['c_count_w', 'c0_1d'], ['w_no_conflict']))
    nodes.append(helper.make_node('Greater', ['o_count_w', 'c0_1d'], ['w_has_overlap']))
    nodes.append(helper.make_node('And', ['w_no_conflict', 'w_has_overlap'], ['valid_w_raw']))
    nodes.append(helper.make_node('And', ['valid_w_raw', 'range_gt_0'], ['valid_w']))
    
    nodes.append(helper.make_node('Cast', ['valid_w'], ['valid_w_i64'], to=I64))
    nodes.append(helper.make_node('ArgMax', ['valid_w_i64'], ['best_w_idx'], axis=1, keepdims=0))
    
    nodes.append(helper.make_node('Squeeze', ['W_dim', 'c0_1d'], ['W_dim_sq']))
    nodes.append(helper.make_node('Equal', ['best_w_idx', 'c0_1d'], ['best_w_is_0']))
    nodes.append(helper.make_node('Where', ['best_w_is_0', 'W_dim_sq', 'best_w_idx'], ['best_w']))
    
    # 5D Coordinates
    make_const('shape_5d_h', [-1, 1, 1, 1, 1])
    nodes.append(helper.make_node('Reshape', ['best_h', 'shape_5d_h'], ['best_h_5d']))
    nodes.append(helper.make_node('Reshape', ['best_w', 'shape_5d_h'], ['best_w_5d']))
    
    make_const('shape_r_q', [1, 30, 1, 1, 1])
    make_const('shape_c_q', [1, 1, 30, 1, 1])
    make_const('shape_r_c', [1, 1, 1, 30, 1])
    make_const('shape_c_c', [1, 1, 1, 1, 30])
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_r_q'], ['R_query']))
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_c_q'], ['C_query']))
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_r_c'], ['R_cand']))
    nodes.append(helper.make_node('Reshape', ['range_30', 'shape_c_c'], ['C_cand']))
    
    nodes.append(helper.make_node('Mod', ['R_query', 'best_h_5d'], ['R_query_mod']))
    nodes.append(helper.make_node('Mod', ['R_cand', 'best_h_5d'], ['R_cand_mod']))
    nodes.append(helper.make_node('Equal', ['R_query_mod', 'R_cand_mod'], ['is_eq_r']))
    
    nodes.append(helper.make_node('Mod', ['C_query', 'best_w_5d'], ['C_query_mod']))
    nodes.append(helper.make_node('Mod', ['C_cand', 'best_w_5d'], ['C_cand_mod']))
    nodes.append(helper.make_node('Equal', ['C_query_mod', 'C_cand_mod'], ['is_eq_c']))
    
    nodes.append(helper.make_node('And', ['is_eq_r', 'is_eq_c'], ['is_eq']))
    
    make_const('shape_pad_in_5d', [-1, 1, 1, 30, 30])
    nodes.append(helper.make_node('Reshape', ['pad_in', 'shape_pad_in_5d'], ['pad_in_5d']))
    nodes.append(helper.make_node('Where', ['is_eq', 'pad_in_5d', 'c0_1d'], ['candidates']))
    
    nodes.append(helper.make_node('ReduceMax', ['candidates'], ['output_30x30'], axes=[3, 4], keepdims=0))
    
    nodes.append(helper.make_node('Concat', ['H_dim', 'W_dim'], ['out_ends'], axis=0))
    make_const('starts_0_0', [0, 0])
    make_const('axes_1_2', [1, 2])
    nodes.append(helper.make_node('Slice', ['output_30x30', 'starts_0_0', 'out_ends', 'axes_1_2'], ['output']))
    
    graph = helper.make_graph(nodes, 'task110_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task110_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task110.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task110.onnx')
    
    with open('task110.json', 'r') as f:
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
model = onnx.load("/project/repairs/task110.onnx")
