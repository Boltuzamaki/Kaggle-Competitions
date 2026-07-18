# Source: predicted/test_onnx_task076.py — ONNX graph construction code
# Verified model: repairs/task076.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    B = TensorProto.BOOL
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, ['batch', 10, 30, 30])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    inits = [
        K('c0_i64', [0]), K('c1_i64', [1]), K('c2_i64', [2]), K('c3_i64', [3]), K('c4_i64', [4]),
        K('c0_f', [0.0], dtype=np.float32), K('c05_f', [0.5], dtype=np.float32),
        K('pads_3', [0, 0, 1, 1, 0, 0, 1, 1]),
        K('pads_15', [0, 0, 15, 15, 0, 0, 15, 15]),
        K('ax_2_3', [2, 3]), K('ax_0', [0]), K('ax_2_1d', [2]),
        K('c_depth_10', [10]), K('c_oh_vals', [0.0, 1.0], dtype=np.float32),
        K('rev_indices', list(range(30, -1, -1))),
        K('slice_15_45', [15, 15]), K('slice_15_45_end', [45, 45]),
    ]
    
    nodes = []
    
    nodes.append(helper.make_node('ArgMax', ['input'], ['argmax'], axis=1, keepdims=1))
    
    # is_gt0
    nodes.append(helper.make_node('Greater', ['argmax', K('c0_i64', [0]).name], ['is_gt0_bool']))
    nodes.append(helper.make_node('Cast', ['is_gt0_bool'], ['is_gt0'], to=F))
    
    for val in [1, 2, 3, 4]:
        nodes.append(helper.make_node('Equal', ['argmax', f'c{val}_i64'], [f'is_{val}_bool']))
        nodes.append(helper.make_node('Cast', [f'is_{val}_bool'], [f'is_{val}'], to=F))
        
    # mask = (argmax == 1) | (argmax == 3)
    nodes.append(helper.make_node('Or', ['is_1_bool', 'is_3_bool'], ['is_13_bool']))
    nodes.append(helper.make_node('Cast', ['is_13_bool'], ['mask_0'], to=F))
    
    curr_mask = 'mask_0'
    for i in range(15):
        padded = f'padded_{i}'
        nodes.append(helper.make_node('Pad', [curr_mask, 'pads_3', 'c0_f'], [padded]))
        pooled = f'pooled_{i}'
        nodes.append(helper.make_node('MaxPool', [padded], [pooled], kernel_shape=[3, 3]))
        next_mask = f'mask_{i+1}'
        nodes.append(helper.make_node('Mul', [pooled, 'is_gt0'], [next_mask]))
        curr_mask = next_mask
        
    nodes.append(helper.make_node('Greater', [curr_mask, 'c05_f'], ['fs_mask_bool']))
    nodes.append(helper.make_node('Not', ['fs_mask_bool'], ['not_fs_mask_bool']))
    nodes.append(helper.make_node('Cast', ['fs_mask_bool'], ['fs_mask'], to=F))
    nodes.append(helper.make_node('Cast', ['not_fs_mask_bool'], ['not_fs_mask'], to=F))
    
    for val in [1, 2, 3, 4]:
        nodes.append(helper.make_node('Mul', [f'is_{val}', 'fs_mask'], [f'F_{val}']))
        nodes.append(helper.make_node('Mul', [f'is_{val}', 'not_fs_mask'], [f'target_{val}']))
        
    nodes.append(helper.make_node('NonZero', ['F_2'], ['coords_2']))
    # coords_2 is [4, 1]. We want r_0 (index 2) and c_0 (index 3)
    inits.extend([K('idx_2', [2]), K('idx_3', [3]), K('idx_4', [4])])
    nodes.append(helper.make_node('Slice', ['coords_2', 'idx_2', 'idx_3', 'ax_0'], ['r_0_2d']))
    nodes.append(helper.make_node('Slice', ['coords_2', 'idx_3', 'idx_4', 'ax_0'], ['c_0_2d']))
    nodes.append(helper.make_node('Squeeze', ['r_0_2d', 'ax_0'], ['r_0']))
    nodes.append(helper.make_node('Squeeze', ['c_0_2d', 'ax_0'], ['c_0']))
    
    # Calculate ends: r_0 + 31, c_0 + 31
    inits.append(K('c31_i64', [31]))
    nodes.append(helper.make_node('Add', ['r_0', 'c31_i64'], ['r_0_end']))
    nodes.append(helper.make_node('Add', ['c_0', 'c31_i64'], ['c_0_end']))
    nodes.append(helper.make_node('Concat', ['r_0', 'c_0'], ['starts_ext'], axis=0))
    nodes.append(helper.make_node('Concat', ['r_0_end', 'c_0_end'], ['ends_ext'], axis=0))
    
    for val in [1, 3, 4]:
        nodes.append(helper.make_node('Pad', [f'F_{val}', 'pads_15', 'c0_f'], [f'padded_F_{val}']))
        nodes.append(helper.make_node('Slice', [f'padded_F_{val}', 'starts_ext', 'ends_ext', 'ax_2_3'], [f'C_{val}']))
        
    nodes.append(helper.make_node('Pad', ['target_4', 'pads_15', 'c0_f'], ['target_4_padded']))
    nodes.append(helper.make_node('ReduceSum', ['C_4'], ['sum_4'], keepdims=0))
    nodes.append(helper.make_node('Sub', ['sum_4', 'c05_f'], ['thresh']))
    
    total_1_terms = []
    total_3_terms = []
    
    for flip_ud in [False, True]:
        for flip_lr in [False, True]:
            for trans in [False, True]:
                suffix = f'_{int(flip_ud)}_{int(flip_lr)}_{int(trans)}'
                
                T_4, T_1, T_3 = 'C_4', 'C_1', 'C_3'
                
                if flip_ud:
                    for v, T in [(4, T_4), (1, T_1), (3, T_3)]:
                        new_T = f'T{v}_ud{suffix}'
                        nodes.append(helper.make_node('Gather', [T, 'rev_indices'], [new_T], axis=2))
                        if v == 4: T_4 = new_T
                        if v == 1: T_1 = new_T
                        if v == 3: T_3 = new_T
                        
                if flip_lr:
                    for v, T in [(4, T_4), (1, T_1), (3, T_3)]:
                        new_T = f'T{v}_lr{suffix}'
                        nodes.append(helper.make_node('Gather', [T, 'rev_indices'], [new_T], axis=3))
                        if v == 4: T_4 = new_T
                        if v == 1: T_1 = new_T
                        if v == 3: T_3 = new_T
                        
                if trans:
                    for v, T in [(4, T_4), (1, T_1), (3, T_3)]:
                        new_T = f'T{v}_tr{suffix}'
                        nodes.append(helper.make_node('Transpose', [T], [new_T], perm=[0, 1, 3, 2]))
                        if v == 4: T_4 = new_T
                        if v == 1: T_1 = new_T
                        if v == 3: T_3 = new_T
                        
                overlap = f'overlap{suffix}'
                nodes.append(helper.make_node('Conv', ['target_4_padded', T_4], [overlap]))
                
                gt = f'gt{suffix}'
                nodes.append(helper.make_node('Greater', [overlap, 'thresh'], [gt]))
                
                target2_gt = f'target2_gt{suffix}'
                nodes.append(helper.make_node('Greater', ['target_2', 'c05_f'], [target2_gt]))
                
                match_bool = f'match_bool{suffix}'
                nodes.append(helper.make_node('And', [gt, target2_gt], [match_bool]))
                
                match_f = f'match_f{suffix}'
                nodes.append(helper.make_node('Cast', [match_bool], [match_f], to=F))
                
                out1_full = f'out1_full{suffix}'
                out3_full = f'out3_full{suffix}'
                
                nodes.append(helper.make_node('ConvTranspose', [match_f, T_1], [out1_full]))
                nodes.append(helper.make_node('ConvTranspose', [match_f, T_3], [out3_full]))
                
                out1 = f'out1{suffix}'
                out3 = f'out3{suffix}'
                
                nodes.append(helper.make_node('Slice', [out1_full, 'slice_15_45', 'slice_15_45_end', 'ax_2_3'], [out1]))
                nodes.append(helper.make_node('Slice', [out3_full, 'slice_15_45', 'slice_15_45_end', 'ax_2_3'], [out3]))
                
                total_1_terms.append(out1)
                total_3_terms.append(out3)
                
    nodes.append(helper.make_node('Sum', total_1_terms, ['total_1']))
    nodes.append(helper.make_node('Sum', total_3_terms, ['total_3']))
    
    nodes.append(helper.make_node('Greater', ['total_1', 'c05_f'], ['is_1_out']))
    nodes.append(helper.make_node('Greater', ['total_3', 'c05_f'], ['is_3_out']))
    
    nodes.append(helper.make_node('Where', ['is_3_out', 'c3_i64', 'argmax'], ['pred_1']))
    nodes.append(helper.make_node('Where', ['is_1_out', 'c1_i64', 'pred_1'], ['pred_argmax']))
    
    nodes.append(helper.make_node('OneHot', ['pred_argmax', 'c_depth_10', 'c_oh_vals'], ['pred_oh'], axis=-1))
    nodes.append(helper.make_node('Transpose', ['pred_oh'], ['pred_trans'], perm=[0, 4, 1, 2, 3]))
    nodes.append(helper.make_node('Squeeze', ['pred_trans', 'ax_2_1d'], ['output']))
    
    graph = helper.make_graph(nodes, 'task076', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

def check_task():
    with open('task076.json', 'r') as f:
        task = json.load(f)
        
    model = create_model()
    onnx.save(model, 'task076.onnx')
    session = onnxruntime.InferenceSession('task076.onnx')
    
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array(ex['input'], dtype=np.int64)
            out = np.array(ex['output'], dtype=np.int64)
            
            padded_inp = np.zeros((30, 30), dtype=np.int64)
            padded_inp[:inp.shape[0], :inp.shape[1]] = inp
            
            oh_inp = np.zeros((1, 10, 30, 30), dtype=np.float32)
            for r in range(30):
                for c in range(30):
                    oh_inp[0, padded_inp[r, c], r, c] = 1.0
                    
            res = session.run(['output'], {'input': oh_inp})[0]
            pred = np.argmax(res[0, :, :inp.shape[0], :inp.shape[1]], axis=0)
            
            if np.array_equal(pred, out):
                print(f"{split} {i}: ONNX MATCH")
            else:
                print(f"{split} {i}: ONNX FAIL")
                print("Expected:\n", out)
                print("Pred:\n", pred)


# Build the model
model = create_model()
