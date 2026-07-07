# Source: predicted/test_onnx_task005.py — ONNX graph construction code
# Verified model: repairs/task005.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    
    def K(name, arr, dtype=np.int64): 
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
        
    inits = [
        K('c1_s', [1]), K('c1_e', [10]), K('ax_c', [1]),
        K('ax_0', [0]), K('ax_1', [1]), K('ax_2', [2]), K('ax_3', [3]),
        K('ax_2_3', [2, 3]),
        K('c1_1d', [1]), K('c0_1d', [0]), K('c2_1d', [2]), K('c3_1d', [3]),
        K('c_neg1_1d', [-1]),
        K('c30_1d', [30]), K('c60_1d', [60]),
        K('c29_1d', [29]),
        K('c29_2d', [[29]]),
        K('arange30', np.arange(30)),
        K('rev_arange30', np.arange(29, -1, -1)),
        K('arange_r', np.arange(30).reshape(1, 1, 30, 1)),
        K('arange_c', np.arange(30).reshape(1, 1, 1, 30)),
        K('conv_weights', np.ones((9, 1, 3, 3), dtype=np.float32), dtype=np.float32),
        K('depth10', 10), K('depth9', 9),
        K('oh_vals', [0.0, 1.0], dtype=np.float32),
        K('c0_f', [0.0], dtype=np.float32), K('c1_f', [1.0], dtype=np.float32), K('c2_f', [2.0], dtype=np.float32),
        K('c0_4d_zero', np.zeros((1, 1, 30, 30), dtype=np.float32), dtype=np.float32),
        K('shape_1_9_1_1', [1, 9, 1, 1]),
        K('shape_900', [900]),
        K('shape_1_1_30_30', [1, 1, 30, 30])
    ]
    
    nodes = [
        # Slice channels 1-9
        helper.make_node('Slice', ['input', 'c1_s', 'c1_e', 'ax_c'], ['c1_9']),
        
        # Conv score
        helper.make_node('Conv', ['c1_9', 'conv_weights'], ['conv'], group=9, pads=[1, 1, 1, 1]),
        helper.make_node('Mul', ['conv', 'c1_9'], ['conv_masked']),
        helper.make_node('ReduceSum', ['conv_masked', 'ax_2_3'], ['conv_scores'], keepdims=0),
        
        # ArgMax for template
        helper.make_node('ArgMax', ['conv_scores'], ['template_channel'], axis=1, keepdims=0),
        helper.make_node('OneHot', ['template_channel', 'depth9', 'oh_vals'], ['template_oh']),
        helper.make_node('Reshape', ['template_oh', 'shape_1_9_1_1'], ['template_oh_4d']),
        
        # Masking
        helper.make_node('Mul', ['c1_9', 'template_oh_4d'], ['template_mask_all_channels']),
        helper.make_node('ReduceMax', ['template_mask_all_channels'], ['template_mask'], axes=[1], keepdims=1),
        helper.make_node('Sub', ['c1_9', 'template_mask_all_channels'], ['seed_masks']),
        
        # Cast to INT64 for bbox math
        helper.make_node('Cast', ['template_mask'], ['template_mask_i'], to=I64),
        
        # Bounding box of template
        helper.make_node('ReduceMax', ['template_mask_i'], ['t_r_proj'], axes=[3], keepdims=0),
        helper.make_node('Mul', ['t_r_proj', 'arange30'], ['t_r_m1']),
        helper.make_node('ReduceMax', ['t_r_m1'], ['t_r_max_2d'], axes=[2], keepdims=0),
        helper.make_node('Mul', ['t_r_proj', 'rev_arange30'], ['t_r_m2']),
        helper.make_node('ReduceMax', ['t_r_m2'], ['t_r_min_inv_2d'], axes=[2], keepdims=0),
        helper.make_node('Sub', ['c29_2d', 't_r_min_inv_2d'], ['t_r_min_2d']),
        
        helper.make_node('ReduceMax', ['template_mask_i'], ['t_c_proj'], axes=[2], keepdims=0),
        helper.make_node('Mul', ['t_c_proj', 'arange30'], ['t_c_m1']),
        helper.make_node('ReduceMax', ['t_c_m1'], ['t_c_max_2d'], axes=[2], keepdims=0),
        helper.make_node('Mul', ['t_c_proj', 'rev_arange30'], ['t_c_m2']),
        helper.make_node('ReduceMax', ['t_c_m2'], ['t_c_min_inv_2d'], axes=[2], keepdims=0),
        helper.make_node('Sub', ['c29_2d', 't_c_min_inv_2d'], ['t_c_min_2d']),
        
        helper.make_node('Squeeze', ['t_r_max_2d', 'ax_0'], ['t_r_max_1d']),
        helper.make_node('Squeeze', ['t_r_min_2d', 'ax_0'], ['t_r_min_1d']),
        helper.make_node('Squeeze', ['t_c_max_2d', 'ax_0'], ['t_c_max_1d']),
        helper.make_node('Squeeze', ['t_c_min_2d', 'ax_0'], ['t_c_min_1d']),
        
        helper.make_node('Sub', ['t_r_max_1d', 't_r_min_1d'], ['h_minus_1']),
        helper.make_node('Add', ['h_minus_1', 'c1_1d'], ['h_1d']),
        helper.make_node('Add', ['h_1d', 'c1_1d'], ['h_plus_1']),
        helper.make_node('Mul', ['h_plus_1', 'c_neg1_1d'], ['neg_h_plus_1']),
        
        helper.make_node('Sub', ['t_c_max_1d', 't_c_min_1d'], ['w_minus_1']),
        helper.make_node('Add', ['w_minus_1', 'c1_1d'], ['w_1d']),
        helper.make_node('Add', ['w_1d', 'c1_1d'], ['w_plus_1']),
        helper.make_node('Mul', ['w_plus_1', 'c_neg1_1d'], ['neg_w_plus_1']),
        
        # Priority Masks (now exact quadrants + pure sides)
        helper.make_node('Less', ['arange_r', 't_r_min_1d'], ['m_up_b']),
        helper.make_node('Greater', ['arange_r', 't_r_max_1d'], ['m_down_b']),
        helper.make_node('Less', ['arange_c', 't_c_min_1d'], ['m_left_b']),
        helper.make_node('Greater', ['arange_c', 't_c_max_1d'], ['m_right_b']),

        helper.make_node('Cast', ['m_up_b'], ['m_up_raw'], to=F),
        helper.make_node('Cast', ['m_down_b'], ['m_down_raw'], to=F),
        helper.make_node('Cast', ['m_left_b'], ['m_left_raw'], to=F),
        helper.make_node('Cast', ['m_right_b'], ['m_right_raw'], to=F),

        helper.make_node('Sub', ['c1_f', 'm_up_raw'], ['not_up']),
        helper.make_node('Sub', ['c1_f', 'm_down_raw'], ['not_down']),
        helper.make_node('Sub', ['c1_f', 'm_left_raw'], ['not_left']),
        helper.make_node('Sub', ['c1_f', 'm_right_raw'], ['not_right']),

        # 8 directions
        helper.make_node('Mul', ['m_up_raw', 'm_left_raw'], ['m_up_left']),
        helper.make_node('Mul', ['m_up_raw', 'm_right_raw'], ['m_up_right']),
        helper.make_node('Mul', ['m_down_raw', 'm_left_raw'], ['m_down_left']),
        helper.make_node('Mul', ['m_down_raw', 'm_right_raw'], ['m_down_right']),
        
        helper.make_node('Mul', ['m_up_raw', 'not_left'], ['m_up_not_left']),
        helper.make_node('Mul', ['m_up_not_left', 'not_right'], ['m_up']),
        
        helper.make_node('Mul', ['m_down_raw', 'not_left'], ['m_down_not_left']),
        helper.make_node('Mul', ['m_down_not_left', 'not_right'], ['m_down']),
        
        helper.make_node('Mul', ['m_left_raw', 'not_up'], ['m_left_not_up']),
        helper.make_node('Mul', ['m_left_not_up', 'not_down'], ['m_left']),
        
        helper.make_node('Mul', ['m_right_raw', 'not_up'], ['m_right_not_up']),
        helper.make_node('Mul', ['m_right_not_up', 'not_down'], ['m_right']),
        
        # Flatten template for gather
        helper.make_node('Reshape', ['template_mask', 'shape_900'], ['template_mask_flat']),
    ]
    
    def make_tiles(name, dy_mul, dx_mul):
        for i in range(1, 12):
            dy = f'dy{i}_{name}'
            dx = f'dx{i}_{name}'
            ci_name = f'c{i}_1d_{name}'
            ci_dx_name = f'c{i}_1d_dx_{name}'
            inits.append(K(ci_name, [i]))
            inits.append(K(ci_dx_name, [i]))
            nodes.extend([
                helper.make_node('Mul', [dy_mul, ci_name], [dy]),
                helper.make_node('Mul', [dx_mul, ci_dx_name], [dx]),
                
                # r_grid is 30x1, c_grid is 1x30
                helper.make_node('Sub', ['arange_r', dy], [f'src_r_{i}_{name}']),
                helper.make_node('Sub', ['arange_c', dx], [f'src_c_{i}_{name}']),
                
                helper.make_node('Less', [f'src_r_{i}_{name}', 'c0_1d'], [f'r_lt_0_{i}_{name}']),
                helper.make_node('GreaterOrEqual', [f'src_r_{i}_{name}', 'c30_1d'], [f'r_ge_30_{i}_{name}']),
                helper.make_node('Or', [f'r_lt_0_{i}_{name}', f'r_ge_30_{i}_{name}'], [f'r_out_{i}_{name}']),
                helper.make_node('Not', [f'r_out_{i}_{name}'], [f'r_valid_{i}_{name}']),
                
                helper.make_node('Less', [f'src_c_{i}_{name}', 'c0_1d'], [f'c_lt_0_{i}_{name}']),
                helper.make_node('GreaterOrEqual', [f'src_c_{i}_{name}', 'c30_1d'], [f'c_ge_30_{i}_{name}']),
                helper.make_node('Or', [f'c_lt_0_{i}_{name}', f'c_ge_30_{i}_{name}'], [f'c_out_{i}_{name}']),
                helper.make_node('Not', [f'c_out_{i}_{name}'], [f'c_valid_{i}_{name}']),
                
                helper.make_node('And', [f'r_valid_{i}_{name}', f'c_valid_{i}_{name}'], [f'valid_{i}_{name}']),
                helper.make_node('Cast', [f'valid_{i}_{name}'], [f'valid_f_{i}_{name}'], to=F),
                
                helper.make_node('Clip', [f'src_r_{i}_{name}', 'c0_1d', 'c29_1d'], [f'safe_r_{i}_{name}']),
                helper.make_node('Clip', [f'src_c_{i}_{name}', 'c0_1d', 'c29_1d'], [f'safe_c_{i}_{name}']),
                
                helper.make_node('Mul', [f'safe_r_{i}_{name}', 'c30_1d'], [f'safe_r_30_{i}_{name}']),
                helper.make_node('Add', [f'safe_r_30_{i}_{name}', f'safe_c_{i}_{name}'], [f'flat_idx_{i}_{name}']),
                
                helper.make_node('Gather', ['template_mask_flat', f'flat_idx_{i}_{name}'], [f'gathered_{i}_{name}']),
                helper.make_node('Mul', [f'gathered_{i}_{name}', f'valid_f_{i}_{name}'], [f'tile_raw_{i}_{name}']),
                helper.make_node('Reshape', [f'tile_raw_{i}_{name}', 'shape_1_1_30_30'], [f'tile{i}_{name}'])
            ])
            
        sum_nodes = []
        for i in range(1, 12):
            if i == 1:
                sum_nodes.append(f'tile1_{name}')
            else:
                nodes.append(helper.make_node('Add', [sum_nodes[-1], f'tile{i}_{name}'], [f'tsum{i-1}_{name}']))
                sum_nodes.append(f'tsum{i-1}_{name}')
        nodes.append(helper.make_node('Clip', [sum_nodes[-1], 'c0_f', 'c1_f'], [f'tiles_{name}']))

    make_tiles('up', 'neg_h_plus_1', 'c0_1d')
    make_tiles('down', 'h_plus_1', 'c0_1d')
    make_tiles('left', 'c0_1d', 'neg_w_plus_1')
    make_tiles('right', 'c0_1d', 'w_plus_1')
    
    make_tiles('up_left', 'neg_h_plus_1', 'neg_w_plus_1')
    make_tiles('up_right', 'neg_h_plus_1', 'w_plus_1')
    make_tiles('down_left', 'h_plus_1', 'neg_w_plus_1')
    make_tiles('down_right', 'h_plus_1', 'w_plus_1')
    
    dirs = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']
    
    out_channels = []
    for ch in range(9):
        ch_s_name = f'ch_s_{ch}'
        ch_e_name = f'ch_e_{ch}'
        inits.extend([
            K(ch_s_name, [ch]),
            K(ch_e_name, [ch+1])
        ])
        
        nodes.extend([
            helper.make_node('Slice', ['seed_masks', ch_s_name, ch_e_name, 'ax_1'], [f'seed_{ch}']),
        ])
        
        for d in dirs:
            nodes.extend([
                helper.make_node('Mul', [f'seed_{ch}', f'm_{d}'], [f'seed_{d}_{ch}']),
                helper.make_node('ReduceMax', [f'seed_{d}_{ch}'], [f'has_{d}_{ch}'], keepdims=1),
                helper.make_node('Mul', [f'tiles_{d}', f'has_{d}_{ch}'], [f't_{d}_{ch}']),
            ])
            
        nodes.extend([
            helper.make_node('Add', [f't_up_{ch}', f't_down_{ch}'], [f't_ud_{ch}']),
            helper.make_node('Add', [f't_left_{ch}', f't_right_{ch}'], [f't_lr_{ch}']),
            helper.make_node('Add', [f't_up_left_{ch}', f't_up_right_{ch}'], [f't_ul_ur_{ch}']),
            helper.make_node('Add', [f't_down_left_{ch}', f't_down_right_{ch}'], [f't_dl_dr_{ch}']),
            
            helper.make_node('Add', [f't_ud_{ch}', f't_lr_{ch}'], [f't_ud_lr_{ch}']),
            helper.make_node('Add', [f't_ul_ur_{ch}', f't_dl_dr_{ch}'], [f't_diags_{ch}']),
            helper.make_node('Add', [f't_ud_lr_{ch}', f't_diags_{ch}'], [f'out_ch_{ch}'])
        ])
        out_channels.append(f'out_ch_{ch}')
        
    nodes.extend([
        helper.make_node('Concat', out_channels, ['out_c1_9'], axis=1),
        
        # Mask out anything outside the valid grid
        helper.make_node('ReduceSum', ['input', 'ax_1'], ['valid_grid_mask'], keepdims=1),
        helper.make_node('Mul', ['out_c1_9', 'valid_grid_mask'], ['out_c1_9_masked']),
        
        helper.make_node('Concat', ['c0_4d_zero', 'out_c1_9_masked'], ['new_additions'], axis=1),
        
        helper.make_node('Mul', ['new_additions', 'c2_f'], ['weighted_additions']),
        helper.make_node('Add', ['input', 'weighted_additions'], ['scores']),
        
        helper.make_node('ArgMax', ['scores'], ['pred'], axis=1, keepdims=0),
        helper.make_node('OneHot', ['pred', 'depth10', 'oh_vals'], ['output_unmasked'], axis=1),
        helper.make_node('Mul', ['output_unmasked', 'valid_grid_mask'], ['output']),
    ])
    
    graph = helper.make_graph(nodes, 'task005', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

def check_task():
    with open('task005.json', 'r') as f:
        task = json.load(f)
        
    model = create_model()
    onnx.save(model, 'task005.onnx')
    session = onnxruntime.InferenceSession('task005.onnx')
    
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
            
            pred_padded = np.argmax(res[0], axis=0)
            pred = pred_padded[:inp.shape[0], :inp.shape[1]]
            
            if np.array_equal(pred, out):
                print(f"{split} {i}: ONNX MATCH")
            else:
                print(f"{split} {i}: ONNX FAIL")


# Build the model
model = create_model()
