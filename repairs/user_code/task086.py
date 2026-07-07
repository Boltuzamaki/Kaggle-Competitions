# Source: predicted/test_onnx_task086.py — ONNX graph construction code
# Verified model: repairs/task086.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    F = TensorProto.FLOAT
    BOOL = TensorProto.BOOL
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'H', 'W'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c_max', [2147483647])
    make_const('c_min_1', [-1])
    make_const('axes_1', [1])
    make_const('axes_2', [2])
    
    make_const('pad_R', [0,0,0, 0,0,1])
    make_const('pad_L', [0,0,1, 0,0,0])
    make_const('pad_D', [0,0,0, 0,1,0])
    make_const('pad_U', [0,1,0, 0,0,0])
    
    # 3x3 ones kernel for Conv
    kernel_3x3 = np.ones((1, 1, 3, 3), dtype=np.float32)
    nodes.append(helper.make_node('Constant', [], ['k3x3'], value=helper.make_tensor('k3x3_v', F, kernel_3x3.shape, kernel_3x3.flatten().tolist())))
    make_const('c9', 9)
    
    # mask_in
    nodes.append(helper.make_node('Greater', ['input', 'c0'], ['mask_in_b']))
    nodes.append(helper.make_node('Cast', ['mask_in_b'], ['mask_in'], to=I64))
    
    # Conv2D requires input shape (N, C, H, W) where C=1
    nodes.append(helper.make_node('Cast', ['mask_in'], ['mask_in_f'], to=F))
    nodes.append(helper.make_node('Unsqueeze', ['mask_in_f', 'c1_1d'], ['mask_in_4d'])) # shape (batch, 1, H, W)
    nodes.append(helper.make_node('Conv', ['mask_in_4d', 'k3x3'], ['conv1_f'], pads=[1,1,1,1]))
    nodes.append(helper.make_node('Cast', ['conv1_f'], ['conv1'], to=I64))
    nodes.append(helper.make_node('Squeeze', ['conv1', 'c1_1d'], ['conv1_3d']))
    
    nodes.append(helper.make_node('Equal', ['conv1_3d', 'c9'], ['centers_b']))
    nodes.append(helper.make_node('Cast', ['centers_b'], ['mask_I'], to=I64))
    
    nodes.append(helper.make_node('Cast', ['mask_I'], ['mask_I_f'], to=F))
    nodes.append(helper.make_node('Unsqueeze', ['mask_I_f', 'c1_1d'], ['mask_I_4d']))
    nodes.append(helper.make_node('Conv', ['mask_I_4d', 'k3x3'], ['conv2_f'], pads=[1,1,1,1]))
    nodes.append(helper.make_node('Cast', ['conv2_f'], ['conv2'], to=I64))
    nodes.append(helper.make_node('Squeeze', ['conv2', 'c1_1d'], ['conv2_3d']))
    nodes.append(helper.make_node('Greater', ['conv2_3d', 'c0'], ['inner_3x3_b']))
    
    nodes.append(helper.make_node('Not', ['centers_b'], ['not_centers_b']))
    nodes.append(helper.make_node('And', ['inner_3x3_b', 'not_centers_b'], ['mask_O_b']))
    nodes.append(helper.make_node('Cast', ['mask_O_b'], ['mask_O'], to=I64))
    
    # Colors
    nodes.append(helper.make_node('Mul', ['input', 'mask_I'], ['I_pixels']))
    nodes.append(helper.make_node('ReduceMax', ['I_pixels'], ['I_color'], keepdims=0))
    nodes.append(helper.make_node('Mul', ['input', 'mask_O'], ['O_pixels']))
    nodes.append(helper.make_node('ReduceMax', ['O_pixels'], ['O_color'], keepdims=0))
    
    # Shifts
    def shift_node(name_in, dir_name, name_out):
        if dir_name == 'L':
            pad = 'pad_R'; starts = 'c1_1d'; ends = 'c_max'; axes = 'axes_2'
        elif dir_name == 'R':
            pad = 'pad_L'; starts = 'c0_1d'; ends = 'c_min_1'; axes = 'axes_2'
        elif dir_name == 'U':
            pad = 'pad_D'; starts = 'c1_1d'; ends = 'c_max'; axes = 'axes_1'
        elif dir_name == 'D':
            pad = 'pad_U'; starts = 'c0_1d'; ends = 'c_min_1'; axes = 'axes_1'
            
        nodes.append(helper.make_node('Pad', [name_in, pad], [name_out + '_pad']))
        nodes.append(helper.make_node('Slice', [name_out + '_pad', starts, ends, axes], [name_out]))

    shift_node('mask_I', 'L', 'mask_I_L')
    shift_node('mask_I', 'R', 'mask_I_R')
    shift_node('mask_I', 'U', 'mask_I_U')
    shift_node('mask_I', 'D', 'mask_I_D')
    
    nodes.append(helper.make_node('Cast', ['mask_I_U'], ['mask_I_U_b'], to=BOOL))
    nodes.append(helper.make_node('Cast', ['mask_I_D'], ['mask_I_D_b'], to=BOOL))
    nodes.append(helper.make_node('Cast', ['mask_I_L'], ['mask_I_L_b'], to=BOOL))
    nodes.append(helper.make_node('Cast', ['mask_I_R'], ['mask_I_R_b'], to=BOOL))
    
    nodes.append(helper.make_node('Or', ['centers_b', 'mask_I_U_b'], ['m_I_V_1']))
    nodes.append(helper.make_node('Or', ['m_I_V_1', 'mask_I_D_b'], ['mask_I_V_b']))
    nodes.append(helper.make_node('Cast', ['mask_I_V_b'], ['mask_I_V'], to=I64))
    
    nodes.append(helper.make_node('Or', ['centers_b', 'mask_I_L_b'], ['m_I_H_1']))
    nodes.append(helper.make_node('Or', ['m_I_H_1', 'mask_I_R_b'], ['mask_I_H_b']))
    nodes.append(helper.make_node('Cast', ['mask_I_H_b'], ['mask_I_H'], to=I64))
    
    def build_ext(dir_name, base_I, spread_axes):
        # dir_name: L, R, U, D
        # base_I: mask_I_V or mask_I_H
        shift_node(base_I, dir_name, f's_{dir_name}_base_I')
        nodes.append(helper.make_node('Cast', [f's_{dir_name}_base_I'], [f's_{dir_name}_base_I_b'], to=BOOL))
        nodes.append(helper.make_node('And', ['mask_O_b', f's_{dir_name}_base_I_b'], [f'mask_O_{dir_name}_b']))
        nodes.append(helper.make_node('Cast', [f'mask_O_{dir_name}_b'], [f'mask_O_{dir_name}'], to=I64))
        
        # D
        nodes.append(helper.make_node('Identity', ['mask_I'], [f'D_{dir_name}_0']))
        for i in range(30):
            shift_node(f'D_{dir_name}_{i}', dir_name, f's_{dir_name}_D_{i}')
            nodes.append(helper.make_node('Add', [f's_{dir_name}_D_{i}', 'c1'], [f's_{dir_name}_D_plus1_{i}']))
            nodes.append(helper.make_node('Where', ['centers_b', f's_{dir_name}_D_plus1_{i}', 'c0'], [f'D_{dir_name}_{i+1}']))
        
        # Spread D
        nodes.append(helper.make_node('Identity', [f'D_{dir_name}_30'], [f'D_{dir_name}_spread_0']))
        if spread_axes == 'V':
            shift_node(f'D_{dir_name}_30', 'U', f'D_{dir_name}_spread_U')
            shift_node(f'D_{dir_name}_30', 'D', f'D_{dir_name}_spread_D')
            nodes.append(helper.make_node('Max', [f'D_{dir_name}_30', f'D_{dir_name}_spread_U'], [f'D_{dir_name}_spread_1']))
            nodes.append(helper.make_node('Max', [f'D_{dir_name}_spread_1', f'D_{dir_name}_spread_D'], [f'D_{dir_name}_spread_final']))
        else:
            shift_node(f'D_{dir_name}_30', 'L', f'D_{dir_name}_spread_L')
            shift_node(f'D_{dir_name}_30', 'R', f'D_{dir_name}_spread_R')
            nodes.append(helper.make_node('Max', [f'D_{dir_name}_30', f'D_{dir_name}_spread_L'], [f'D_{dir_name}_spread_1']))
            nodes.append(helper.make_node('Max', [f'D_{dir_name}_spread_1', f'D_{dir_name}_spread_R'], [f'D_{dir_name}_spread_final']))
            
        # E
        shift_node(f'D_{dir_name}_spread_final', dir_name, f's_{dir_name}_D_spread')
        nodes.append(helper.make_node('Add', [f's_{dir_name}_D_spread', 'c1'], [f's_{dir_name}_D_spread_plus1']))
        nodes.append(helper.make_node('Where', [f'mask_O_{dir_name}_b', f's_{dir_name}_D_spread_plus1', 'c0'], [f'E_{dir_name}_0']))
        
        for i in range(30):
            shift_node(f'E_{dir_name}_{i}', dir_name, f's_{dir_name}_E_{i}')
            nodes.append(helper.make_node('Sub', [f's_{dir_name}_E_{i}', 'c1'], [f's_{dir_name}_E_sub1_{i}']))
            nodes.append(helper.make_node('Max', [f's_{dir_name}_E_sub1_{i}', 'c0'], [f's_{dir_name}_E_sub1_max_{i}']))
            nodes.append(helper.make_node('Greater', [f'E_{dir_name}_{i}', 'c0'], [f'E_{dir_name}_gt0_{i}']))
            nodes.append(helper.make_node('Where', [f'E_{dir_name}_gt0_{i}', f'E_{dir_name}_{i}', f's_{dir_name}_E_sub1_max_{i}'], [f'E_{dir_name}_{i+1}']))
            
        nodes.append(helper.make_node('Greater', [f'E_{dir_name}_30', 'c0'], [f'ext_{dir_name}_1']))
        nodes.append(helper.make_node('Not', [f'mask_O_{dir_name}_b'], [f'not_mask_O_{dir_name}']))
        nodes.append(helper.make_node('And', [f'ext_{dir_name}_1', f'not_mask_O_{dir_name}'], [f'ext_{dir_name}']))
    
    build_ext('L', 'mask_I_V', 'V')
    build_ext('R', 'mask_I_V', 'V')
    build_ext('U', 'mask_I_H', 'H')
    build_ext('D', 'mask_I_H', 'H')
    
    nodes.append(helper.make_node('Or', ['ext_L', 'ext_R'], ['ext_LR']))
    nodes.append(helper.make_node('Or', ['ext_U', 'ext_D'], ['ext_UD']))
    nodes.append(helper.make_node('Or', ['ext_LR', 'ext_UD'], ['ext_all']))
    
    nodes.append(helper.make_node('Shape', ['input'], ['shape']))
    nodes.append(helper.make_node('ConstantOfShape', ['shape'], ['zeros'], value=helper.make_tensor('z_v', I64, [1], [0])))
    
    nodes.append(helper.make_node('Where', ['ext_all', 'O_color', 'zeros'], ['out1']))
    nodes.append(helper.make_node('Where', ['centers_b', 'O_color', 'out1'], ['out2']))
    nodes.append(helper.make_node('Where', ['mask_O_b', 'I_color', 'out2'], ['output']))
    
    graph = helper.make_graph(nodes, 'task086_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task086_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task086.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task086.onnx')
    
    with open('../data/task086.json', 'r') as f:
        task = json.load(f)
        
    for split in ['train', 'test']:
        for i, ex in enumerate(task[split]):
            inp = np.array(ex['input'], dtype=np.int64)
            if len(inp.shape) == 2:
                inp = inp[np.newaxis, ...]
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
                    sys.exit(1)


# Build the model (the function saves it internally, so we load the result)
create_onnx_model()
import glob
model = onnx.load("/project/repairs/task086.onnx")
