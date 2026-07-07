# Source: predicted/test_onnx_task123.py — ONNX graph construction code
# Verified model: repairs/task123.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    I64 = TensorProto.INT64
    
    input_info = helper.make_tensor_value_info('input', I64, ['batch', 'H', 'W'])
    output_info = helper.make_tensor_value_info('output', I64, ['batch', 'outH', 'outW'])
    
    nodes = []
    
    def make_const(name, val, dtype=I64):
        val_arr = np.array(val)
        nodes.append(helper.make_node('Constant', [], [name], value=helper.make_tensor(name+'_v', dtype, val_arr.shape, val_arr.flatten().tolist())))
    
    make_const('c0', 0)
    make_const('c1', 1)
    make_const('c2', 2)
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('axes_0', [0])
    
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_dim']))
    nodes.append(helper.make_node('Squeeze', ['H_dim', 'axes_0'], ['H']))
    nodes.append(helper.make_node('Squeeze', ['W_dim', 'axes_0'], ['W']))
    
    # Output size = 2*H x 2*W
    nodes.append(helper.make_node('Mul', ['H', 'c2'], ['outH']))
    nodes.append(helper.make_node('Mul', ['W', 'c2'], ['outW']))
    
    # Check if last row is all zeros: sum of last row
    nodes.append(helper.make_node('Sub', ['H', 'c1'], ['H_m1']))
    nodes.append(helper.make_node('Sub', ['W', 'c1'], ['W_m1']))
    # Get last row: input[:, H-1:H, :]  
    nodes.append(helper.make_node('Unsqueeze', ['H_m1', 'axes_0'], ['H_m1_1d']))
    nodes.append(helper.make_node('Unsqueeze', ['H', 'axes_0'], ['H_1d']))
    nodes.append(helper.make_node('Slice', ['input', 'H_m1_1d', 'H_1d', 'c1_1d'], ['last_row']))
    make_const('reduce_axes', [1, 2])
    nodes.append(helper.make_node('ReduceSum', ['last_row', 'reduce_axes'], ['last_row_sum'], keepdims=0))
    nodes.append(helper.make_node('Equal', ['last_row_sum', 'c0'], ['last_row_all_zeros']))
    nodes.append(helper.make_node('Where', ['last_row_all_zeros', 'H_m1', 'H'], ['pH']))
    nodes.append(helper.make_node('Where', ['last_row_all_zeros', 'W_m1', 'W'], ['pW']))
    
    # Indices for output
    nodes.append(helper.make_node('Range', ['c0', 'outH', 'c1'], ['r_out']))
    nodes.append(helper.make_node('Range', ['c0', 'outW', 'c1'], ['c_out']))
    
    # lev_r = r // pH, lev_c = c // pW
    # ri = r % pH, ci = c % pW
    nodes.append(helper.make_node('Div', ['r_out', 'pH'], ['lev_r']))  # [outH]
    nodes.append(helper.make_node('Div', ['c_out', 'pW'], ['lev_c']))  # [outW]
    nodes.append(helper.make_node('Mod', ['r_out', 'pH'], ['ri']))     # [outH]
    nodes.append(helper.make_node('Mod', ['c_out', 'pW'], ['ci']))     # [outW]
    
    # Reshape for broadcasting [outH] -> [1, outH, 1] and [outW] -> [1, 1, outW]
    make_const('shape_r', [1, -1, 1])
    make_const('shape_c', [1, 1, -1])
    nodes.append(helper.make_node('Reshape', ['lev_r', 'shape_r'], ['lev_r_col']))
    nodes.append(helper.make_node('Reshape', ['lev_c', 'shape_c'], ['lev_c_row']))
    nodes.append(helper.make_node('Reshape', ['ri', 'shape_r'], ['ri_col']))
    nodes.append(helper.make_node('Reshape', ['ci', 'shape_c'], ['ci_row']))
    
    # eff_pat_r = where(lev_r >= lev_c, ri, 0)
    # eff_pat_c = where(lev_c >= lev_r, ci, 0)
    nodes.append(helper.make_node('GreaterOrEqual', ['lev_r_col', 'lev_c_row'], ['lev_r_ge_c']))
    nodes.append(helper.make_node('GreaterOrEqual', ['lev_c_row', 'lev_r_col'], ['lev_c_ge_r']))
    
    nodes.append(helper.make_node('Where', ['lev_r_ge_c', 'ri_col', 'c0'], ['eff_r']))
    nodes.append(helper.make_node('Where', ['lev_c_ge_r', 'ci_row', 'c0'], ['eff_c']))
    
    # Build output shape
    nodes.append(helper.make_node('Unsqueeze', ['outH', 'axes_0'], ['outH_1d']))
    nodes.append(helper.make_node('Unsqueeze', ['outW', 'axes_0'], ['outW_1d']))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'outH_1d', 'outW_1d'], ['out_shape'], axis=0))
    
    # Expand eff_r and eff_c to full output shape
    nodes.append(helper.make_node('Expand', ['eff_r', 'out_shape'], ['eff_r_full']))
    nodes.append(helper.make_node('Expand', ['eff_c', 'out_shape'], ['eff_c_full']))
    
    # Compute flat index into pattern: eff_r * W + eff_c, then Gather from flattened pattern
    nodes.append(helper.make_node('Mul', ['eff_r', 'W'], ['eff_r_scaled']))
    nodes.append(helper.make_node('Add', ['eff_r_scaled', 'eff_c'], ['flat_idx']))  # [1, outH, outW]
    
    # Flatten pattern: [batch, H*W]
    nodes.append(helper.make_node('Mul', ['H', 'W'], ['HW']))
    nodes.append(helper.make_node('Unsqueeze', ['HW', 'axes_0'], ['HW_1d']))
    nodes.append(helper.make_node('Concat', ['batch_dim', 'HW_1d'], ['flat_pat_shape'], axis=0))
    nodes.append(helper.make_node('Reshape', ['input', 'flat_pat_shape'], ['pat_flat']))  # [batch, H*W]
    
    # Gather: for each batch, gather flat_idx from pat_flat
    # pat_flat [1, H*W], flat_idx [1, outH, outW] -> output [1, outH, outW]
    # Use GatherElements along axis=1 - need same shape: expand pat_flat to [1, outH*outW]
    # or just use Gather op
    nodes.append(helper.make_node('Squeeze', ['flat_idx', 'axes_0'], ['flat_idx_2d']))  # [outH, outW]
    nodes.append(helper.make_node('Squeeze', ['pat_flat', 'axes_0'], ['pat_flat_1d']))   # [H*W]
    nodes.append(helper.make_node('Gather', ['pat_flat_1d', 'flat_idx_2d'], ['out_2d']))  # [outH, outW]
    nodes.append(helper.make_node('Unsqueeze', ['out_2d', 'axes_0'], ['output']))          # [1, outH, outW]
    
    graph = helper.make_graph(nodes, 'task123_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task123_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task123.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task123.onnx')
    
    with open('task123.json', 'r') as f:
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
model = onnx.load("/project/repairs/task123.onnx")
