# Source: predicted/test_onnx_task125.py — ONNX graph construction code
# Verified model: repairs/task125.onnx
import json
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_onnx_model():
    """
    Task 125: 8=background. Non-8 regions = colored rectangles.
    Output:
    - 3-border placed AROUND (1 cell outside) each non-8 region
    - 4s placed at 8-cells INSIDE non-8 regions (enclosed 8-cells)
    - Original non-8 cells remain
    
    For 8-cells: "interior" means there's a non-8 cell to the left, right, above, AND below
    in the SAME row/column (not necessarily adjacent, but within the grid).
    
    Strategy:
    - For each 8-cell, check if scanning left hits non-8 AND scanning right hits non-8
      AND scanning up hits non-8 AND scanning down hits non-8
    - Use large Conv kernels (row, column) to detect "has non-8 to the left/right/up/down"
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
    make_const('c1', 1)
    make_const('c3', 3)
    make_const('c4', 4)
    make_const('c8', 8)
    make_const('c0_1d', [0])
    make_const('c1_1d', [1])
    make_const('c2_1d', [2])
    make_const('c3_1d', [3])
    make_const('axes_0', [0])
    
    nodes.append(helper.make_node('Shape', ['input'], ['in_shape']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c0_1d', 'c1_1d', 'axes_0'], ['batch_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c1_1d', 'c2_1d', 'axes_0'], ['H_dim']))
    nodes.append(helper.make_node('Slice', ['in_shape', 'c2_1d', 'c3_1d', 'axes_0'], ['W_dim']))
    
    # mask_non8: is not background (8)
    nodes.append(helper.make_node('Equal', ['input', 'c8'], ['mask8_bool']))
    nodes.append(helper.make_node('Not', ['mask8_bool'], ['mask_non8_bool']))
    
    nodes.append(helper.make_node('Cast', ['mask_non8_bool'], ['mask_non8_f32'], to=int(F32)))
    
    # 4D reshape for Conv
    nodes.append(helper.make_node('Concat', ['batch_dim', 'c1_1d', 'H_dim', 'W_dim'], ['shape_4d'], axis=0))
    nodes.append(helper.make_node('Reshape', ['mask_non8_f32', 'shape_4d'], ['mask_4d']))
    
    # 1. Border detection: cells that ARE 8 but adjacent to non-8 → 3
    k3 = np.ones((1, 1, 3, 3), dtype=np.float32)
    make_const('k3', k3, dtype=F32)
    nodes.append(helper.make_node('Conv', ['mask_4d', 'k3'], ['dilated_4d'], pads=[1,1,1,1]))
    nodes.append(helper.make_node('Reshape', ['dilated_4d', 'in_shape'], ['dilated']))
    
    make_const('c0_5_f', 0.5, dtype=F32)
    nodes.append(helper.make_node('Cast', ['dilated'], ['dilated_f32'], to=int(F32)))
    nodes.append(helper.make_node('Greater', ['dilated_f32', 'c0_5_f'], ['is_near_non8']))
    nodes.append(helper.make_node('And', ['mask8_bool', 'is_near_non8'], ['is_3']))
    
    # 2. Interior 8-cell detection: 8-cell with non-8 in all 4 "ray" directions
    # Use large horizontal kernel to detect if there's non-8 to the LEFT of this cell
    # and similarly for right, up, down using column kernels
    
    # Max size: assume <= 30x30 grid; use size-30 kernels pointing in each direction
    # For "has non-8 to the left": use causal conv in col direction (left half all 1s)
    # For any grid up to 30 wide:
    max_size = 30
    
    # Left kernel: horizontal kernel [1, 1, 1, ..., 0] (non-8 to the left)
    # We want: for position (r, c), is there non-8 at (r, c') for any c' < c?
    # Compute cumsum of non-8 from left. If > 0 at position, then yes.
    # For ONNX: use large triangular kernels or cumulative sum.
    
    # Actually simpler: use horizontal kernel of all 1s with asymmetric padding
    # Full row sum of non-8: ReduceSum along cols
    # But we need per-position "has non-8 to left/right/up/down"
    
    # Approximate: a cell is "interior 8" if the sum of non-8 in its row > 0
    # AND sum of non-8 in its col > 0 (i.e., there's non-8 both horizontally and vertically)
    # This is a necessary but not sufficient condition.
    # Better: check that there's non-8 strictly to the left AND right in same row
    # AND non-8 strictly above AND below in same column.
    
    # Use 1D kernels to scan each direction:
    # Row scan: left kernel = [1]*N + [0] with pad on right = N to get "left sum"
    # Col scan: up kernel = [1]*N + [0] col with pad on bottom = N to get "up sum"
    
    N = max_size
    k_left = np.zeros((1, 1, 1, N+1), dtype=np.float32)
    k_left[0, 0, 0, :N] = 1.0  # N ones then 0 (center at position N)
    make_const('k_left', k_left, dtype=F32)
    
    k_right = np.zeros((1, 1, 1, N+1), dtype=np.float32)
    k_right[0, 0, 0, 1:] = 1.0  # 0 then N ones (center at position 0)
    make_const('k_right', k_right, dtype=F32)
    
    k_up = np.zeros((1, 1, N+1, 1), dtype=np.float32)
    k_up[0, 0, :N, 0] = 1.0
    make_const('k_up', k_up, dtype=F32)
    
    k_down = np.zeros((1, 1, N+1, 1), dtype=np.float32)
    k_down[0, 0, 1:, 0] = 1.0
    make_const('k_down', k_down, dtype=F32)
    
    # Apply: left scan = conv(mask, k_left) with pad right=N
    nodes.append(helper.make_node('Conv', ['mask_4d', 'k_left'], ['scan_left_4d'], pads=[0,N,0,0]))
    nodes.append(helper.make_node('Conv', ['mask_4d', 'k_right'], ['scan_right_4d'], pads=[0,0,0,N]))
    nodes.append(helper.make_node('Conv', ['mask_4d', 'k_up'], ['scan_up_4d'], pads=[N,0,0,0]))
    nodes.append(helper.make_node('Conv', ['mask_4d', 'k_down'], ['scan_down_4d'], pads=[0,0,N,0]))
    
    # Has non-8 to left/right/up/down?
    nodes.append(helper.make_node('Reshape', ['scan_left_4d', 'in_shape'], ['scan_left']))
    nodes.append(helper.make_node('Reshape', ['scan_right_4d', 'in_shape'], ['scan_right']))
    nodes.append(helper.make_node('Reshape', ['scan_up_4d', 'in_shape'], ['scan_up']))
    nodes.append(helper.make_node('Reshape', ['scan_down_4d', 'in_shape'], ['scan_down']))
    
    nodes.append(helper.make_node('Cast', ['scan_left'], ['sl_f32'], to=int(F32)))
    nodes.append(helper.make_node('Cast', ['scan_right'], ['sr_f32'], to=int(F32)))
    nodes.append(helper.make_node('Cast', ['scan_up'], ['su_f32'], to=int(F32)))
    nodes.append(helper.make_node('Cast', ['scan_down'], ['sd_f32'], to=int(F32)))
    
    nodes.append(helper.make_node('Greater', ['sl_f32', 'c0_5_f'], ['has_left']))
    nodes.append(helper.make_node('Greater', ['sr_f32', 'c0_5_f'], ['has_right']))
    nodes.append(helper.make_node('Greater', ['su_f32', 'c0_5_f'], ['has_up']))
    nodes.append(helper.make_node('Greater', ['sd_f32', 'c0_5_f'], ['has_down']))
    
    nodes.append(helper.make_node('And', ['has_left', 'has_right'], ['has_lr']))
    nodes.append(helper.make_node('And', ['has_up', 'has_down'], ['has_ud']))
    nodes.append(helper.make_node('And', ['has_lr', 'has_ud'], ['enclosed']))
    nodes.append(helper.make_node('And', ['mask8_bool', 'enclosed'], ['is_4']))
    
    # Build output: 
    # Priority: is_4 > is_3 > original
    nodes.append(helper.make_node('Where', ['is_3', 'c3', 'input'], ['step1']))
    nodes.append(helper.make_node('Where', ['is_4', 'c4', 'step1'], ['output']))
    
    graph = helper.make_graph(nodes, 'task125_graph', [input_info], [output_info])
    model = helper.make_model(graph, producer_name='task125_model', opset_imports=[helper.make_opsetid('', 15)])
    onnx.save(model, 'task125.onnx')

def check_task():
    import onnxruntime as ort
    
    create_onnx_model()
    session = ort.InferenceSession('task125.onnx')
    
    with open('task125.json', 'r') as f:
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
model = onnx.load("/project/repairs/task125.onnx")
