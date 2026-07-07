# --- task022 (ARC 137eaa0f): stamp each object into a 3x3 canvas centred on its colour-5 cell.
# Every object contains exactly one 5. Overlay all objects so their 5-cells coincide at the
# centre (1,1) of a 3x3 grid; each object contributes its cells at offsets (dr,dc) in {-1,0,1}.
# YOUR LOGIC WAS CORRECT and elegant: for each of the 9 offsets, shift the padded input by
# (dr,dc), multiply by M=channel-5 mask, ReduceMax over space -> "which colour sits at offset
# (dr,dc) from ANY 5" -> place at grid cell (1+dr, 1+dc). Reproduces the example exactly.
# One fix — output SHAPE (your check_task() compared the native 3x3 and hid this):
#   The scorer encodes EVERY target as a fixed [1,10,30,30] canvas and does np.array_equal.
#   A [1,10,3,3] output mismatches on shape -> all fail. Build the 3x3, then Pad to 30x30 with
#   the 3x3 at top-left and everything else all-zero (empty). ch0=0.5 tie-break keeps in-grid
#   background as colour-0; the pad stays empty because we pad ALL channels with 0.
# Verified 266/266. (Cost ~694865 -> 11.55 pts vs baseline 2302 -> 17.26 pts, so the submission
#  keeps baseline; you still own a correct build.)
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])  # FIX: full 30x30 canvas
    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
    inits = [
        K('c5_s', [5]), K('c5_e', [6]), K('ax_1', [1]), K('ax_0_1_2_3', [0, 1, 2, 3]),
        K('pads_1', [0, 0, 1, 1, 0, 0, 1, 1]),
        K('depth10', 10), K('oh_vals', [0.0, 1.0], dtype=np.float32),
        K('ch0_05', 0.5 * np.ones((1, 1, 3, 3), dtype=np.float32), dtype=np.float32),
        K('c1_s', [1]), K('c1_e', [10]),
        K('pads_out', [0, 0, 0, 0, 0, 0, 27, 27]),  # FIX: pad 3x3 -> 30x30 (top-left)
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'c5_s', 'c5_e', 'ax_1'], ['M']),   # colour-5 positions
        helper.make_node('Pad', ['input', 'pads_1'], ['padded_X']),
    ]
    for r_idx, dr in enumerate([-1, 0, 1]):
        for c_idx, dc in enumerate([-1, 0, 1]):
            start_r, end_r = 1 + dr, 31 + dr
            start_c, end_c = 1 + dc, 31 + dc
            inits.extend([
                K(f'starts_{r_idx}_{c_idx}', [0, 0, start_r, start_c]),
                K(f'ends_{r_idx}_{c_idx}', [1, 10, end_r, end_c]),
            ])
            name = f'{r_idx}_{c_idx}'
            nodes.extend([
                # shifted[k,i,j] = input[k, i+dr, j+dc]
                helper.make_node('Slice', ['padded_X', f'starts_{name}', f'ends_{name}', 'ax_0_1_2_3'], [f'shifted_{name}']),
                helper.make_node('Mul', [f'shifted_{name}', 'M'], [f'masked_{name}']),      # keep only cells that ARE a 5
                helper.make_node('ReduceMax', [f'masked_{name}'], [f'val_{name}'], axes=[2, 3], keepdims=1),
            ])
    for r_idx in range(3):
        nodes.append(helper.make_node('Concat', [f'val_{r_idx}_0', f'val_{r_idx}_1', f'val_{r_idx}_2'], [f'row_{r_idx}'], axis=3))
    nodes.append(helper.make_node('Concat', ['row_0', 'row_1', 'row_2'], ['out_grid'], axis=2))
    nodes.extend([
        helper.make_node('Slice', ['out_grid', 'c1_s', 'c1_e', 'ax_1'], ['out_grid_1_9']),
        helper.make_node('Concat', ['ch0_05', 'out_grid_1_9'], ['out_fixed'], axis=1),   # bg tie-break at 0.5
        helper.make_node('ArgMax', ['out_fixed'], ['pred_idx'], axis=1, keepdims=0),
        helper.make_node('OneHot', ['pred_idx', 'depth10', 'oh_vals'], ['oh_3x3'], axis=1),
        # FIX: pad the 3x3 one-hot out to the full 30x30 canvas (rest all-zero = empty)
        helper.make_node('Pad', ['oh_3x3', 'pads_out'], ['output']),
    ])
    graph = helper.make_graph(nodes, 'task022', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

model = create_model()
