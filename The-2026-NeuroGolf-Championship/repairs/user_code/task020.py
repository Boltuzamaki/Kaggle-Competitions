# --- task020 (ARC 11852cab): symmetrize the object under the full dihedral group D4.
# Overlay the object with all 8 reflections/rotations of itself, each aligned about the
# object's own bounding-box centre (identity, hflip, vflip, 180, and the 4 transposed variants).
# YOUR LOGIC WAS CORRECT. One fix — the canvas trap (your check_task() cropped `pred` to the
# grid before comparing, which hid it; the real scorer compares the RAW 30x30 one-hot tensor):
#   `ch0_05` makes background(=0.5) win via ArgMax everywhere no colour lands, so OneHot lit
#   channel-0 across the whole 30x30 -> decoder returned 30x30 instead of HxW. Multiply the
#   output by the grid-rectangle mask  M = ReduceSum(input over all 10 channels)  (=1 inside the
#   grid incl. background, 0 outside). Every reflection stays within the object's bbox âŠ‚ grid,
#   so this mask is exactly right.
# Also baked value_info for the 8 dynamic-Slice outputs so shape inference stays static (cost!=None).
# Verified 266/266. (Cost ~3.29M -> 9.99 pts vs baseline 1526 -> 17.67 pts: the 8 padded 88x88
#  buffers are memory-heavy, so the submission keeps baseline; you still own a correct build.)
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def create_model():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    def K(name, arr, dtype=np.int64):
        return numpy_helper.from_array(np.array(arr, dtype=dtype), name=name)
    inits = [
        K('c1_s', [1]), K('c1_e', [10]), K('ax_1', [1]),
        K('ax_0', [0]), K('ax_0_1_2_3', [0, 1, 2, 3]),
        K('c0_1d', [0]), K('c29_1d', [29]), K('c30_1d', [30]), K('c58_1d', [58]),
        K('arange30', np.arange(30)), K('rev_arange30', np.arange(29, -1, -1)),
        K('pads_29', [0, 0, 29, 29, 0, 0, 29, 29]),
        K('depth10', 10), K('oh_vals', [0.0, 1.0], dtype=np.float32),
        K('ch0_05', 0.5 * np.ones((1, 1, 30, 30), dtype=np.float32), dtype=np.float32),
    ]
    nodes = [
        helper.make_node('Slice', ['input', 'c1_s', 'c1_e', 'ax_1'], ['c1_9']),
        helper.make_node('ReduceMax', ['c1_9'], ['mask'], axes=[1], keepdims=1),
        helper.make_node('Cast', ['mask'], ['mask_i'], to=I64),
        # bounding box of the object (row/col projections, no NonZero)
        helper.make_node('ReduceMax', ['mask_i'], ['r_proj'], axes=[3], keepdims=0),
        helper.make_node('Mul', ['r_proj', 'arange30'], ['r_m1']),
        helper.make_node('ReduceMax', ['r_m1'], ['r_max_2d'], axes=[2], keepdims=0),
        helper.make_node('Squeeze', ['r_max_2d', 'ax_0'], ['r_max']),
        helper.make_node('Mul', ['r_proj', 'rev_arange30'], ['r_m2']),
        helper.make_node('ReduceMax', ['r_m2'], ['r_min_inv_2d'], axes=[2], keepdims=0),
        helper.make_node('Squeeze', ['r_min_inv_2d', 'ax_0'], ['r_min_inv']),
        helper.make_node('Sub', ['c29_1d', 'r_min_inv'], ['r_min']),
        helper.make_node('ReduceMax', ['mask_i'], ['c_proj'], axes=[2], keepdims=0),
        helper.make_node('Mul', ['c_proj', 'arange30'], ['c_m1']),
        helper.make_node('ReduceMax', ['c_m1'], ['c_max_2d'], axes=[2], keepdims=0),
        helper.make_node('Squeeze', ['c_max_2d', 'ax_0'], ['c_max']),
        helper.make_node('Mul', ['c_proj', 'rev_arange30'], ['c_m2']),
        helper.make_node('ReduceMax', ['c_m2'], ['c_min_inv_2d'], axes=[2], keepdims=0),
        helper.make_node('Squeeze', ['c_min_inv_2d', 'ax_0'], ['c_min_inv']),
        helper.make_node('Sub', ['c29_1d', 'c_min_inv'], ['c_min']),
        # reflection offsets: a flip about the bbox maps index k -> (min+max) - k
        helper.make_node('Add', ['r_min', 'r_max'], ['sum_R']),
        helper.make_node('Add', ['c_min', 'c_max'], ['sum_C']),
        helper.make_node('Sub', ['r_min', 'c_min'], ['diff_R_C']),
        helper.make_node('Sub', ['c_min', 'r_min'], ['diff_C_R']),
        helper.make_node('Add', ['r_max', 'c_min'], ['rmax_cmin']),
        helper.make_node('Add', ['c_max', 'r_min'], ['cmax_rmin']),
        helper.make_node('Sub', ['c58_1d', 'sum_R'], ['s1_r']),
        helper.make_node('Sub', ['c58_1d', 'sum_C'], ['s2_c']),
        helper.make_node('Sub', ['c29_1d', 'diff_R_C'], ['s4_r']),
        helper.make_node('Sub', ['c29_1d', 'diff_C_R'], ['s4_c']),
        helper.make_node('Sub', ['c58_1d', 'rmax_cmin'], ['s5_r']),
        helper.make_node('Sub', ['c58_1d', 'cmax_rmin'], ['s6_c']),
        # per-variant slice starts (into the 29-padded grid), one per D4 element
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 'c29_1d', 'c29_1d'], ['starts0'], axis=0),  # identity
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 's1_r', 'c29_1d'], ['starts1'], axis=0),     # hflip
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 'c29_1d', 's2_c'], ['starts2'], axis=0),     # vflip
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 's1_r', 's2_c'], ['starts3'], axis=0),       # 180
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 's4_r', 's4_c'], ['starts4'], axis=0),       # transpose
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 's5_r', 's4_c'], ['starts5'], axis=0),
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 's4_r', 's6_c'], ['starts6'], axis=0),
        helper.make_node('Concat', ['c0_1d', 'c0_1d', 's5_r', 's6_c'], ['starts7'], axis=0),
        # the 8 transformed base grids (flips via reversed Gather, transpose via Transpose)
        helper.make_node('Identity', ['input'], ['g0']),
        helper.make_node('Gather', ['input', 'rev_arange30'], ['g1'], axis=2),
        helper.make_node('Gather', ['input', 'rev_arange30'], ['g2'], axis=3),
        helper.make_node('Gather', ['g1', 'rev_arange30'], ['g3'], axis=3),
        helper.make_node('Transpose', ['input'], ['gt'], perm=[0, 1, 3, 2]),
        helper.make_node('Identity', ['gt'], ['g4']),
        helper.make_node('Gather', ['gt', 'rev_arange30'], ['g5'], axis=2),
        helper.make_node('Gather', ['gt', 'rev_arange30'], ['g6'], axis=3),
        helper.make_node('Gather', ['g5', 'rev_arange30'], ['g7'], axis=3),
    ]
    outputs = []
    for i in range(8):
        nodes.extend([
            helper.make_node('Pad', [f'g{i}', 'pads_29'], [f'padded_g{i}']),
            helper.make_node('Add', [f'starts{i}', 'c30_1d'], [f'ends{i}']),
            helper.make_node('Slice', [f'padded_g{i}', f'starts{i}', f'ends{i}', 'ax_0_1_2_3'], [f'out_g{i}']),
        ])
        outputs.append(f'out_g{i}')
    nodes.extend([
        helper.make_node('Max', outputs, ['max_all']),                       # overlay all 8 copies
        helper.make_node('Slice', ['max_all', 'c1_s', 'c1_e', 'ax_1'], ['max_all_1_9']),
        helper.make_node('Concat', ['ch0_05', 'max_all_1_9'], ['max_all_fixed'], axis=1),  # bg tie-break at 0.5
        helper.make_node('ArgMax', ['max_all_fixed'], ['pred_idx'], axis=1, keepdims=0),
        helper.make_node('OneHot', ['pred_idx', 'depth10', 'oh_vals'], ['oh_f'], axis=1),
        # FIX (grid size): keep only the actual grid rectangle so padding decodes as empty
        helper.make_node('ReduceSum', ['input', 'ax_1'], ['content'], keepdims=1),
        helper.make_node('Mul', ['oh_f', 'content'], ['output']),
    ])
    graph = helper.make_graph(nodes, 'task020', [x], [y], inits)
    # keep the 8 dynamic-Slice outputs statically shaped so cost != None
    for i in range(8):
        graph.value_info.append(helper.make_tensor_value_info(f'out_g{i}', F, [1, 10, 30, 30]))
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 13)])

model = create_model()
