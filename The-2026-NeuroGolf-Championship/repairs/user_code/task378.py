import onnx
from onnx import helper
import onnx.shape_inference
from onnx import TensorProto
import numpy as np
import copy as _copy

def _bake(m, t, out_names=None):
    if out_names is None:
        out_names = []
    
    missing = [x for x in out_names if x not in {vi.name for vi in m.graph.output}]
    if not missing: return m
    
    tmp = _copy.deepcopy(m)
    inf = onnx.shape_inference.infer_shapes(tmp, strict_mode=True)
    
    for nm in missing:
        found = False
        for vi in inf.graph.value_info:
            if vi.name == nm: 
                tmp.graph.output.append(vi)
                found = True
                break
        if not found:
            for vi in inf.graph.input:
                if vi.name == nm:
                    tmp.graph.output.append(vi)
                    found = True
                    break
        if not found:
            for nd in tmp.graph.node:
                if nm in nd.output:
                    tmp.graph.output.append(onnx.helper.make_tensor_value_info(nm, onnx.TensorProto.INT64, [1]))
                    
    return tmp

def _make():
    F = TensorProto.FLOAT
    I64 = TensorProto.INT64
    B = TensorProto.BOOL

    nodes = []
    inits = []
    
    def addK(name, v, dtype):
        inits.append(helper.make_tensor(name, dtype, np.array(v).shape, np.array(v).flatten().tolist()))

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    addK('row_idx2d', np.arange(30).reshape(1, 1, 30, 1), I64)
    addK('col_idx2d', np.arange(30).reshape(1, 1, 1, 30), I64)
    addK('m1', [-1], I64)
    addK('p999', [999], I64)
    addK('c0f', [0.0], F)
    addK('c1f', [1.0], F)
    addK('c0i_i64', [0], I64)

    # Base inputs
    nn('ArgMax', ['input'], ['I_idx_f'], axis=1, keepdims=1)
    nn('Cast', ['I_idx_f'], ['I_idx_raw'], to=I64)
    
    # ---- true grid bounds (h_true, w_true) via one-hot "presence" ----
    nn('ReduceMax', ['input'], ['presence_f'], axes=[1], keepdims=1)
    
    # h_true
    nn('ReduceMax', ['presence_f'], ['row_any_p'], axes=[3], keepdims=1)
    addK('row_idx2d_f', np.arange(30).reshape(1, 1, 30, 1).astype(np.float32), F)
    nn('Mul', ['row_idx2d_f', 'row_any_p'], ['row_idx_masked'])
    nn('ReduceMax', ['row_idx_masked'], ['r_max_p_f'], axes=[2], keepdims=1)
    nn('Add', ['r_max_p_f', 'c1f'], ['h_true_f'])
    
    # w_true
    nn('ReduceMax', ['presence_f'], ['col_any_p'], axes=[2], keepdims=1)
    addK('col_idx2d_f', np.arange(30).reshape(1, 1, 1, 30).astype(np.float32), F)
    nn('Mul', ['col_idx2d_f', 'col_any_p'], ['col_idx_masked'])
    nn('ReduceMax', ['col_idx_masked'], ['c_max_p_f'], axes=[3], keepdims=1)
    nn('Add', ['c_max_p_f', 'c1f'], ['w_true_f'])
    
    nn('Cast', ['h_true_f'], ['h_true_i64'], to=I64)
    nn('Cast', ['w_true_f'], ['w_true_i64'], to=I64)
    
    nn('Less', ['row_idx2d', 'h_true_i64'], ['r_valid'])
    nn('Less', ['col_idx2d', 'w_true_i64'], ['c_valid'])
    nn('And', ['r_valid', 'c_valid'], ['in_grid'])
    
    # Set out-of-bounds to -1
    nn('Where', ['in_grid', 'I_idx_raw', 'm1'], ['I_idx'])

    # 1. Colors loop to find bounding box properties
    addK('colors_1_9', np.arange(1, 10).reshape(1, 9, 1, 1), I64) # [1, 9, 1, 1]
    nn('Equal', ['I_idx', 'colors_1_9'], ['M_c']) # shape [1, 9, 30, 30]
    
    nn('Cast', ['M_c'], ['M_c_i64'], to=I64)
    nn('ReduceMax', ['M_c_i64'], ['has_c'], axes=[2, 3], keepdims=1)
    addK('c1i_i64', [1], I64)
    nn('Equal', ['has_c', 'c1i_i64'], ['has_c_b'])

    # r_max_c
    nn('Where', ['M_c', 'row_idx2d', 'm1'], ['r_mask_max_c']) # shape [1, 9, 30, 30]
    nn('ReduceMax', ['r_mask_max_c'], ['r_max_c'], axes=[2, 3], keepdims=1) # shape [1, 9, 1, 1]
    
    # r_min_c
    nn('Where', ['M_c', 'row_idx2d', 'p999'], ['r_mask_min_c'])
    nn('ReduceMin', ['r_mask_min_c'], ['r_min_c'], axes=[2, 3], keepdims=1)
    
    # c_max_c
    nn('Where', ['M_c', 'col_idx2d', 'm1'], ['c_mask_max_c'])
    nn('ReduceMax', ['c_mask_max_c'], ['c_max_c'], axes=[2, 3], keepdims=1)
    
    # c_min_c
    nn('Where', ['M_c', 'col_idx2d', 'p999'], ['c_mask_min_c'])
    nn('ReduceMin', ['c_mask_min_c'], ['c_min_c'], axes=[2, 3], keepdims=1)

    # area_c
    nn('Sub', ['r_max_c', 'r_min_c'], ['r_diff_area'])
    nn('Add', ['r_diff_area', 'c1i_i64'], ['r_len'])
    nn('Sub', ['c_max_c', 'c_min_c'], ['c_diff_area'])
    nn('Add', ['c_diff_area', 'c1i_i64'], ['c_len'])
    nn('Mul', ['r_len', 'c_len'], ['area_c_raw'])
    
    addK('c0_i64', [0], I64)
    nn('Where', ['has_c_b', 'area_c_raw', 'c0_i64'], ['area_c']) # [1, 9, 1, 1]
    
    # ArgMax of area_c over axis 1
    nn('Cast', ['area_c'], ['area_c_f'], to=F)
    nn('ArgMax', ['area_c_f'], ['x5_idx_0'], axis=1, keepdims=1) # [1, 1, 1, 1]
    nn('Cast', ['x5_idx_0'], ['x5_idx_0_i64'], to=I64)
    nn('Add', ['x5_idx_0_i64', 'c1i_i64'], ['x5']) # [1, 1, 1, 1]
    
    # x7 = the other color
    nn('ReduceSum', ['M_c_i64'], ['count_c'], axes=[2, 3], keepdims=1) # [1, 9, 1, 1]
    nn('Equal', ['colors_1_9', 'x5'], ['is_x5']) # [1, 9, 1, 1]
    nn('Where', ['is_x5', 'c0_i64', 'count_c'], ['count_c_masked'])
    nn('Cast', ['count_c_masked'], ['count_c_masked_f'], to=F)
    nn('ArgMax', ['count_c_masked_f'], ['x7_idx_0'], axis=1, keepdims=1)
    nn('Cast', ['x7_idx_0'], ['x7_idx_0_i64'], to=I64)
    nn('Add', ['x7_idx_0_i64', 'c1i_i64'], ['x7']) # [1, 1, 1, 1]

    # Gather rmin, rmax, cmin, cmax for x5
    nn('GatherElements', ['r_min_c', 'x5_idx_0_i64'], ['rmin'], axis=1) # [1, 1, 1, 1]
    nn('GatherElements', ['r_max_c', 'x5_idx_0_i64'], ['rmax'], axis=1)
    nn('GatherElements', ['c_min_c', 'x5_idx_0_i64'], ['cmin'], axis=1)
    nn('GatherElements', ['c_max_c', 'x5_idx_0_i64'], ['cmax'], axis=1)
    
    # Ray 1: r > rmax AND r - rmax == c - cmax
    nn('Sub', ['row_idx2d', 'rmax'], ['r_diff_max']) # [1, 1, 30, 1]
    nn('Sub', ['col_idx2d', 'cmax'], ['c_diff_max']) # [1, 1, 1, 30]
    nn('Greater', ['r_diff_max', 'c0_i64'], ['r_gt_max'])
    nn('Equal', ['r_diff_max', 'c_diff_max'], ['eq1'])
    nn('And', ['r_gt_max', 'eq1'], ['ray1'])
    
    # Ray 2: r > rmax AND r - rmax == cmin - c
    nn('Sub', ['cmin', 'col_idx2d'], ['c_diff_min']) # [1, 1, 1, 30]
    nn('Equal', ['r_diff_max', 'c_diff_min'], ['eq2'])
    nn('And', ['r_gt_max', 'eq2'], ['ray2'])
    
    # Ray 3: r < rmin AND rmin - r == c - cmax
    nn('Sub', ['rmin', 'row_idx2d'], ['r_diff_min']) # [1, 1, 30, 1]
    nn('Greater', ['r_diff_min', 'c0_i64'], ['r_lt_min'])
    nn('Equal', ['r_diff_min', 'c_diff_max'], ['eq3'])
    nn('And', ['r_lt_min', 'eq3'], ['ray3'])
    
    # Ray 4: r < rmin AND rmin - r == cmin - c
    nn('Equal', ['r_diff_min', 'c_diff_min'], ['eq4'])
    nn('And', ['r_lt_min', 'eq4'], ['ray4'])
    
    nn('Or', ['ray1', 'ray2'], ['r12'])
    nn('Or', ['ray3', 'ray4'], ['r34'])
    nn('Or', ['r12', 'r34'], ['any_ray'])
    
    # Background mask: I_idx == 0
    nn('Equal', ['I_idx', 'c0i_i64'], ['M0'])
    nn('And', ['any_ray', 'M0'], ['paint_mask'])
    
    # Final output index grid
    nn('Where', ['paint_mask', 'x7', 'I_idx'], ['O_idx'])
    
    # Map back to 10-channel one-hot
    addK('c_range', np.arange(10).reshape(1, 10, 1, 1).astype(np.int64), I64)
    nn('Equal', ['O_idx', 'c_range'], ['O_eq'])
    nn('Cast', ['O_eq'], ['output'], to=F)

    graph = helper.make_graph(nodes, 'task378', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])

model = _bake(_make(), 378)
if __name__ == "__main__":
    import onnx
    onnx.save(model, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\repairs\task378.onnx")
