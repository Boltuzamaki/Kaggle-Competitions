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
        return name

    def nn(op, ins, outs, **kw):
        nodes.append(helper.make_node(op, ins, outs, **kw))
        return outs[0] if len(outs) == 1 else outs

    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])

    addK('c0', [0], I64)
    addK('c1', [1], I64)
    addK('c2', [2], I64)
    addK('m1', [-1], I64)
    addK('p999', [999], I64)
    addK('c0f', [0.0], F)
    addK('c1f', [1.0], F)

    # Base inputs
    nn('ArgMax', ['input'], ['I_idx_f'], axis=1, keepdims=0)
    nn('Cast', ['I_idx_f'], ['I_idx_1'], to=I64)
    nn('Reshape', ['I_idx_1', addK('sh3030', [30, 30], I64)], ['I_idx_raw'])
    
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
    
    nn('Cast', ['h_true_f'], ['h_true_i64_1'], to=I64)
    nn('Reshape', ['h_true_i64_1', addK('sh1', [1], I64)], ['h_true_i64'])
    nn('Cast', ['w_true_f'], ['w_true_i64_1'], to=I64)
    nn('Reshape', ['w_true_i64_1', 'sh1'], ['w_true_i64'])

    addK('r_idx', np.arange(30).reshape(30, 1), I64)
    addK('c_idx', np.arange(30).reshape(1, 30), I64)
    nn('Less', ['r_idx', 'h_true_i64'], ['r_valid']) # [30, 1]
    nn('Less', ['c_idx', 'w_true_i64'], ['c_valid']) # [1, 30]
    nn('And', ['r_valid', 'c_valid'], ['in_grid']) # [30, 30]
    
    nn('Where', ['in_grid', 'I_idx_raw', 'm1'], ['I_idx']) # [30, 30]

    # Row logic
    nn('Reshape', ['I_idx', addK('sh30_1_30', [30, 1, 30], I64)], ['I_row_A'])
    nn('Reshape', ['I_idx', addK('sh1_30_30', [1, 30, 30], I64)], ['I_row_B'])
    nn('Equal', ['I_row_A', 'I_row_B'], ['row_eq']) # [30, 30, 30]
    nn('Cast', ['row_eq'], ['row_eq_i64'], to=I64)
    nn('ReduceMin', ['row_eq_i64'], ['row_eq_matrix_i64'], axes=[2], keepdims=0) # [30, 30]
    nn('Equal', ['row_eq_matrix_i64', 'c1'], ['row_eq_matrix']) # [30, 30]

    addK('mask_ij', (np.arange(30).reshape(30, 1) > np.arange(30).reshape(1, 30)).astype(np.int64), I64)
    nn('Equal', ['mask_ij', 'c1'], ['mask_ij_b'])
    nn('And', ['row_eq_matrix', 'mask_ij_b'], ['prev_eq'])
    nn('Cast', ['prev_eq'], ['prev_eq_i64'], to=I64)
    nn('ReduceMax', ['prev_eq_i64'], ['has_prev_i64'], axes=[1], keepdims=0) # [30]
    nn('Equal', ['has_prev_i64', 'c0'], ['is_first_1'])
    addK('arange_30', np.arange(30), I64)
    nn('Less', ['arange_30', 'h_true_i64'], ['in_h']) # [30]
    nn('And', ['is_first_1', 'in_h'], ['is_first']) # [30]

    addK('mask_incl_ij', (np.arange(30).reshape(30, 1) >= np.arange(30).reshape(1, 30)).astype(np.int64), I64)
    nn('Cast', ['is_first'], ['is_first_i64'], to=I64)
    nn('Reshape', ['is_first_i64', addK('sh1_30', [1, 30], I64)], ['is_first_1x30'])
    nn('Mul', ['is_first_1x30', 'mask_incl_ij'], ['is_first_masked']) # [30, 30]
    nn('ReduceSum', ['is_first_masked'], ['prefix_sum'], axes=[1], keepdims=0) # [30]

    nn('Reshape', ['prefix_sum', 'sh1_30'], ['prefix_sum_1x30'])
    nn('Add', ['r_idx', 'c1'], ['target']) # [30, 1]
    nn('Equal', ['prefix_sum_1x30', 'target'], ['match']) # [30, 30]
    nn('Where', ['match', 'c_idx', 'p999'], ['match_idx']) # c_idx is [1, 30]
    nn('ReduceMin', ['match_idx'], ['U_r_raw'], axes=[1], keepdims=0) # [30]
    nn('Equal', ['U_r_raw', 'p999'], ['is_999_r'])
    nn('Where', ['is_999_r', 'm1', 'U_r_raw'], ['U_r']) # [30]

    nn('ReduceMax', ['prefix_sum'], ['R'], axes=[0], keepdims=1) # [1]
    nn('Mul', ['R', 'c2'], ['R_mul_2'])
    nn('Sub', ['R_mul_2', 'c1'], ['limit'])
    nn('Less', ['arange_30', 'R'], ['is_less_R'])
    nn('Less', ['arange_30', 'limit'], ['is_less_limit'])
    nn('Not', ['is_less_R'], ['not_less_R'])
    nn('And', ['not_less_R', 'is_less_limit'], ['is_mid'])

    nn('Sub', ['R_mul_2', 'c2'], ['Rm2'])
    nn('Sub', ['Rm2', 'arange_30'], ['mirror_val'])
    nn('Where', ['is_mid', 'mirror_val', 'm1'], ['M_r_tmp'])
    nn('Where', ['is_less_R', 'arange_30', 'M_r_tmp'], ['M_r']) # [30]

    nn('GreaterOrEqual', ['M_r', 'c0'], ['valid_r_out'])
    nn('Where', ['valid_r_out', 'M_r', 'c0'], ['safe_M_r'])
    nn('Gather', ['U_r', 'safe_M_r'], ['out_row_idx_raw'], axis=0)
    nn('Where', ['valid_r_out', 'out_row_idx_raw', 'm1'], ['out_row_idx']) # [30]

    # Col logic
    nn('Reshape', ['I_idx', addK('sh30_30_1', [30, 30, 1], I64)], ['I_col_A'])
    nn('Reshape', ['I_idx', addK('sh30_1_30_col', [30, 1, 30], I64)], ['I_col_B'])
    nn('Equal', ['I_col_A', 'I_col_B'], ['col_eq']) # [30, 30, 30]
    nn('Cast', ['col_eq'], ['col_eq_i64'], to=I64)
    nn('ReduceMin', ['col_eq_i64'], ['col_eq_matrix_i64'], axes=[0], keepdims=0) # [30, 30]
    nn('Equal', ['col_eq_matrix_i64', 'c1'], ['col_eq_matrix']) # [30, 30]

    nn('And', ['col_eq_matrix', 'mask_ij_b'], ['prev_eq_c'])
    nn('Cast', ['prev_eq_c'], ['prev_eq_c_i64'], to=I64)
    nn('ReduceMax', ['prev_eq_c_i64'], ['has_prev_c_i64'], axes=[1], keepdims=0) # [30]
    nn('Equal', ['has_prev_c_i64', 'c0'], ['is_first_c_1'])
    nn('Less', ['arange_30', 'w_true_i64'], ['in_w']) # [30]
    nn('And', ['is_first_c_1', 'in_w'], ['is_first_c']) # [30]

    nn('Cast', ['is_first_c'], ['is_first_c_i64'], to=I64)
    nn('Reshape', ['is_first_c_i64', 'sh1_30'], ['is_first_c_1x30'])
    nn('Mul', ['is_first_c_1x30', 'mask_incl_ij'], ['is_first_c_masked']) # [30, 30]
    nn('ReduceSum', ['is_first_c_masked'], ['prefix_sum_c'], axes=[1], keepdims=0) # [30]

    nn('Reshape', ['prefix_sum_c', 'sh1_30'], ['prefix_sum_c_1x30'])
    nn('Equal', ['prefix_sum_c_1x30', 'target'], ['match_c']) # [30, 30]
    nn('Where', ['match_c', 'c_idx', 'p999'], ['match_idx_c'])
    nn('ReduceMin', ['match_idx_c'], ['U_c_raw'], axes=[1], keepdims=0) # [30]
    nn('Equal', ['U_c_raw', 'p999'], ['is_999_c'])
    nn('Where', ['is_999_c', 'm1', 'U_c_raw'], ['U_c']) # [30]

    nn('ReduceMax', ['prefix_sum_c'], ['C'], axes=[0], keepdims=1) # [1]
    nn('Mul', ['C', 'c2'], ['C_mul_2'])
    nn('Sub', ['C_mul_2', 'c1'], ['limit_c'])
    nn('Less', ['arange_30', 'C'], ['is_less_C'])
    nn('Less', ['arange_30', 'limit_c'], ['is_less_limit_c'])
    nn('Not', ['is_less_C'], ['not_less_C'])
    nn('And', ['not_less_C', 'is_less_limit_c'], ['is_mid_c'])

    nn('Sub', ['C_mul_2', 'c2'], ['Cm2'])
    nn('Sub', ['Cm2', 'arange_30'], ['mirror_val_c'])
    nn('Where', ['is_mid_c', 'mirror_val_c', 'm1'], ['M_c_tmp'])
    nn('Where', ['is_less_C', 'arange_30', 'M_c_tmp'], ['M_c']) # [30]

    nn('GreaterOrEqual', ['M_c', 'c0'], ['valid_c_out'])
    nn('Where', ['valid_c_out', 'M_c', 'c0'], ['safe_M_c'])
    nn('Gather', ['U_c', 'safe_M_c'], ['out_col_idx_raw'], axis=0)
    nn('Where', ['valid_c_out', 'out_col_idx_raw', 'm1'], ['out_col_idx']) # [30]

    # Combine indices
    nn('Reshape', ['out_row_idx', addK('sh30_1', [30, 1], I64)], ['row_idx_2d_out']) # shape is [30, 1]
    nn('Reshape', ['out_col_idx', 'sh1_30'], ['col_idx_2d_out']) # [1, 30]
    
    addK('c30', [30], I64)
    nn('Mul', ['row_idx_2d_out', 'c30'], ['r_mul_30'])
    nn('Add', ['r_mul_30', 'col_idx_2d_out'], ['flat_idx']) # [30, 30]
    
    nn('GreaterOrEqual', ['row_idx_2d_out', 'c0'], ['v_r'])
    nn('GreaterOrEqual', ['col_idx_2d_out', 'c0'], ['v_c'])
    nn('And', ['v_r', 'v_c'], ['valid_out']) # [30, 30]
    
    nn('Where', ['valid_out', 'flat_idx', 'c0'], ['safe_flat_idx']) # [30, 30]
    nn('Reshape', ['safe_flat_idx', addK('sh900', [900], I64)], ['safe_flat_idx_1d'])
    
    nn('Reshape', ['I_idx', 'sh900'], ['I_idx_1d'])
    nn('Gather', ['I_idx_1d', 'safe_flat_idx_1d'], ['O_idx_flat_raw'], axis=0)
    
    nn('Reshape', ['valid_out', 'sh900'], ['valid_out_1d'])
    nn('Where', ['valid_out_1d', 'O_idx_flat_raw', 'm1'], ['O_idx_flat']) # [900]
    
    nn('Reshape', ['O_idx_flat', 'sh3030'], ['O_idx_raw'])
    nn('Reshape', ['O_idx_raw', addK('sh1_1_30_30', [1, 1, 30, 30], I64)], ['O_idx'])
    
    # 10-channel one-hot
    addK('c_range', np.arange(10).reshape(1, 10, 1, 1).astype(np.int64), I64)
    nn('Equal', ['O_idx', 'c_range'], ['O_eq'])
    nn('Cast', ['O_eq'], ['output'], to=F)

    graph = helper.make_graph(nodes, 'task377', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])

model = _bake(_make(), 377)
if __name__ == "__main__":
    import onnx
    onnx.save(model, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\repairs\task377.onnx")
