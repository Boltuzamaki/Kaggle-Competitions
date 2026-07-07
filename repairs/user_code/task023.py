import numpy as np
import onnx
from onnx import helper
import copy as _copy
import onnx.shape_inference

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
    # Model signature:
    # input: float32[B, 10, 30, 30]
    # output: float32[B, 10, 30, 30]
    # Actually neurogolf expects Bx10x30x30 in and out
    F = getattr(onnx.TensorProto, 'FLOAT')
    I64 = getattr(onnx.TensorProto, 'INT64')
    
    x = helper.make_tensor_value_info('input', F, [1, 10, 30, 30])
    y = helper.make_tensor_value_info('output', F, [1, 10, 30, 30])
    
    nodes = []
    inits = []
    
    def addK(name, val, dtype):
        inits.append(helper.make_tensor(name, dtype, val.shape, val.flatten().tolist()))
        return name
        
    def nn(op, inputs, outputs, **kwargs):
        nodes.append(helper.make_node(op, inputs, outputs, **kwargs))
        
    addK('c_range', np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1), I64)
    addK('sh1_1_30_30', np.array([1, 1, 30, 30], dtype=np.int64), I64)
    addK('sh3030', np.array([30, 30], dtype=np.int64), I64)
    addK('shB_1_30_30', np.array([-1, 1, 30, 30], dtype=np.int64), I64)
    
    # 1. Get input grid as I_idx [B, 30, 30] using ArgMax
    nn('ArgMax', ['input'], ['I_idx_1'], axis=1, keepdims=1)
    nn('Cast', ['I_idx_1'], ['I_idx_f'], to=F)
    
    # Identify 5s
    addK('f5', np.array([5.0], dtype=np.float32), F)
    addK('f0', np.array([0.0], dtype=np.float32), F)
    addK('f1', np.array([1.0], dtype=np.float32), F)
    addK('f2', np.array([2.0], dtype=np.float32), F)
    addK('f8', np.array([8.0], dtype=np.float32), F)
    
    addK('k2', np.ones((1, 1, 2, 2), dtype=np.float32), F)
    addK('k13', np.ones((1, 1, 1, 3), dtype=np.float32), F)
    addK('k31', np.ones((1, 1, 3, 1), dtype=np.float32), F)
    
    addK('f4', np.array([4.0], dtype=np.float32), F)
    addK('f3', np.array([3.0], dtype=np.float32), F)
    
    nn('Equal', ['I_idx_f', 'f5'], ['is_5_bool'])
    nn('Cast', ['is_5_bool'], ['uncovered'], to=F)
    
    # final_2x2 is accumulated
    nn('Constant', [], ['final_2x2'], value=helper.make_tensor('zero_init', F, [1, 1, 30, 30], np.zeros((1, 1, 30, 30), dtype=np.float32).flatten().tolist()))
    
    # 15 iterations of unrolled exact cover
    curr_uncovered = 'uncovered'
    curr_final_2x2 = 'final_2x2'
    
    for i in range(15):
        v2_sum = f'v2_sum_{i}'
        v2_bool = f'v2_bool_{i}'
        v2 = f'v2_{i}'
        
        v13_sum = f'v13_sum_{i}'
        v13_bool = f'v13_bool_{i}'
        v13 = f'v13_{i}'
        
        v31_sum = f'v31_sum_{i}'
        v31_bool = f'v31_bool_{i}'
        v31 = f'v31_{i}'
        
        nn('Conv', [curr_uncovered, 'k2'], [v2_sum], pads=[0, 0, 1, 1])
        nn('Equal', [v2_sum, 'f4'], [v2_bool])
        nn('Cast', [v2_bool], [v2], to=F)
        
        nn('Conv', [curr_uncovered, 'k13'], [v13_sum], pads=[0, 0, 0, 2])
        nn('Equal', [v13_sum, 'f3'], [v13_bool])
        nn('Cast', [v13_bool], [v13], to=F)
        
        nn('Conv', [curr_uncovered, 'k31'], [v31_sum], pads=[0, 0, 2, 0])
        nn('Equal', [v31_sum, 'f3'], [v31_bool])
        nn('Cast', [v31_bool], [v31], to=F)
        
        c2 = f'c2_{i}'
        c13 = f'c13_{i}'
        c31 = f'c31_{i}'
        c_sum1 = f'c_sum1_{i}'
        c = f'c_{i}'
        
        nn('Conv', [v2, 'k2'], [c2], pads=[1, 1, 0, 0])
        nn('Conv', [v13, 'k13'], [c13], pads=[0, 2, 0, 0])
        nn('Conv', [v31, 'k31'], [c31], pads=[2, 0, 0, 0])
        nn('Add', [c2, c13], [c_sum1])
        nn('Add', [c_sum1, c31], [c])
        
        c_eq1 = f'c_eq1_{i}'
        single_f = f'single_f_{i}'
        single = f'single_{i}'
        
        nn('Equal', [c, 'f1'], [c_eq1])
        nn('Cast', [c_eq1], [single_f], to=F)
        nn('Mul', [curr_uncovered, single_f], [single])
        
        s_sum2 = f's_sum2_{i}'
        a2_bool = f'a2_bool_{i}'
        a2_f = f'a2_f_{i}'
        a2 = f'a2_{i}'
        nn('Conv', [single, 'k2'], [s_sum2], pads=[0, 0, 1, 1])
        nn('Greater', [s_sum2, 'f0'], [a2_bool])
        nn('Cast', [a2_bool], [a2_f], to=F)
        nn('Mul', [v2, a2_f], [a2])
        
        s_sum13 = f's_sum13_{i}'
        a13_bool = f'a13_bool_{i}'
        a13_f = f'a13_f_{i}'
        a13 = f'a13_{i}'
        nn('Conv', [single, 'k13'], [s_sum13], pads=[0, 0, 0, 2])
        nn('Greater', [s_sum13, 'f0'], [a13_bool])
        nn('Cast', [a13_bool], [a13_f], to=F)
        nn('Mul', [v13, a13_f], [a13])
        
        s_sum31 = f's_sum31_{i}'
        a31_bool = f'a31_bool_{i}'
        a31_f = f'a31_f_{i}'
        a31 = f'a31_{i}'
        nn('Conv', [single, 'k31'], [s_sum31], pads=[0, 0, 2, 0])
        nn('Greater', [s_sum31, 'f0'], [a31_bool])
        nn('Cast', [a31_bool], [a31_f], to=F)
        nn('Mul', [v31, a31_f], [a31])
        
        cov2 = f'cov2_{i}'
        cov13 = f'cov13_{i}'
        cov31 = f'cov31_{i}'
        cov_sum1 = f'cov_sum1_{i}'
        cov = f'cov_{i}'
        cov_bool = f'cov_bool_{i}'
        cov_f = f'cov_f_{i}'
        not_cov = f'not_cov_{i}'
        
        nn('Conv', [a2, 'k2'], [cov2], pads=[1, 1, 0, 0])
        nn('Conv', [a13, 'k13'], [cov13], pads=[0, 2, 0, 0])
        nn('Conv', [a31, 'k31'], [cov31], pads=[2, 0, 0, 0])
        
        nn('Add', [cov2, cov13], [cov_sum1])
        nn('Add', [cov_sum1, cov31], [cov])
        nn('Greater', [cov, 'f0'], [cov_bool])
        nn('Cast', [cov_bool], [cov_f], to=F)
        
        nn('Sub', ['f1', cov_f], [not_cov])
        
        next_uncovered = f'uncovered_{i+1}'
        nn('Mul', [curr_uncovered, not_cov], [next_uncovered])
        
        next_final_2x2_sum = f'final_2x2_sum_{i+1}'
        next_final_2x2_bool = f'final_2x2_bool_{i+1}'
        next_final_2x2 = f'final_2x2_{i+1}'
        
        nn('Add', [curr_final_2x2, a2], [next_final_2x2_sum])
        nn('Greater', [next_final_2x2_sum, 'f0'], [next_final_2x2_bool])
        nn('Cast', [next_final_2x2_bool], [next_final_2x2], to=F)
        
        curr_uncovered = next_uncovered
        curr_final_2x2 = next_final_2x2

    # Now we have the final 2x2 mask
    nn('Conv', [curr_final_2x2, 'k2'], ['cov_final_2x2'], pads=[1, 1, 0, 0])
    nn('Greater', ['cov_final_2x2', 'f0'], ['is_final_2x2_bool'])
    nn('Cast', ['is_final_2x2_bool'], ['is_final_2x2_f'], to=F)
    
    is_5 = 'uncovered' # (I == 5) was evaluated into uncovered initially, wait!
    # I replaced uncovered above.
    is_5_f = 'is_5_bool_f'
    nn('Cast', ['is_5_bool'], [is_5_f], to=F)
    
    not_is_5_f = 'not_is_5_f'
    nn('Sub', ['f1', is_5_f], [not_is_5_f])
    
    O_pred_0 = 'O_pred_0'
    nn('Mul', ['I_idx_f', not_is_5_f], [O_pred_0])
    
    is_2_f = 'is_2_f'
    not_final_2x2_f = 'not_final_2x2_f'
    nn('Sub', ['f1', 'is_final_2x2_f'], [not_final_2x2_f])
    nn('Mul', [is_5_f, not_final_2x2_f], [is_2_f])
    
    O_pred_1 = 'O_pred_1'
    is_2_f_x2 = 'is_2_f_x2'
    nn('Mul', [is_2_f, 'f2'], [is_2_f_x2])
    nn('Add', [O_pred_0, is_2_f_x2], [O_pred_1])
    
    is_8_f = 'is_8_f'
    nn('Mul', [is_5_f, 'is_final_2x2_f'], [is_8_f])
    
    O_pred_2 = 'O_pred_2'
    is_8_f_x8 = 'is_8_f_x8'
    nn('Mul', [is_8_f, 'f8'], [is_8_f_x8])
    nn('Add', [O_pred_1, is_8_f_x8], [O_pred_2])
    
    nn('Cast', [O_pred_2], ['O_idx'], to=I64)
    
    # Convert back to one-hot output [B, 10, 30, 30]
    nn('Equal', ['O_idx', 'c_range'], ['O_eq'])
    
    # mask out padding
    nn('ReduceSum', ['input'], ['valid_mask'], axes=[1], keepdims=1)
    nn('Cast', ['O_eq'], ['O_eq_f'], to=F)
    nn('Mul', ['O_eq_f', 'valid_mask'], ['output'])
    
    graph = helper.make_graph(nodes, 'task023', [x], [y], inits)
    return helper.make_model(graph, ir_version=8, opset_imports=[helper.make_opsetid('', 12)])

model = _bake(_make(), 23)
if __name__ == "__main__":
    import onnx
    onnx.save(model, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\repairs\task023.onnx")
