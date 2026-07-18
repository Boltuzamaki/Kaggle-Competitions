# --- task010: recolour each vertical colour-5 bar by its HEIGHT RANK (tallest -> 1, next -> 2, ...).
# Your logic was correct; two fixes applied:
#   (1) crop to the 9x9 grid before OneHot so the 30x30 padding stays all-zero (not channel-0),
#   (2) opset-12: ReduceSum / Squeeze take `axes` as an ATTRIBUTE (not an input).
# Verified 265/265. (Cost 16865 > baseline 1070, so submission keeps baseline; you still own it.)
import numpy as np
F = TensorProto.FLOAT
x = helper.make_tensor_value_info('input',  F, [1,10,30,30])
y = helper.make_tensor_value_info('output', F, [1,10,30,30])
def K(n,a): return numpy_helper.from_array(a, name=n)
inits = [
 K('c5_s',np.array([5],np.int64)), K('c5_e',np.array([6],np.int64)), K('c5_ax',np.array([1],np.int64)),
 K('shape_col',np.array([1,1,30,1],np.int64)), K('shape_row',np.array([1,1,1,30],np.int64)), K('one',np.array(1.0,np.float32)),
 K('depth',np.array(10,np.int64)), K('vals',np.array([0.0,1.0],np.float32)),
 K('g_s',np.array([0,0],np.int64)), K('g_e',np.array([9,9],np.int64)), K('g_ax',np.array([2,3],np.int64)),
 K('pad',np.array([0,0,0,0,0,0,21,21],np.int64)),
]
nodes = [
 helper.make_node('Slice',['input','c5_s','c5_e','c5_ax'],['ch5']),               # colour-5 mask [1,1,30,30]
 helper.make_node('ReduceSum',['ch5'],['heights'],axes=[2],keepdims=1),           # per-column height [1,1,1,30]
 helper.make_node('Reshape',['heights','shape_col'],['heights_T']),               # [1,1,30,1]
 helper.make_node('Greater',['heights','heights_T'],['is_taller']),               # heights[j] > heights[i]
 helper.make_node('Cast',['is_taller'],['is_taller_f'],to=TensorProto.FLOAT),
 helper.make_node('ReduceSum',['is_taller_f'],['taller_count'],axes=[3],keepdims=1),  # #taller per column
 helper.make_node('Reshape',['taller_count','shape_row'],['taller_row']),
 helper.make_node('Add',['taller_row','one'],['ranks']),                          # rank = #taller + 1
 helper.make_node('Mul',['ranks','ch5'],['colored_grid']),                        # paint rank onto the bars
 helper.make_node('Slice',['colored_grid','g_s','g_e','g_ax'],['crop9']),         # FIX: crop to 9x9
 helper.make_node('Squeeze',['crop9'],['cvs'],axes=[1]),
 helper.make_node('Cast',['cvs'],['cvi'],to=TensorProto.INT64),
 helper.make_node('OneHot',['cvi','depth','vals'],['oh'],axis=1),                 # [1,10,9,9]
 helper.make_node('Pad',['oh','pad'],['output']),                                 # FIX: pad back to 30x30
]
model = helper.make_model(helper.make_graph(nodes,'task010',[x],[y],inits),
                          ir_version=10, opset_imports=[helper.make_opsetid('',12)])
