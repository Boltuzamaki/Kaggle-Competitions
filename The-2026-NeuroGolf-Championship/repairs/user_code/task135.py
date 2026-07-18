# --- Available: onnx, helper, numpy_helper, np, TensorProto. Assign final model to `model`. ---
import numpy as np
F=TensorProto.FLOAT
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
# task135: fixed crop input[0:3, 6:9] then Pad back to 30x30.
def K(n,a): return numpy_helper.from_array(a,n)
inits=[K('s',np.array([0,6],np.int64)),K('e',np.array([3,9],np.int64)),K('ax',np.array([2,3],np.int64)),K('pad',np.array([0,0,0,0,0,0,27,27],np.int64))]
nodes=[helper.make_node('Slice',['input','s','e','ax'],['c']),helper.make_node('Pad',['c','pad'],['output'])]
model=helper.make_model(helper.make_graph(nodes,'task135',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
