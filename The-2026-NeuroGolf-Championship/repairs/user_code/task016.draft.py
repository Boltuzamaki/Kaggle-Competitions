# --- Available: onnx, helper, numpy_helper, np, TensorProto. Assign final model to `model`. ---
import numpy as np
F=TensorProto.FLOAT
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
# task016: fixed colour permutation -> one channel Gather (0 params, cheap)
# colour map: 1->5, 2->6, 3->4, 4->3, 5->1, 6->2, 8->9, 9->8
idx=np.array([0, 5, 6, 4, 3, 1, 2, 7, 9, 8],np.int64)
inits=[numpy_helper.from_array(idx,'idx')]
nodes=[helper.make_node('Gather',['input','idx'],['output'],axis=1)]
model=helper.make_model(helper.make_graph(nodes,'task016',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
