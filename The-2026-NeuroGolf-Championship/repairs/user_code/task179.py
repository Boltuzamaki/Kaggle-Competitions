# --- Available: onnx, helper, numpy_helper, np, TensorProto. Assign final model to `model`. ---
import numpy as np
F=TensorProto.FLOAT
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
# task179: geometric transform = transpose (grid 3x3). reversed-index Gather / Transpose.
inits=[]
nodes=[helper.make_node('Transpose',['input'],['output'],perm=[0,1,3,2])]
model=helper.make_model(helper.make_graph(nodes,'task179',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
