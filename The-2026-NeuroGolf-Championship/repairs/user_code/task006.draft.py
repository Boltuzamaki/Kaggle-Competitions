# --- task006: output=2 where left half (cols0-2) AND right half (cols4-6) are both non-background.
import numpy as np
F=TensorProto.FLOAT
x=helper.make_tensor_value_info('input',F,[1,10,30,30]); y=helper.make_tensor_value_info('output',F,[1,10,30,30])
def K(n,a): return numpy_helper.from_array(a,n)
inits=[K('ls',np.array([0,0,0],np.int64)),K('le',np.array([1,3,3],np.int64)),K('ax',np.array([1,2,3],np.int64)),
 K('rs',np.array([0,0,4],np.int64)),K('re',np.array([1,3,7],np.int64)),K('one',np.array(1.0,np.float32)),K('two',np.array(2.0,np.float32)),
 K('depth',np.array(10,np.int64)),K('vals',np.array([0.0,1.0],np.float32)),K('pad',np.array([0,0,0,0,0,0,27,27],np.int64))]
nodes=[helper.make_node('Slice',['input','ls','le','ax'],['L0']),helper.make_node('Slice',['input','rs','re','ax'],['R0']),
 helper.make_node('Sub',['one','L0'],['Lnz']),helper.make_node('Sub',['one','R0'],['Rnz']),helper.make_node('Mul',['Lnz','Rnz'],['ov']),
 helper.make_node('Mul',['ov','two'],['cv']),helper.make_node('Squeeze',['cv'],['cvs'],axes=[1]),
 helper.make_node('Cast',['cvs'],['cvi'],to=TensorProto.INT64),helper.make_node('OneHot',['cvi','depth','vals'],['oh'],axis=1),
 helper.make_node('Pad',['oh','pad'],['output'])]
model=helper.make_model(helper.make_graph(nodes,'task006',[x],[y],inits),ir_version=10,opset_imports=[helper.make_opsetid('',12)])
