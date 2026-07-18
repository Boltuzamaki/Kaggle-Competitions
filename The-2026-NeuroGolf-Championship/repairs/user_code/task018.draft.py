# Build an ONNX model for task 18. Assign the final onnx.ModelProto to `model`.
# Available names: onnx, helper, numpy_helper, np, TensorProto, T (=task number)
# Contract: input tensor 'input' [1,10,30,30] one-hot; produce 'output' [1,10,30,30].
x = helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1,10,30,30])
y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1,10,30,30])
node = helper.make_node('Identity', ['input'], ['output'])
model = helper.make_model(helper.make_graph([node],'g',[x],[y],[]),
                          ir_version=10, opset_imports=[helper.make_opsetid('',12)])
