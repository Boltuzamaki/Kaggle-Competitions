import onnx
model = onnx.load('repairs/task005.onnx')
ops = [n.op_type for n in model.graph.node]
print("Constant in ops?", 'Constant' in ops)
for n in model.graph.node:
    if n.op_type == 'Constant':
        print("Constant attributes:")
        for attr in n.attribute:
            print(f" - {attr.name}: type={attr.type}")
