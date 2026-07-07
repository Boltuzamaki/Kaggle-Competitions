import ast

with open('predicted/build_onnx_366.py', 'r') as f:
    tree = ast.parse(f.read())

outputs = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'make_node':
        # args[2] is the outputs list
        try:
            out_list = node.args[2]
            for elt in out_list.elts:
                out_name = elt.value
                if out_name in outputs:
                    print('Duplicate:', out_name)
                outputs.add(out_name)
        except:
            pass
