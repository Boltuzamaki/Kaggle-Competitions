import sys, os, json
sys.path.insert(0, 'webapp')
os.environ['PROJECT_DIR'] = '.'
from app import parse_onnx_to_graph

g = parse_onnx_to_graph('repairs/task005.onnx')
print("Is g valid?", bool(g))
print(json.dumps(g)[:200])
