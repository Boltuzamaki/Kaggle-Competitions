"""Build the verified 300-cost task099 context-cell model as two ONNX Einsums."""

from pathlib import Path
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "other_model_onnx" / "task099_ctxcell_r5q3d2_refine.npz"
OUT = ROOT / "other_model_onnx" / "task099.onnx"

d = np.load(NPZ)
names = ("S", "ck", "ci", "cj", "co", "cc", "cd", "qa", "qb", "qr", "qc")
initializers = [numpy_helper.from_array(d[name].astype(np.float32), name) for name in names]

context = helper.make_node(
    "Einsum", ["input", "ck", "ci", "S", "cj", "S"], ["context"],
    name="context", equation="nkij,dk,dx,xi,dy,yj->nd",
)
output = helper.make_node(
    "Einsum",
    ["input", "qa", "S", "qb", "S", "context", "cd", "co",
     "input", "cc", "qr", "S", "qc", "S"],
    ["output"], name="output",
    equation="noab,tx,xa,ty,yb,nd,td,to,nkrc,tk,tz,zr,tw,wc->norc",
)

shape = [1, 10, 30, 30]
graph = helper.make_graph(
    [context, output], "task099_context_cell",
    [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
    [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    initializers,
    value_info=[helper.make_tensor_value_info("context", TensorProto.FLOAT, [1, 2])],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
model.ir_version = 8
onnx.checker.check_model(model, full_check=True)
onnx.save(model, OUT)
print(OUT)
