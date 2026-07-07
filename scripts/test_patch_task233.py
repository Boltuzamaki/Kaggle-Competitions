import os, sys, json, copy, math
sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import onnx, onnxruntime
import numpy as np
from onnx import helper, numpy_helper, TensorProto
import neurogolf_utils as ngu

T = 233

def load_task(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))

def audit_one(model, t):
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:300]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True; o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = f"patchtest_{t}"
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"session-load: {str(e)[:300]}"
    d = load_task(t); nfail = 0
    for ex in d["train"] + d["test"] + d["arc-gen"]:
        b = ngu.convert_to_numpy(ex)
        if not b: continue
        try:
            out = ngu.run_network(s, b["input"])
            if not np.array_equal(out, b["output"]):
                nfail += 1
        except Exception:
            nfail += 1
    tp = s.end_profiling()
    mem, par = ngu.score_network(san, tp)
    try: os.remove(tp)
    except Exception: pass
    if mem is None or par is None or mem < 0 or par < 0:
        return nfail, None, None, "cost could not be measured"
    cost = mem + par
    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return nfail, cost, pts, "ok"

# baseline: current repairs/task233.onnx, unmodified
base_model = onnx.load("repairs/task233.onnx")
print("BASELINE (current repairs/task233.onnx):", audit_one(base_model, T))

# patched copy
model = onnx.load("repairs/task233.onnx")

for a in model.graph.node[5].attribute:
    if a.name == "to":
        a.i = TensorProto.UINT8

new_inits = []
for init in model.graph.initializer:
    if init.name == "safe_name_19":
        new_inits.append(numpy_helper.from_array(np.ones((324,), dtype=np.uint8), name="safe_name_19"))
    elif init.name == "safe_name_25":
        new_inits.append(numpy_helper.from_array(np.array(0, dtype=np.uint8), name="safe_name_25"))
    elif init.name == "safe_name_27":
        new_inits.append(numpy_helper.from_array(np.array(0, dtype=np.uint8), name="safe_name_27"))
    else:
        new_inits.append(init)
del model.graph.initializer[:]
model.graph.initializer.extend(new_inits)

n = model.graph.node[50]
assert n.output[0] == "safe_name_107", n.output[0]
n.op_type = "Cast"
del n.input[:]
n.input.extend(["safe_name_106"])
del n.attribute[:]
n.attribute.extend([helper.make_attribute("to", TensorProto.UINT8)])

n = model.graph.node[60]
assert n.output[0] == "safe_name_118", n.output[0]
n.op_type = "Cast"
del n.input[:]
n.input.extend(["safe_name_115"])
del n.attribute[:]
n.attribute.extend([helper.make_attribute("to", TensorProto.BOOL)])

used = set()
for node in model.graph.node:
    used.update(x for x in node.input if x)
kept = [init for init in model.graph.initializer if init.name in used]
del model.graph.initializer[:]
model.graph.initializer.extend(kept)

try:
    onnx.checker.check_model(model)
    print("checker (non-full): PASSED")
except Exception as e:
    print("checker (non-full) FAILED:", e)

os.makedirs("scratch_onnx", exist_ok=True)
onnx.save(model, "scratch_onnx/task233_patched_test.onnx")
print("PATCHED:", audit_one(model, T))
