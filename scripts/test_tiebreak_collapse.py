import os, sys, json, copy, math
sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import onnx, onnxruntime
import numpy as np
from onnx import numpy_helper
import neurogolf_utils as ngu

CASES = {
    101: ["safe_name_21", "safe_name_22"],
    178: ["desc_weights"],
    218: ["rank_scores"],
    247: ["score_weights"],
}

def load_task(t):
    return json.load(open(os.path.join("data", f"task{t:03d}.json")))

def audit_one(model, t, prefix):
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:250]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True; o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = prefix
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"session-load: {str(e)[:250]}"
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

for t, init_names in CASES.items():
    src = f"repairs/task{t:03d}.onnx"
    base = onnx.load(src)
    base_result = audit_one(base, t, f"base_{t}")
    print(f"task{t:03d} BASELINE: {base_result}")

    m = onnx.load(src)
    new_inits = []
    for init in m.graph.initializer:
        if init.name in init_names:
            arr = numpy_helper.to_array(init)
            scalar = np.array(arr.max(), dtype=arr.dtype)
            new_inits.append(numpy_helper.from_array(scalar, name=init.name))
        else:
            new_inits.append(init)
    del m.graph.initializer[:]
    m.graph.initializer.extend(new_inits)
    try:
        onnx.checker.check_model(m)
    except Exception as e:
        print(f"task{t:03d} PATCHED : checker failed: {str(e)[:200]}")
        continue
    result = audit_one(m, t, f"patched_{t}")
    print(f"task{t:03d} PATCHED : {result}")
    if result[0] == 0 and base_result[0] == 0 and result[2] and base_result[2] and result[2] > base_result[2]:
        onnx.save(m, f"scratch_onnx/task{t:03d}_tiebreak_collapsed.onnx")
        print(f"  -> WIN, saved to scratch_onnx/task{t:03d}_tiebreak_collapsed.onnx")
    print()
