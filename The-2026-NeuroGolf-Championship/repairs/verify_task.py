"""Standalone CLI audit — mirrors webapp/app.py::audit() exactly, no Flask/docker needed.

Usage:
    python repairs/verify_task.py <task_num> <path/to/model.onnx>
    python repairs/verify_task.py <task_num> <path/to/user_code.py>   # module must set `model` = onnx.ModelProto

Prints: n_fail, cost, points, and (if failing) the first few mismatching examples.
Exit code 0 = solved (n_fail==0 and cost measurable), else 1.
"""
import sys, os, json, copy, math

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
UTILS = os.path.join(DATA, "neurogolf_utils")
sys.path.insert(0, UTILS)

import numpy as np
import onnx, onnxruntime
import neurogolf_utils as ngu


def load_task(t):
    return json.load(open(os.path.join(DATA, f"task{t:03d}.json")))


def audit(model, t, verbose=True):
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:300]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True
    o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = "cli"
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"load: {str(e)[:300]}"
    d = load_task(t)
    nfail = 0
    fail_examples = []
    all_ex = d["train"] + d["test"] + d["arc-gen"]
    for ex_id, ex in enumerate(all_ex):
        b = ngu.convert_to_numpy(ex)
        if not b:
            continue
        try:
            out = ngu.run_network(s, b["input"])
            if not np.array_equal(out, b["output"]):
                nfail += 1
                if len(fail_examples) < 5:
                    fail_examples.append(ex_id)
        except Exception as e:
            nfail += 1
            if len(fail_examples) < 5:
                fail_examples.append((ex_id, str(e)[:150]))
    tp = s.end_profiling()
    mem, par = ngu.score_network(san, tp)
    try:
        os.remove(tp)
    except Exception:
        pass
    if mem is None or par is None or mem < 0 or par < 0:
        return nfail, None, None, "cost could not be measured"
    cost = mem + par
    pts = max(1.0, 25.0 - math.log(max(1.0, cost)))
    if verbose:
        print(f"task{t:03d}: n_examples={len(all_ex)} n_fail={nfail} cost={cost} points={pts:.3f}")
        if fail_examples:
            print(f"  first failing examples: {fail_examples}")
    return nfail, cost, pts, "ok"


def load_model_from_path(path):
    if path.endswith(".onnx"):
        return onnx.load(path)
    ns = {}
    src = open(path, encoding="utf-8").read()
    exec(compile(src, path, "exec"), ns)
    m = ns.get("model")
    if m is None:
        raise RuntimeError(f"{path} did not set a `model` variable")
    return m


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    t = int(sys.argv[1])
    model = load_model_from_path(sys.argv[2])
    nfail, cost, pts, msg = audit(model, t)
    if nfail is None:
        print(f"INVALID: {msg}")
        sys.exit(1)
    sys.exit(0 if (nfail == 0 and cost is not None) else 1)
