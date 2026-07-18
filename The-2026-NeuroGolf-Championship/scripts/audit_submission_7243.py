import os, sys, json, copy, math, csv
sys.path.insert(0, os.path.join("data", "neurogolf_utils"))
import onnx, onnxruntime
import numpy as np
import neurogolf_utils as ngu

SUB_DIR = "submissions/7243"
DATA_DIR = "data"

def load_task(t):
    return json.load(open(os.path.join(DATA_DIR, f"task{t:03d}.json")))

def audit_one(path, t):
    try:
        model = onnx.load(path)
    except Exception as e:
        return None, None, None, f"load-onnx: {str(e)[:200]}"
    san = ngu.sanitize_model(copy.deepcopy(model))
    if san is None:
        return None, None, None, "sanitize failed"
    try:
        onnx.checker.check_model(san, full_check=True)
    except Exception as e:
        return None, None, None, f"checker: {str(e)[:200]}"
    o = onnxruntime.SessionOptions()
    o.enable_profiling = True; o.log_severity_level = 3
    o.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    o.profile_file_prefix = f"audit7243_{t}"
    try:
        s = onnxruntime.InferenceSession(san.SerializeToString(), o)
    except Exception as e:
        return None, None, None, f"session-load: {str(e)[:200]}"
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

def main():
    rows = []
    total = 0.0
    n_ok = 0; n_fail_tasks = 0; n_err = 0
    for t in range(1, 401):
        path = os.path.join(SUB_DIR, f"task{t:03d}.onnx")
        if not os.path.exists(path):
            rows.append([t, "MISSING", "", "", "missing file"])
            n_err += 1
            continue
        nfail, cost, pts, status = audit_one(path, t)
        if status != "ok":
            rows.append([t, nfail, cost, pts, status])
            n_err += 1
            continue
        if nfail and nfail > 0:
            rows.append([t, nfail, cost, pts, "wrong-output"])
            n_fail_tasks += 1
            continue
        rows.append([t, nfail, cost, pts, "ok"])
        total += pts
        n_ok += 1
        if t % 20 == 0:
            print(f"...{t}/400 done, running total={total:.2f}", flush=True)

    with open("submissions/audits/7243_audit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n_fail", "cost", "points", "status"])
        w.writerows(rows)

    print(f"\nOK tasks: {n_ok}, wrong-output tasks: {n_fail_tasks}, error/checker tasks: {n_err}")
    print(f"TOTAL SCORE (ok tasks only, errors/wrong = 0): {total:.2f}")

if __name__ == "__main__":
    main()
