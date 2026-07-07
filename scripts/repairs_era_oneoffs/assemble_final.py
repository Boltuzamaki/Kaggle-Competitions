import sys, os, json, math, zipfile
sys.path.insert(0, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data\neurogolf_utils")
import onnx, onnxruntime, numpy as np
import neurogolf_utils as ngu

BASE_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\baseline_v22"
TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"
WORK = os.path.dirname(os.path.abspath(__file__))

# All 8 originally-broken tasks now genuinely fixed (7 via export-bug graph
# surgery, task347 via a from-scratch rebuild after its baseline model was
# found to crash at runtime on every example).
REPAIRED = ["task045.onnx", "task127.onnx", "task384.onnx", "task135.onnx",
            "task146.onnx", "task149.onnx", "task240.onnx", "task347.onnx"]

def audit_path(path, examples):
    try:
        model = onnx.load(path)
        sanitized = ngu.sanitize_model(model)
        if sanitized is None:
            return {"status": "sanitize_fail"}
        options = onnxruntime.SessionOptions()
        options.enable_profiling = True
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.profile_file_prefix = "finalprof"
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
        n_fail = 0
        for ex in examples["train"] + examples["test"] + examples["arc-gen"]:
            bench = ngu.convert_to_numpy(ex)
            if not bench: continue
            try:
                out = ngu.run_network(session, bench["input"])
                if not np.array_equal(out, bench["output"]): n_fail += 1
            except Exception:
                n_fail += 1
        trace_path = session.end_profiling()
        memory, params = ngu.score_network(sanitized, trace_path)
        try: os.remove(trace_path)
        except Exception: pass
        if memory is None or params is None or memory < 0 or params < 0:
            return {"status": "cost_fail", "n_fail": n_fail}
        cost = memory + params
        if n_fail > 0:
            return {"status": "incorrect", "n_fail": n_fail, "cost": cost}
        points = max(1.0, 25.0 - math.log(max(1.0, cost)))
        return {"status": "ok", "cost": cost, "points": points}
    except Exception as e:
        return {"status": "error", "msg": str(e)[:150]}

def main():
    total = 0.0
    bad = []
    files = {}
    for t in range(1, 401):
        fn = f"task{t:03d}.onnx"
        path = os.path.join(WORK, fn) if fn in REPAIRED else os.path.join(BASE_DIR, fn)
        files[fn] = path
        ex = json.load(open(os.path.join(TASK_DIR, fn.replace(".onnx", ".json"))))
        r = audit_path(path, ex)
        if r["status"] == "ok":
            total += r["points"]
        else:
            bad.append((t, r))
        if t % 50 == 0:
            print(f"...task{t:03d} running total={total:.2f}, bad so far={len(bad)}")

    print()
    print(f"TOTAL FINAL SCORE (local, baseline_v22 + 8 repairs, zero placeholders): {total:.2f}")
    print(f"Bad tasks ({len(bad)}):")
    for t, r in bad:
        print(" ", t, r)

    out_zip = os.path.join(WORK, "submission.zip")
    if os.path.exists(out_zip): os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for fn, path in sorted(files.items()):
            z.write(path, fn)
    print(f"Wrote {out_zip}")

if __name__ == "__main__":
    main()
