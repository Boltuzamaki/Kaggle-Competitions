import sys, os, json, math, csv
sys.path.insert(0, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data\neurogolf_utils")
import onnx, onnxruntime, numpy as np
import neurogolf_utils as ngu

BASE_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\baseline_v22"
TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"
REPAIRS_DIR = os.path.dirname(os.path.abspath(__file__))

REPAIRED = ["task045.onnx", "task127.onnx", "task384.onnx", "task135.onnx",
            "task146.onnx", "task149.onnx", "task240.onnx", "task347.onnx"]

def node_type_summary(model):
    counts = {}
    for n in model.graph.node:
        counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return ";".join(f"{k}:{v}" for k, v in sorted(counts.items()))

def audit_path(path, examples):
    model = onnx.load(path)
    sanitized = ngu.sanitize_model(model)
    if sanitized is None:
        return None
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = "cpprof"
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
    if memory is None or params is None or memory < 0 or params < 0 or n_fail > 0:
        return None
    cost = memory + params
    points = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return dict(n_fail=n_fail, memory=memory, params=params, cost=cost, points=points,
                n_nodes=len(model.graph.node), n_init=len(model.graph.initializer),
                node_types=node_type_summary(model))

def main():
    rows = []
    for t in range(1, 401):
        fn = f"task{t:03d}.onnx"
        path = os.path.join(REPAIRS_DIR, fn) if fn in REPAIRED else os.path.join(BASE_DIR, fn)
        ex = json.load(open(os.path.join(TASK_DIR, fn.replace(".onnx", ".json"))))
        r = audit_path(path, ex)
        if r is None:
            print(f"task{t:03d}: AUDIT FAILED (shouldn't happen, all 400 verified ok already)")
            continue
        r["task"] = t
        rows.append(r)
        if t % 50 == 0:
            print(f"...task{t:03d} done")

    rows.sort(key=lambda r: -r["cost"])
    out_csv = os.path.join(REPAIRS_DIR, "cost_profile.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "cost", "points", "memory", "params", "n_nodes", "n_init", "node_types"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    print(f"\nWrote {out_csv}")
    print("\nTop 30 most expensive tasks (best repair candidates):")
    for r in rows[:30]:
        print(f"  task{r['task']:03d}: cost={r['cost']:>8}  points={r['points']:.3f}  nodes={r['n_nodes']:>3}  types={r['node_types'][:80]}")

if __name__ == "__main__":
    main()
