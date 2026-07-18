import sys, os, csv, json, math, time
sys.path.insert(0, r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data\neurogolf_utils")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnx, onnxruntime
from onnx import helper, TensorProto, numpy_helper
import neurogolf_utils as ngu

BASE_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\baseline_v22"
REPAIRS_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\repairs"
TASK_DIR = r"C:\Users\chand\OneDrive\Desktop\get_a_job\kaggle_competitions\The 2026 NeuroGolf Championship\data"

REPAIRED = {"task045.onnx", "task127.onnx", "task384.onnx", "task135.onnx",
            "task146.onnx", "task149.onnx", "task240.onnx", "task347.onnx"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C, H, W = 10, 30, 30
IR_VER = 10
OPSET = [helper.make_opsetid("", 12)]

TOP_N = 40
KERNEL_SIZES = [1, 3, 5]
RESTARTS = 2
MAX_ITERS = 400
ARC_GEN_SAMPLE = 40   # examples used during training loss; final check uses ALL examples

def grid_to_np(grid):
    t = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= H: break
        for c, color in enumerate(row):
            if c >= W: break
            if 0 <= color <= 9:
                t[0, color, r, c] = 1.0
    return t

def onnx_conv(weight_np, kernel_size):
    pad = kernel_size // 2
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, C, H, W])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])
    w_init = numpy_helper.from_array(weight_np.astype(np.float32), name="W")
    node = helper.make_node("Conv", ["input", "W"], ["output"],
                             kernel_shape=[kernel_size, kernel_size], pads=[pad, pad, pad, pad])
    graph = helper.make_graph([node], "g", [x], [y], [w_init])
    return helper.make_model(graph, ir_version=IR_VER, opset_imports=OPSET)

def official_audit(model, examples):
    """Matches the proven-accurate recipe used throughout this project."""
    sanitized = ngu.sanitize_model(model)
    if sanitized is None:
        return None
    try:
        onnx.checker.check_model(sanitized, full_check=True)
    except Exception:
        return None
    options = onnxruntime.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.profile_file_prefix = "trainprof"
    try:
        session = onnxruntime.InferenceSession(sanitized.SerializeToString(), options)
    except Exception:
        return None
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
    return {"cost": cost, "points": max(1.0, 25.0 - math.log(max(1.0, cost)))}

def train_one_kernel(task_data, kernel_size, restarts, max_iters):
    all_ex = task_data["train"] + task_data["test"] + task_data["arc-gen"][:ARC_GEN_SAMPLE]
    xs, ys = [], []
    for ex in all_ex:
        if max(len(ex["input"]), len(ex["input"][0]) if ex["input"] else 0) > 30:
            continue
        xs.append(torch.tensor(grid_to_np(ex["input"]), device=DEVICE))
        ys.append(torch.tensor(grid_to_np(ex["output"]), device=DEVICE))
    if not xs:
        return None
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    pad, center = kernel_size // 2, kernel_size // 2

    for restart in range(restarts):
        conv = nn.Conv2d(C, C, kernel_size, padding=pad, bias=False).to(DEVICE)
        with torch.no_grad():
            if restart == 0:
                nn.init.zeros_(conv.weight)
                for i in range(C):
                    conv.weight[i, i, center, center] = 1.0
                conv.weight += 0.02 * torch.randn_like(conv.weight)
            else:
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
                conv.weight *= 0.3

        opt = optim.Adam(conv.parameters(), lr=0.05)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=60, factor=0.5, min_lr=1e-4)
        best_loss = float("inf")
        stall = 0
        for step in range(max_iters):
            opt.zero_grad()
            pred = conv(X)
            loss = nn.functional.mse_loss(pred, Y) + nn.functional.binary_cross_entropy_with_logits(pred, Y)
            loss.backward()
            opt.step()
            sched.step(loss.item())
            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                stall = 0
            else:
                stall += 1
            if loss.item() < 1e-7:
                break
            if stall > 150:
                break

        w = conv.weight.detach().cpu().numpy()
        model = onnx_conv(w, kernel_size)
        yield model

def main():
    rows = list(csv.DictReader(open(os.path.join(REPAIRS_DIR, "cost_profile.csv"))))
    candidates = []
    for r in rows:
        t = int(r["task"])
        fn = f"task{t:03d}.onnx"
        if fn == "task240.onnx":
            continue  # already optimized this session, cost is now tiny
        d = json.load(open(os.path.join(TASK_DIR, f"task{t:03d}.json")))
        all_ex = d["train"] + d["test"] + d["arc-gen"]
        same_size = all(len(e["input"]) == len(e["output"]) and len(e["input"][0]) == len(e["output"][0]) for e in all_ex)
        if same_size:
            candidates.append((t, float(r["cost"]), d))
    candidates.sort(key=lambda x: -x[1])
    candidates = candidates[:TOP_N]
    print(f"Attempting training on top {len(candidates)} same-size candidates by cost.")

    improved = []
    t0 = time.time()
    for t, base_cost, task_data in candidates:
        found = None
        for ks in KERNEL_SIZES:
            for model in train_one_kernel(task_data, ks, RESTARTS, MAX_ITERS):
                result = official_audit(model, task_data)
                if result and result["cost"] < base_cost:
                    found = (model, result, ks)
                    break
            if found:
                break
        elapsed = time.time() - t0
        if found:
            model, result, ks = found
            out_path = os.path.join(REPAIRS_DIR, f"task{t:03d}.onnx")
            onnx.save(model, out_path)
            gain_pts = max(1.0, 25.0 - math.log(max(1.0, base_cost))) * -1  # placeholder, recompute below
            base_pts = max(1.0, 25.0 - math.log(max(1.0, base_cost)))
            gain = result["points"] - base_pts
            improved.append((t, base_cost, result["cost"], gain, ks))
            print(f"[{elapsed:.0f}s] task{t:03d}: TRAINED kernel={ks}x{ks} cost {base_cost:.0f} -> {result['cost']:.0f} (+{gain:.4f} pts)")
        else:
            print(f"[{elapsed:.0f}s] task{t:03d}: no trained net beat cost={base_cost:.0f}, keeping existing")

    print(f"\nDone in {time.time()-t0:.0f}s. Improved {len(improved)}/{len(candidates)} tasks.")
    print(f"Total gain: {sum(g for _,_,_,g,_ in improved):.4f}")
    for t, bc, nc, g, ks in improved:
        print(f"  task{t:03d}: {bc:.0f} -> {nc:.0f} (+{g:.4f}, kernel {ks}x{ks})")

if __name__ == "__main__":
    main()
