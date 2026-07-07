#!/usr/bin/env python3
"""
2026 NeuroGolf Championship — Base GPU Solution
GPU: NVIDIA RTX 4060 Laptop (8 GB VRAM)

Scoring:  points = max(1.0, 25.0 - log(memory_bytes + param_count))
          - Identity network  : 0 params, 0 memory  → 25.0 pts
          - 1x1 Conv (100 p)  : 0 memory            → ~20.4 pts
          - 3x3 Conv (900 p)  : 0 memory            → ~18.2 pts

Strategy (tried in order, cheapest first):
  0. Identity           – passthrough, 0 params
  1. Analytical remap   – infer exact 1x1 conv weights from training data
  2. GPU 1x1 conv       – gradient search, 100 params
  3. GPU 3x3 conv       – gradient search, 900 params
  4. GPU 5x5 conv       – gradient search, 2500 params
"""

import json, math, os, sys, zipfile, time, subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime
from onnx import helper, TensorProto, numpy_helper
from tqdm import tqdm

# ── Setup ─────────────────────────────────────────────────────────────────────

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR  = Path("data")
OUT_DIR   = Path("output_onnx")
OUT_DIR.mkdir(exist_ok=True)

C, H, W   = 10, 30, 30          # channels, height, width
IR_VER    = 10
OPSET     = [helper.make_opsetid("", 10)]

print(f"\n{'='*60}")
print(f"  2026 NeuroGolf Championship — Base GPU Solution")
print(f"  Device : {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'})")
print(f"{'='*60}\n")

# ── Grid / tensor utils ───────────────────────────────────────────────────────

def grid_to_np(grid) -> np.ndarray:
    """2-D grid of color ints → float32 [1, 10, 30, 30] one-hot."""
    t = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= H: break
        for c, color in enumerate(row):
            if c >= W: break
            if 0 <= color <= 9:
                t[0, color, r, c] = 1.0
    return t

def np_to_grid(t: np.ndarray):
    """float32 [1, 10, 30, 30] → 2-D grid (threshold > 0)."""
    out = t > 0.0
    grid = []
    for r in range(H):
        row = []
        for c in range(W):
            cols = [ch for ch in range(C) if out[0, ch, r, c]]
            row.append(cols[0] if len(cols) == 1 else (11 if cols else 10))
        while row and row[-1] == 10:
            row.pop()
        grid.append(row)
    while grid and not grid[-1]:
        grid.pop()
    return grid

def grids_eq(a, b) -> bool:
    return len(a) == len(b) and all(ra == rb for ra, rb in zip(a, b))

def skip_grid(grid) -> bool:
    """True if grid exceeds 30×30 (competition ignores these)."""
    if not grid: return True
    return max(len(grid), max((len(r) for r in grid), default=0)) > 30

# ── ONNX model builders ───────────────────────────────────────────────────────

def onnx_identity():
    """Zero-parameter identity: output = input."""
    x = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, C, H, W])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])
    node  = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "g", [x], [y])
    return helper.make_model(graph, ir_version=IR_VER, opset_imports=OPSET)

def onnx_conv(weight_np: np.ndarray, kernel_size: int):
    """Single Conv2d layer (no bias)."""
    pad = kernel_size // 2
    x = helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, C, H, W])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, C, H, W])
    weight_init = numpy_helper.from_array(weight_np.astype(np.float32), name="W")
    node  = helper.make_node("Conv", ["input", "W"], ["output"],
                             kernel_shape=[kernel_size, kernel_size],
                             pads=[pad, pad, pad, pad])
    graph = helper.make_graph([node], "g", [x], [y], [weight_init])
    return helper.make_model(graph, ir_version=IR_VER, opset_imports=OPSET)

# ── Verification ──────────────────────────────────────────────────────────────

_ORT_OPTS = onnxruntime.SessionOptions()
_ORT_OPTS.log_severity_level = 3   # suppress ORT noise
_ORT_OPTS.enable_profiling    = False

def verify(model: onnx.ModelProto, task_data: dict) -> bool:
    """True iff model output matches every example (train + test + arc-gen)."""
    all_ex = (task_data.get("train",   []) +
              task_data.get("test",    []) +
              task_data.get("arc-gen", []))
    try:
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), _ORT_OPTS,
            providers=["CPUExecutionProvider"])
    except Exception:
        return False
    for ex in all_ex:
        if skip_grid(ex["input"]): continue
        try:
            pred = sess.run(["output"], {"input": grid_to_np(ex["input"])})[0]
        except Exception:
            return False
        if not grids_eq(np_to_grid(pred), ex["output"]):
            return False
    return True

# ── Solver 0: Identity ────────────────────────────────────────────────────────

def solve_identity(task_data: dict):
    model = onnx_identity()
    return model if verify(model, task_data) else None

# ── Solver 1: Analytical color remap (1×1 conv) ───────────────────────────────

def solve_color_remap(task_data: dict):
    """
    Derive a 1×1 conv from ALL examples (train + test + arc-gen).
    Fixes the key bug: using only train left arc-gen colors unmapped,
    causing verification to fail on unseen colors.
    """
    # Use every example available to build the complete color mapping
    all_examples = (task_data.get("train",   []) +
                    task_data.get("test",    []) +
                    task_data.get("arc-gen", []))

    mapping = {}
    for ex in all_examples:
        if skip_grid(ex["input"]): continue
        inp, out = ex["input"], ex["output"]
        h = min(len(inp), len(out))
        for r in range(h):
            w = min(len(inp[r]) if r < len(inp) else 0,
                    len(out[r]) if r < len(out) else 0)
            for c in range(w):
                ci, co = inp[r][c], out[r][c]
                if ci in mapping:
                    if mapping[ci] != co:
                        return None   # not a pure color function
                else:
                    mapping[ci] = co
    if not mapping:
        return None

    weight = np.zeros((C, C, 1, 1), dtype=np.float32)
    for ci, co in mapping.items():
        weight[co, ci, 0, 0] = 1.0
    for c in range(C):
        if c not in mapping:
            weight[c, c, 0, 0] = 1.0   # identity for truly unseen colors

    model = onnx_conv(weight, 1)
    return model if verify(model, task_data) else None


# ── Solver 2: Analytical least-squares 1×1 conv ───────────────────────────────

def solve_least_squares(task_data: dict):
    """
    Solve W (10×10) analytically via least squares: W @ x_pixels = y_pixels.
    No gradient descent — exact closed-form solution from all pixel pairs.
    Works for any task where the output is a linear function of the input channels.
    """
    all_examples = (task_data.get("train",   []) +
                    task_data.get("test",    []) +
                    task_data.get("arc-gen", [])[:30])

    X_cols, Y_cols = [], []
    for ex in all_examples:
        if skip_grid(ex["input"]): continue
        x = grid_to_np(ex["input"]).reshape(C, -1)   # [10, H*W]
        y = grid_to_np(ex["output"]).reshape(C, -1)  # [10, H*W]
        X_cols.append(x)
        Y_cols.append(y)

    if not X_cols:
        return None

    X = np.concatenate(X_cols, axis=1)  # [10, total_pixels]
    Y = np.concatenate(Y_cols, axis=1)  # [10, total_pixels]

    # Solve: W @ X ≈ Y  →  W = Y @ pinv(X)
    try:
        W, _, _, _ = np.linalg.lstsq(X.T, Y.T, rcond=None)
        W = W.T  # [10, 10]
    except np.linalg.LinAlgError:
        return None

    weight = W.reshape(C, C, 1, 1).astype(np.float32)
    model  = onnx_conv(weight, 1)
    return model if verify(model, task_data) else None

# ── Solver 2: GPU gradient search ─────────────────────────────────────────────

def _load_examples_as_tensors(task_data: dict, splits, max_gen: int = 30):
    xs, ys = [], []
    for split in splits:
        exs = task_data.get(split, [])
        if split == "arc-gen":
            exs = exs[:max_gen]
        for ex in exs:
            if skip_grid(ex["input"]): continue
            xs.append(torch.tensor(grid_to_np(ex["input"]),  device=DEVICE))
            ys.append(torch.tensor(grid_to_np(ex["output"]), device=DEVICE))
    return xs, ys


def solve_gpu(task_data: dict, kernel_size: int,
              restarts: int = 8, max_iters: int = 2000) -> onnx.ModelProto | None:
    """
    Search for conv weights via GPU gradient descent.
    Uses train + a sample of arc-gen for robustness.
    """
    xs, ys = _load_examples_as_tensors(task_data, ["train", "arc-gen"], max_gen=5)
    if not xs:
        return None

    pad    = kernel_size // 2
    center = kernel_size // 2

    for restart in range(restarts):
        conv = nn.Conv2d(C, C, kernel_size, padding=pad, bias=False).to(DEVICE)

        # Initialization strategy varies per restart
        with torch.no_grad():
            if restart == 0:
                # Start near identity (diagonal weight at center)
                nn.init.zeros_(conv.weight)
                for i in range(C):
                    conv.weight[i, i, center, center] = 1.0
                conv.weight += 0.02 * torch.randn_like(conv.weight)
            elif restart == 1:
                # Near-zero init (good for constant-shift tasks)
                nn.init.zeros_(conv.weight)
                conv.weight += 0.05 * torch.randn_like(conv.weight)
            else:
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
                scale = 0.1 if restart < restarts // 2 else 1.0
                conv.weight *= scale

        opt   = optim.Adam(conv.parameters(), lr=0.05)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=150, factor=0.5, min_lr=1e-4)

        prev_loss = float("inf")
        for step in range(max_iters):
            opt.zero_grad()
            loss = torch.zeros(1, device=DEVICE)
            for x, y in zip(xs, ys):
                pred = conv(x)
                loss += nn.functional.mse_loss(pred, y)
                loss += nn.functional.binary_cross_entropy_with_logits(pred, y)
            loss.backward()
            opt.step()
            sched.step(loss.item())

            if loss.item() < 1e-6:
                break

        w     = conv.weight.detach().cpu().numpy()
        model = onnx_conv(w, kernel_size)
        if verify(model, task_data):
            return model

    return None

# ── Score estimator ────────────────────────────────────────────────────────────

def estimate_score(model: onnx.ModelProto) -> tuple[int, float]:
    params = sum(math.prod(init.dims) for init in model.graph.initializer)
    pts    = max(1.0, 25.0 - math.log(max(1.0, float(params))))
    return params, pts

# ── Main loop ─────────────────────────────────────────────────────────────────

APPROACHES = [
    ("identity",      solve_identity,      {}),
    ("color_remap",   solve_color_remap,   {}),
    ("least_squares", solve_least_squares, {}),
    # GPU approaches as a time-budgeted last resort (skip if already solved above)
    ("gpu_1x1",       solve_gpu,           {"kernel_size": 1, "restarts": 2, "max_iters": 300}),
    ("gpu_3x3",       solve_gpu,           {"kernel_size": 3, "restarts": 2, "max_iters": 300}),
]

def main():
    task_files = sorted(DATA_DIR.glob("task*.json"))
    print(f"Tasks found: {len(task_files)}\n")

    solved         = 0
    total_pts      = 0.0
    approach_tally = {}
    t0             = time.time()

    for task_path in tqdm(task_files, desc="Solving tasks", unit="task"):
        task_num = int(task_path.stem[4:])   # "task042" → 42

        with open(task_path) as f:
            task_data = json.load(f)

        model          = None
        approach_used  = None

        # Check if all examples have same input/output size (required for GPU conv)
        all_ex_check = (task_data.get("train", []) + task_data.get("test", []))
        same_size = all(
            len(e["input"]) == len(e["output"]) and
            (len(e["input"][0]) if e["input"] else 0) == (len(e["output"][0]) if e["output"] else 0)
            for e in all_ex_check if e["input"] and e["output"]
        )

        for name, fn, kwargs in APPROACHES:
            if name.startswith("gpu_") and not same_size:
                continue   # conv can't change grid dimensions
            try:
                model = fn(task_data, **kwargs)
            except Exception as exc:
                tqdm.write(f"  ! {name} task{task_num:03d} error: {exc}")
                model = None
            if model is not None:
                approach_used = name
                break

        if model is not None:
            out_path = OUT_DIR / f"task{task_num:03d}.onnx"
            onnx.save(model, str(out_path))
            params, pts = estimate_score(model)
            solved    += 1
            total_pts += pts
            approach_tally[approach_used] = approach_tally.get(approach_used, 0) + 1
            tqdm.write(
                f"  ✓ task{task_num:03d}  [{approach_used:12s}]  "
                f"params={params:5d}  pts={pts:.2f}"
            )

    elapsed = time.time() - t0

    # ── Build submission zip ─────────────────────────────────────────────────
    zip_path = Path("submission.zip")
    onnx_files = sorted(OUT_DIR.glob("task*.onnx"))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            zf.write(f, f.name)

    print(f"\n{'='*60}")
    print(f"  Solved      : {solved} / {len(task_files)} tasks")
    print(f"  Est. score  : {total_pts:.2f} pts")
    print(f"  Elapsed     : {elapsed:.1f}s")
    print(f"  Zip size    : {zip_path.stat().st_size / 1024:.1f} KB  ({len(onnx_files)} files)")
    print(f"  Breakdown   : {approach_tally}")
    print(f"{'='*60}\n")

    return zip_path


if __name__ == "__main__":
    zip_path = main()

    # ── Submit to Kaggle ─────────────────────────────────────────────────────
    if zip_path.stat().st_size > 0:
        print("Submitting to Kaggle...")
        res = subprocess.run(
            ["python", "-m", "kaggle", "competitions", "submit",
             "neurogolf-2026", "-f", str(zip_path),
             "-m", "v4: fixed W-shadow bug; identity+color_remap+lstsq+gpu"],
            capture_output=True, text=True
        )
        print(res.stdout)
        if res.returncode != 0:
            print("Kaggle CLI error:", res.stderr)
    else:
        print("No solutions found — nothing to submit.")
