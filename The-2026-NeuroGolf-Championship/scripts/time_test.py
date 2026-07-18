"""Quick benchmark: how fast is one GPU gradient step on a single task?"""
import json, time
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda"
C, H, W = 10, 30, 30

def grid_to_np(grid):
    t = np.zeros((1, C, H, W), dtype=np.float32)
    for r, row in enumerate(grid):
        if r >= H: break
        for c, color in enumerate(row):
            if c >= W: break
            if 0 <= color <= 9:
                t[0, color, r, c] = 1.0
    return t

with open("data/task001.json") as f:
    d = json.load(f)

xs = [torch.tensor(grid_to_np(e["input"]),  device=DEVICE) for e in d["train"]]
ys = [torch.tensor(grid_to_np(e["output"]), device=DEVICE) for e in d["train"]]

conv = nn.Conv2d(C, C, 3, padding=1, bias=False).to(DEVICE)
opt  = torch.optim.Adam(conv.parameters(), lr=0.05)

# Warmup
for _ in range(10):
    opt.zero_grad()
    loss = sum(nn.functional.mse_loss(conv(x), y) for x, y in zip(xs, ys))
    loss.backward(); opt.step()

# Benchmark 1000 steps
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    opt.zero_grad()
    loss = sum(nn.functional.mse_loss(conv(x), y) for x, y in zip(xs, ys))
    loss.backward(); opt.step()
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

steps_per_sec = 1000 / elapsed
print(f"1000 steps in {elapsed:.2f}s  →  {steps_per_sec:.0f} steps/sec")

# Per-approach estimates
for name, restarts, iters in [
    ("gpu_1x1", 6, 1500),
    ("gpu_3x3", 8, 2500),
    ("gpu_5x5", 6, 2500),
]:
    total_steps = restarts * iters
    per_task_s  = total_steps / steps_per_sec
    total_400   = 400 * per_task_s
    print(f"  {name}: {per_task_s:.1f}s/task  →  {total_400/60:.0f} min for 400 tasks (if all unsolvable)")
