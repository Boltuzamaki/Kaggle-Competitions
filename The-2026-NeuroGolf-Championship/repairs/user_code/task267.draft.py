import json
import os
import sys

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data", "neurogolf_utils"))
import neurogolf_utils as ngu


class Task267Model(nn.Module):
    def __init__(self):
        super().__init__()
        coords = torch.arange(30, dtype=torch.long)
        self.register_buffer("rows", coords.view(30, 1).expand(30, 30))
        self.register_buffer("cols", coords.view(1, 30).expand(30, 30))

    def forward(self, x):
        grid = torch.argmax(x, dim=1)[0].long()
        target = grid[6, 0]
        valid = (self.rows < 7) & (self.cols < 7)
        marker = (self.rows == 6) & (self.cols == 0)
        foreground = (grid != 0) & valid & (~marker)
        out_grid = torch.where(foreground, target.expand_as(grid), torch.zeros_like(grid))
        oh = torch.nn.functional.one_hot(out_grid, num_classes=10).permute(2, 0, 1).float().unsqueeze(0)
        return oh * valid.float().view(1, 1, 30, 30)


def to_onehot(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def check_torch():
    with open(os.path.join(ROOT, "data", "task267.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    model = Task267Model()
    wrong = 0
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            pred = model(torch.from_numpy(to_onehot(ex["input"]))).detach().numpy()
            expected = ngu.convert_to_numpy(ex)["output"]
            if not np.array_equal((pred > 0).astype(np.float32), expected):
                print(f"{split} {i} failed")
                wrong += 1
    print(f"torch check: {wrong} fail")
    return wrong == 0


def export_onnx():
    path = os.path.join(ROOT, "predicted", "task267.onnx")
    with open(os.path.join(ROOT, "data", "task267.json"), "r", encoding="utf-8") as f:
        dummy_grid = json.load(f)["train"][0]["input"]
    torch.onnx.export(
        Task267Model(),
        torch.from_numpy(to_onehot(dummy_grid)),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=15,
        do_constant_folding=False,
    )
    model = onnx.load(path)
    model.ir_version = 10
    onnx.save(model, path)
    return path


def check_onnx(path):
    with open(os.path.join(ROOT, "data", "task267.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    sess = onnxruntime.InferenceSession(path)
    wrong = 0
    for split in ("train", "test", "arc-gen"):
        for i, ex in enumerate(data[split]):
            b = ngu.convert_to_numpy(ex)
            pred = (sess.run(["output"], {"input": b["input"]})[0] > 0).astype(np.float32)
            if not np.array_equal(pred, b["output"]):
                print(f"{split} {i} failed")
                wrong += 1
    print(f"onnx check: {wrong} fail")
    return wrong == 0


if __name__ == "__main__":
    if not check_torch():
        raise SystemExit(1)
    onnx_path = export_onnx()
    if not check_onnx(onnx_path):
        raise SystemExit(1)
    print(onnx_path)
