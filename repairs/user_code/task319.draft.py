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


class Task319Model(nn.Module):
    def __init__(self):
        super().__init__()
        coords30 = torch.arange(30, dtype=torch.long)
        coords10 = torch.arange(10, dtype=torch.long)
        coords20 = torch.arange(20, dtype=torch.long)
        self.register_buffer("rows30", coords30.view(30, 1).expand(30, 30))
        self.register_buffer("cols30", coords30.view(1, 30).expand(30, 30))
        self.register_buffer("rows10", coords10.view(10, 1).expand(10, 10))
        self.register_buffer("cols10", coords10.view(1, 10).expand(10, 10))
        self.register_buffer("off_r", coords20.view(20, 1, 1).expand(20, 20, 1).reshape(400, 1, 1))
        self.register_buffer("off_c", coords20.view(1, 20, 1).expand(20, 20, 1).reshape(400, 1, 1))
        self.register_buffer("cand_colors", torch.arange(1, 10, dtype=torch.long))

    def bbox_for(self, mask):
        any_cell = mask.any()
        r_min = torch.min(torch.where(mask, self.rows30, torch.full_like(self.rows30, 30)))
        c_min = torch.min(torch.where(mask, self.cols30, torch.full_like(self.cols30, 30)))
        r_max = torch.max(torch.where(mask, self.rows30, torch.zeros_like(self.rows30))) + 1
        c_max = torch.max(torch.where(mask, self.cols30, torch.zeros_like(self.cols30))) + 1
        r_min = torch.where(any_cell, r_min, torch.zeros_like(r_min))
        c_min = torch.where(any_cell, c_min, torch.zeros_like(c_min))
        r_max = torch.where(any_cell, r_max, torch.zeros_like(r_max))
        c_max = torch.where(any_cell, c_max, torch.zeros_like(c_max))
        return r_min, r_max, c_min, c_max, any_cell

    def crop10(self, mask, r_min, c_min, h, w):
        sr = torch.clamp(r_min + self.rows10, 0, 29)
        sc = torch.clamp(c_min + self.cols10, 0, 29)
        sampled = torch.gather(mask.reshape(900), 0, (sr * 30 + sc).reshape(100)).reshape(10, 10)
        return sampled & (self.rows10 < h) & (self.cols10 < w)

    def contains_scaled_crop(self, cand, ch, cw, other, oh, ow):
        # Does any crop of the 2x-expanded candidate mask exactly equal other?
        rr = self.rows10.unsqueeze(0)
        cc = self.cols10.unsqueeze(0)
        in_other = (rr < oh) & (cc < ow)
        sr2 = self.off_r + rr
        sc2 = self.off_c + cc
        inside_scaled = (sr2 < ch * 2) & (sc2 < cw * 2)
        src_r = torch.clamp(torch.div(sr2, 2, rounding_mode="floor"), 0, 9)
        src_c = torch.clamp(torch.div(sc2, 2, rounding_mode="floor"), 0, 9)
        scaled_val = torch.gather(cand.reshape(100), 0, (src_r * 10 + src_c).reshape(40000)).reshape(400, 10, 10)
        other_val = other.unsqueeze(0).expand(400, 10, 10)
        ok = (~in_other) | (inside_scaled & (scaled_val == other_val))
        return ok.reshape(400, 100).all(dim=1).any()

    def forward(self, x):
        x0 = x[0]
        valid = x0.sum(dim=0) > 0
        counts = (x0 * valid.float().unsqueeze(0)).reshape(10, 900).sum(dim=1)
        bg = torch.argmax(counts).long()
        grid = torch.argmax(x, dim=1)[0].long()

        masks = []
        hs = []
        ws = []
        has = []
        for color in range(1, 10):
            mask = (grid == color) & valid
            r1, r2, c1, c2, any_cell = self.bbox_for(mask)
            h = r2 - r1
            w = c2 - c1
            masks.append(self.crop10(mask, r1, c1, h, w))
            hs.append(h)
            ws.append(w)
            has.append(any_cell & (torch.tensor(color, dtype=torch.long, device=x.device) != bg))

        scores = []
        for i in range(9):
            relation = torch.tensor(False, device=x.device)
            for j in range(9):
                if i == j:
                    continue
                contains = self.contains_scaled_crop(masks[i], hs[i], ws[i], masks[j], hs[j], ws[j])
                relation = relation | (has[j] & contains)
            area = hs[i] * ws[i]
            valid_candidate = has[i] & relation
            color_penalty = torch.tensor(i, dtype=torch.long, device=x.device)
            scores.append(torch.where(valid_candidate, 10000 - area * 10 - color_penalty, torch.zeros_like(area)))
        score_t = torch.stack(scores)
        idx = torch.argmax(score_t)

        selected_color = idx + 1
        mask_stack = torch.stack(masks).reshape(9, 100)
        selected_mask = torch.gather(mask_stack, 0, idx.view(1, 1).expand(1, 100)).reshape(10, 10)
        h_t = torch.stack(hs)
        w_t = torch.stack(ws)
        sel_h = torch.gather(h_t, 0, idx.view(1))[0]
        sel_w = torch.gather(w_t, 0, idx.view(1))[0]

        out_valid = (self.rows30 < sel_h) & (self.cols30 < sel_w)
        small_r = torch.clamp(self.rows30, 0, 9)
        small_c = torch.clamp(self.cols30, 0, 9)
        fg = torch.gather(selected_mask.reshape(100), 0, (small_r * 10 + small_c).reshape(900)).reshape(30, 30)
        out_grid = torch.where(fg & out_valid, selected_color.expand_as(grid), bg.expand_as(grid))
        oh = torch.nn.functional.one_hot(out_grid, num_classes=10).permute(2, 0, 1).float().unsqueeze(0)
        return oh * out_valid.float().view(1, 1, 30, 30)


def to_onehot(grid):
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            arr[0, color, r, c] = 1.0
    return arr


def check_torch():
    with open(os.path.join(ROOT, "data", "task319.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    model = Task319Model()
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
    path = os.path.join(ROOT, "predicted", "task319.onnx")
    with open(os.path.join(ROOT, "data", "task319.json"), "r", encoding="utf-8") as f:
        dummy_grid = json.load(f)["train"][0]["input"]
    torch.onnx.export(
        Task319Model(),
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
    with open(os.path.join(ROOT, "data", "task319.json"), "r", encoding="utf-8") as f:
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
