"""Compress task099's exact linear rule to smaller shared spatial ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]


def load_data(device, unique=False):
    data = json.loads((ROOT / "data" / "task099.json").read_text())
    examples = data["train"] + data["test"] + data["arc-gen"]
    if unique:
        pairs = {}
        for example in examples:
            inp = np.where(np.asarray(example["input"]) > 1, 2, np.asarray(example["input"]))
            out = np.where(np.asarray(example["output"]) > 1, 2, np.asarray(example["output"]))
            pairs[(tuple(inp.ravel()), tuple(out.ravel()))] = (inp, out)
        examples = [{"input": a.tolist(), "output": b.tolist()} for a, b in pairs.values()]
    x = np.zeros((len(examples), 10, 10, 10), np.float32)
    y = np.zeros_like(x)
    for n, example in enumerate(examples):
        for r, row in enumerate(example["input"]):
            for c, color in enumerate(row):
                x[n, color, r, c] = 1
        for r, row in enumerate(example["output"]):
            for c, color in enumerate(row):
                y[n, color, r, c] = 1
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


class DenseCompact(torch.nn.Module):
    def __init__(self, rank, basis, channel_rank, color_factor=0):
        super().__init__()
        source = onnx.load(ROOT / "repairs" / "task099.onnx")
        old = {x.name: numpy_helper.to_array(x).copy() for x in source.graph.initializer}
        full = [old[name] @ old["S"][:, :10] for name in ("cqi", "csj", "cpr", "cqc")]
        importance = np.linalg.norm(old["coef"], axis=1)
        for factor in full:
            importance *= np.linalg.norm(factor, axis=1)
        selected = np.argsort(importance)[-rank:]
        stack = np.concatenate([factor[selected] for factor in full], axis=0)
        _, _, vh = np.linalg.svd(stack, full_matrices=False)
        shared = vh[:basis].astype(np.float32)
        reduced = [(factor[selected] @ shared.T).astype(np.float32) for factor in full]

        uc, sc, vhc = np.linalg.svd(old["C"].reshape(2, 100), full_matrices=False)
        compact_c = vhc[:channel_rank].reshape(channel_rank, 10, 10).astype(np.float32)
        compact_coef = (old["coef"][selected] @ uc[:, :channel_rank] * sc[:channel_rank]).astype(np.float32)
        self.channel_rank = channel_rank
        self.color_factor = color_factor
        if color_factor:
            angles = 2 * np.pi * np.arange(10, dtype=np.float32) / 10
            columns = [np.ones(10, np.float32), np.cos(angles), np.sin(angles)]
            while len(columns) < color_factor:
                frequency = (len(columns) + 1) // 2
                columns.append(np.cos(frequency * angles) if len(columns) % 2 else np.sin(frequency * angles))
            u = np.stack(columns[:color_factor], axis=1).astype(np.float32)
            v = u.copy()
            atoms = np.stack([np.outer(u[:, z], v[:, z]).ravel() for z in range(color_factor)], axis=1)
            a = np.stack([np.linalg.lstsq(atoms, compact_c[s].ravel(), rcond=None)[0] for s in range(channel_rank)])
            self.color_scale = torch.nn.Parameter(torch.from_numpy(a.astype(np.float32)))
            self.color_out = torch.nn.Parameter(torch.from_numpy(u))
            self.color_in = torch.nn.Parameter(torch.from_numpy(v))
        else:
            self.C = torch.nn.Parameter(torch.from_numpy(compact_c))
        self.coef = torch.nn.Parameter(torch.from_numpy(compact_coef))
        self.S = torch.nn.Parameter(torch.from_numpy(shared))
        self.qi = torch.nn.Parameter(torch.from_numpy(reduced[0]))
        self.sj = torch.nn.Parameter(torch.from_numpy(reduced[1]))
        self.pr = torch.nn.Parameter(torch.from_numpy(reduced[2]))
        self.qc = torch.nn.Parameter(torch.from_numpy(reduced[3]))

    def forward(self, x):
        qi, sj, pr, qc = [factor @ self.S for factor in (self.qi, self.sj, self.pr, self.qc)]
        n = x.shape[0]
        # Channel transform: [N,I,J,K] @ [K,S*O].
        if self.color_factor:
            color = torch.einsum("su,ou,ku->sok", self.color_scale, self.color_out, self.color_in)
        else:
            color = self.C
        xc = x.permute(0, 2, 3, 1).reshape(-1, 10) @ color.permute(2, 0, 1).reshape(10, 10 * self.channel_rank)
        xc = xc.reshape(n, 10, 10, self.channel_rank, 10).permute(0, 3, 4, 1, 2)
        # Left/right separable spatial projections.
        z = torch.matmul(qi, xc)
        z = (z * sj.reshape(1, 1, 1, sj.shape[0], 10)).sum(-1)
        z = (z * self.coef.T.reshape(1, self.channel_rank, 1, -1)).sum(1)
        spatial = (pr[:, :, None] * qc[:, None, :]).reshape(pr.shape[0], 100)
        return (z @ spatial).reshape(n, 10, 10, 10)


def stats(logits, target):
    prediction = logits > 0
    errors = int((prediction != (target > 0.5)).sum().item())
    signed = torch.where(target > 0.5, logits, -logits)
    return errors, float(signed.min().item()), float(signed.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--basis", type=int, default=2)
    ap.add_argument("--channel-rank", type=int, default=2)
    ap.add_argument("--color-factor", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--freeze-c", action="store_true")
    ap.add_argument("--freeze-spatial", action="store_true")
    ap.add_argument("--output", default="other_model_onnx/task099_dense_compact.npz")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    x, target = load_data(device, args.unique)
    model = DenseCompact(args.rank, args.basis, args.channel_rank, args.color_factor).to(device)
    if args.freeze_c:
        if hasattr(model, "C"):
            model.C.requires_grad_(False)
    if args.freeze_spatial:
        for name in ("coef", "S", "qi", "sj", "pr", "qc"):
            getattr(model, name).requires_grad_(False)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.012)

    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        if args.batch and args.batch < x.shape[0]:
            index = torch.randint(x.shape[0], (args.batch,), device=device)
            xb, yb = x[index], target[index]
        else:
            xb, yb = x, target
        logits = model(xb)
        signed = torch.where(yb > 0.5, logits, -logits)
        pos = torch.nn.functional.softplus(-logits[yb > 0.5]).mean()
        neg = torch.nn.functional.softplus(logits[yb < 0.5]).mean()
        loss = pos + neg
        if step > 2000:
            loss = loss + 0.5 * torch.relu(0.25 - signed).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
        opt.step()
        if step % 200 == 0:
            with torch.no_grad():
                result = stats(model(x), target)
            print(step, float(loss.item()), result, flush=True)
            if result[0] == 0 and result[1] >= 0.12:
                break

    with torch.no_grad():
        result = stats(model(x), target)
    arrays = {name: value.detach().cpu().numpy() for name, value in model.named_parameters()}
    arrays["stats"] = np.asarray(result, np.float64)
    arrays["rank"] = np.asarray(args.rank, np.int64)
    arrays["basis"] = np.asarray(args.basis, np.int64)
    np.savez(ROOT / args.output, **arrays)
    print("saved", ROOT / args.output, result, flush=True)


if __name__ == "__main__":
    main()
