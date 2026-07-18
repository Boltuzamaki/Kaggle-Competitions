"""Train a compact color-equivariant quadratic tensor model for task099."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_data(device):
    data = json.loads((ROOT / "data" / "task099.json").read_text())
    examples = data["train"] + data["test"] + data["arc-gen"]
    x = np.zeros((len(examples), 10, 10, 10), np.float32)
    y = np.zeros_like(x)
    for n, example in enumerate(examples):
        a, b = np.asarray(example["input"]), np.asarray(example["output"])
        x[n] = np.eye(10, dtype=np.float32)[a].transpose(2, 0, 1)
        y[n] = np.eye(10, dtype=np.float32)[b].transpose(2, 0, 1)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


class Quadratic(torch.nn.Module):
    def __init__(self, rank, basis):
        super().__init__()
        self.rank, self.basis = rank, basis
        # A DCT-like starting basis gives every position a distinct compact code.
        p = np.arange(30, dtype=np.float32)
        S = np.stack([np.cos(np.pi * (p + .5) * k / 30) for k in range(basis)])
        self.S = torch.nn.Parameter(torch.from_numpy(S))
        def param(*shape, scale=.3):
            return torch.nn.Parameter(torch.randn(*shape) * scale)
        self.co = param(rank, 10)
        self.ck = param(rank, 10)
        self.qa = param(rank, basis)
        self.qb = param(rank, basis)
        self.qi = param(rank, basis)
        self.qj = param(rank, basis)
        self.qr = param(rank, basis)
        self.qc = param(rank, basis)

    def forward(self, x):
        S = self.S[:, :10]
        fa, fb, fi, fj, fr, fc = [q @ S for q in
                                  (self.qa, self.qb, self.qi, self.qj, self.qr, self.qc)]
        # First view retains the color index that becomes the output color.
        first = torch.einsum("noab,ta,tb->nto", x, fa, fb)
        # Second view summarizes structural context (especially outline channel 1).
        context = torch.einsum("nkij,tk,ti,tj->nt", x, self.ck, fi, fj)
        return torch.einsum("nto,nt,to,tr,tc->norc", first, context, self.co, fr, fc)


def stats(logits, target):
    wrong = (logits > 0) != (target > .5)
    signed = torch.where(target > .5, logits, -logits)
    return int(wrong.sum()), float(signed[target > .5].min()), float(signed.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=5)
    ap.add_argument("--basis", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="other_model_onnx/task099_quadratic.npz")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = load_data(device)
    model = Quadratic(args.rank, args.basis).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.015)
    best = None
    for step in range(args.steps + 1):
        ix = torch.randint(len(x), (min(args.batch, len(x)),), device=device)
        z = model(x[ix]); target = y[ix]
        signed = torch.where(target > .5, z, -z)
        # Equal weighting of positive and negative cells prevents the sparse target
        # from winning by simply predicting every channel as absent.
        loss = torch.nn.functional.softplus(-z[target > .5]).mean()
        present = x[ix].sum(dim=(-1, -2), keepdim=True) > 0
        active_negative = (target < .5) & present
        loss = loss + torch.nn.functional.softplus(z[active_negative]).mean()
        if step >= 3000:
            loss = loss + torch.relu(.2 - signed).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 30); opt.step()
        if step % 250 == 0:
            with torch.no_grad(): result = stats(model(x), y)
            print(step, float(loss), result, flush=True)
            if best is None or result[0] < best[0]:
                best = result
                arrays = {n: p.detach().cpu().numpy() for n, p in model.named_parameters()}
                arrays["stats"] = np.asarray(result); arrays["rank"] = args.rank; arrays["basis"] = args.basis
                np.savez(ROOT / args.output, **arrays)
            if result[0] == 0 and result[1] >= .1:
                break
    print("best", best, "saved", ROOT / args.output, flush=True)


if __name__ == "__main__":
    main()
