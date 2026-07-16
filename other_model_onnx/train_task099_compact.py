"""Train a compact color-equivariant separable classifier for task099.

This is not an example lookup: every output channel is obtained from the same
low-rank spatial rule applied to that input color channel.  Output-channel
coefficients let background, outline, and marker channels use different linear
combinations of those shared spatial components.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_data(device: torch.device):
    data = json.loads((ROOT / "data" / "task099.json").read_text())
    examples = data["train"] + data["test"] + data["arc-gen"]
    unique = {}
    for example in examples:
        inp = np.where(np.asarray(example["input"]) > 1, 2, np.asarray(example["input"]))
        out = np.where(np.asarray(example["output"]) > 1, 2, np.asarray(example["output"]))
        unique[(tuple(inp.ravel()), tuple(out.ravel()))] = (inp, out)
    pairs = list(unique.values())
    x = np.zeros((len(pairs), 3, 10, 10), np.float32)
    y = np.zeros_like(x)
    for n, (inp, out) in enumerate(pairs):
        for r, row in enumerate(inp):
            for c, color in enumerate(row):
                x[n, color, r, c] = 1.0
        for r, row in enumerate(out):
            for c, color in enumerate(row):
                y[n, color, r, c] = 1.0
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


class Compact099(torch.nn.Module):
    def __init__(self, rank: int, basis: int):
        super().__init__()
        self.rank = rank
        self.basis = basis
        self.S = torch.nn.Parameter(torch.randn(basis, 10) * 0.35)
        self.qi = torch.nn.Parameter(torch.randn(rank, basis) * 0.35)
        self.sj = torch.nn.Parameter(torch.randn(rank, basis) * 0.35)
        self.pr = torch.nn.Parameter(torch.randn(rank, basis) * 0.35)
        self.qc = torch.nn.Parameter(torch.randn(rank, basis) * 0.35)
        self.coef = torch.nn.Parameter(torch.randn(rank, 3) * 0.35)

    def forward(self, x):
        qi = self.qi @ self.S
        sj = self.sj @ self.S
        pr = self.pr @ self.S
        qc = self.qc @ self.S
        z = torch.einsum("noij,ti->notj", x, qi)
        z = torch.einsum("notj,tj->not", z, sj)
        z = z * self.coef.T.unsqueeze(0)
        return torch.einsum("not,tr,tc->norc", z, pr, qc)


def sign_stats(logits, target):
    signed = torch.where(target > 0.5, logits, -logits)
    nfail = int((signed <= 0).sum().item())
    return nfail, float(signed.min().item()), float(signed.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--basis", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="other_model_onnx/task099_compact.npz")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    x, target = load_data(device)
    model = Compact099(args.rank, args.basis).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.025)

    best = None
    for step in range(args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        signed = torch.where(target > 0.5, logits, -logits)
        # Balance the one positive channel against the nine negative channels.
        pos = torch.nn.functional.softplus(-logits[target > 0.5]).mean()
        neg = torch.nn.functional.softplus(logits[target < 0.5]).mean()
        loss = pos + neg
        if step >= 2500:
            loss = loss + 0.3 * torch.relu(0.35 - signed).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if step % 250 == 0:
            with torch.no_grad():
                logits = model(x)
                stats = sign_stats(logits, target)
            print(step, float(loss.item()), stats, flush=True)
            if best is None or stats < best:
                best = stats
            if stats[0] == 0 and stats[1] >= 0.20:
                break

    with torch.no_grad():
        logits = model(x)
        stats = sign_stats(logits, target)
    arrays = {name: value.detach().cpu().numpy() for name, value in model.named_parameters()}
    arrays["stats"] = np.asarray(stats, np.float64)
    arrays["rank"] = np.asarray(args.rank, np.int64)
    arrays["basis"] = np.asarray(args.basis, np.int64)
    np.savez(ROOT / args.output, **arrays)
    print("saved", ROOT / args.output, "stats", stats, flush=True)


if __name__ == "__main__":
    main()
