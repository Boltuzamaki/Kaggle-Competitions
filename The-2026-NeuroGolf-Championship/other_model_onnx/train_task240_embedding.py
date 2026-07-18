"""Fit a shared low-dimensional coordinate embedding for task240."""

import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch

from train_task240_lowrank import arrays, OUT


def train(dim, rank, seeds, steps):
    model, place, x_np, y_np = arrays()
    print(f"unique_color_states={len(x_np)}", flush=True)
    x = torch.tensor(x_np, dtype=torch.float32)
    sign = torch.tensor(y_np * 2 - 1, dtype=torch.float32)
    best = None

    def unpack(params):
        emb, a, d, r, c = params
        return emb.T @ a, emb.T @ d, r @ emb, c @ emb

    def forward(params):
        left, right, out_r, out_c = unpack(params)
        coeff = (torch.matmul(x, right) * left.unsqueeze(0)).sum(dim=1)
        kernels = (out_r[:, :, None] * out_c[:, None, :]).reshape(rank, 36)
        return torch.matmul(coeff, kernels).reshape(-1, 6, 6)

    for seed in range(seeds):
        torch.manual_seed(seed)
        emb = torch.nn.Parameter(torch.randn(dim, 6) * 0.4)
        params = [emb,
                  torch.nn.Parameter(torch.randn(dim, rank) * 0.3),
                  torch.nn.Parameter(torch.randn(dim, rank) * 0.3),
                  torch.nn.Parameter(torch.randn(rank, dim) * 0.3),
                  torch.nn.Parameter(torch.randn(rank, dim) * 0.3)]
        opt = torch.optim.Adam(params, lr=0.025)
        for step in range(steps):
            opt.zero_grad()
            score = forward(params)
            margin = sign * score
            loss = torch.relu(1.0 - margin).square().mean()
            loss += 1e-9 * sum(q.square().mean() for q in params)
            loss.backward()
            opt.step()
            if step % 250 == 249 and int((margin <= 0).sum()) == 0:
                break
        with torch.no_grad():
            margin = sign * forward(params)
            bad = int((margin <= 0).sum())
            minimum = float(margin.min())
            print(f"dim={dim} rank={rank} seed={seed} bad={bad} min_margin={minimum:.6g}", flush=True)
            state = [q.detach().cpu().numpy().astype(np.float32) for q in params]
            key = (bad, -minimum)
            if best is None or key < best[0]: best = (key, state)
            if bad == 0 and minimum > 1e-4: break
    return model, place, best


def export(model, place, state):
    emb, left, right, out_r, out_c = state
    compact_place = emb @ place
    del model.graph.node[:]
    model.graph.node.append(helper.make_node(
        "Einsum",
        ["input", "p", "p", "lr", "lc", "rr", "rc", "p", "p"],
        ["output"], equation="nchw,ah,dw,ab,db,be,bf,er,fs->ncrs", name="output"))
    del model.graph.initializer[:]
    model.graph.initializer.extend([
        numpy_helper.from_array(compact_place.astype(np.float32), "p"),
        numpy_helper.from_array(left.astype(np.float32), "lr"),
        numpy_helper.from_array(right.astype(np.float32), "lc"),
        numpy_helper.from_array(out_r.astype(np.float32), "rr"),
        numpy_helper.from_array(out_c.astype(np.float32), "rc"),
    ])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--rank", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    model, place, best = train(args.dim, args.rank, args.seeds, args.steps)
    print("best", best[0])
    export(model, place, best[1])
