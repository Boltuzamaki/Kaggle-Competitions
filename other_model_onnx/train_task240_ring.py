"""Fit/export a rank-2 tensor-ring implementation of task240's class map."""

import argparse
import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch

from train_task240_lowrank import arrays, OUT


def train(rank, seeds, steps):
    _, _, x_np, y_np = arrays()
    print(f"unique_color_states={len(x_np)}", flush=True)
    x = torch.tensor(x_np, dtype=torch.float32)
    sign = torch.tensor(y_np * 2 - 1, dtype=torch.float32)
    best = None
    for seed in range(seeds):
        torch.manual_seed(seed)
        cores = [torch.nn.Parameter(torch.randn(6, rank, rank) * 0.25) for _ in range(4)]
        opt = torch.optim.Adam(cores, lr=0.03)
        for step in range(steps):
            opt.zero_grad()
            tensor = torch.einsum("aij,djk,ekl,fli->adef", *cores)
            score = torch.einsum("zad,adef->zef", x, tensor)
            margin = sign * score
            loss = torch.relu(1.0 - margin).square().mean()
            loss = loss + 1e-8 * sum(q.square().mean() for q in cores)
            loss.backward()
            opt.step()
            if step % 250 == 249 and int((margin <= 0).sum()) == 0:
                break
        with torch.no_grad():
            tensor = torch.einsum("aij,djk,ekl,fli->adef", *cores)
            score = torch.einsum("zad,adef->zef", x, tensor)
            margin = sign * score
            bad = int((margin <= 0).sum())
            minimum = float(margin.min())
            print(f"ring={rank} seed={seed} bad={bad} min_margin={minimum:.6g}", flush=True)
            state = [q.detach().cpu().numpy().astype(np.float32) for q in cores]
            key = (bad, -minimum)
            if best is None or key < best[0]: best = (key, state)
            if bad == 0 and minimum > 1e-4: break
    return best


def export(cores):
    model, place, _, _ = arrays()
    del model.graph.node[:]
    model.graph.node.append(helper.make_node(
        "Einsum",
        ["input", "place", "place", "g1", "g2", "g3", "g4", "place", "place"],
        ["output"],
        equation="nchw,ah,dw,aij,djk,ekl,fli,er,fs->ncrs",
        name="output",
    ))
    del model.graph.initializer[:]
    model.graph.initializer.append(numpy_helper.from_array(place.astype(np.float32), "place"))
    for i, core in enumerate(cores, 1):
        model.graph.initializer.append(numpy_helper.from_array(core, f"g{i}"))
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    best = train(args.rank, args.seeds, args.steps)
    print("best", best[0])
    export(best[1])
