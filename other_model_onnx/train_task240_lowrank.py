"""Search for a lower sign-rank task240 Einsum, then export the best candidate.

The task is color-linear: after grouping spatial positions into six symmetry
classes, the same 6x6 -> 6x6 transform is applied independently to every
color.  The competition thresholds outputs at zero, so the relevant object is
the sign rank of that transform on the complete task domain, not its ordinary
least-squares rank.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repairs" / "task240.onnx"
DATA = ROOT / "data" / "task240.json"
OUT = ROOT / "other_model_onnx" / "task240_lowrank.onnx"


def arrays():
    model = onnx.load(SOURCE)
    place = numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == "place"))
    raw = json.loads(DATA.read_text())
    xs, ys = [], []
    reps = [1, 3, 5, 7, 9, 0]
    for ex in raw["train"] + raw["test"] + raw["arc-gen"]:
        inp = np.zeros((10, 30, 30), np.float32)
        out = np.zeros((10, 30, 30), np.float32)
        for name, dst in (("input", inp), ("output", out)):
            for r, row in enumerate(ex[name]):
                for c, color in enumerate(row):
                    dst[color, r, c] = 1
        x = np.einsum("ah,chw,dw->cad", place, inp, place)
        y = out[:, reps][:, :, reps]
        # Every member of a symmetry class must have the representative value.
        reconstructed = np.einsum("cef,er,fs->crs", y, place, place)
        assert np.array_equal(reconstructed, out)
        xs.append(x)
        ys.append(y)
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # Color channels are independent.  Collapse repeated per-color class
    # states (especially absent colors) before optimization.
    unique = {}
    for x, y in zip(xs.reshape(-1, 6, 6), ys.reshape(-1, 6, 6)):
        unique.setdefault(x.tobytes() + y.tobytes(), (x, y))
    ux = np.asarray([v[0] for v in unique.values()])
    uy = np.asarray([v[1] for v in unique.values()])
    return model, place, ux, uy


def train(rank, seeds, steps, share, init_current=False, neg_margin=0.0, neg_weight=1.0,
          anchor=0.0, resume=None, learning_rate=None, explicit_omit=None):
    source_model, _, x_np, y_np = arrays()
    print(f"unique_color_states={len(x_np)}", flush=True)
    x = torch.tensor(x_np, dtype=torch.float32)
    positive = torch.tensor(y_np > 0.5)
    best = None
    def unpack(factors):
        if share == "input":
            return factors[0], factors[0], factors[1], factors[2]
        if share == "output":
            return factors[0], factors[1], factors[2], factors[2]
        if share == "both":
            return factors[0], factors[0], factors[1], factors[1]
        return factors
    def forward(factors):
        left, right, out_r, out_c = unpack(factors)
        projected = torch.matmul(x, right)  # [state, input_row, rank]
        coeff = (projected * left.unsqueeze(0)).sum(dim=1)
        kernels = (out_r[:, :, None] * out_c[:, None, :]).reshape(rank, 36)
        return torch.matmul(coeff, kernels).reshape(-1, 6, 6)
    for seed in range(seeds):
        torch.manual_seed(seed)
        scale = 0.35
        if resume and share == "none":
            resumed = onnx.load(resume)
            vals = {q.name: numpy_helper.to_array(q) for q in resumed.graph.initializer}
            initial = [vals["lr"], vals["lc"], vals["rr"], vals["rc"]]
            all_factors = [torch.nn.Parameter(torch.tensor(q, dtype=torch.float32)) for q in initial]
        elif init_current and share == "none" and rank <= 12:
            vals = {q.name: numpy_helper.to_array(q) for q in source_model.graph.initializer}
            # Rotate the omitted columns across starts; adjacent omissions are
            # especially meaningful because off-diagonal rules occur in pairs.
            omit = set(explicit_omit) if explicit_omit is not None else {(seed + j) % 12 for j in range(12 - rank)}
            keep = [j for j in range(12) if j not in omit]
            initial = [vals["sx"][:, keep], vals["sy"][:, keep],
                       vals["tr"][keep], vals["tc"][keep]]
            all_factors = [torch.nn.Parameter(torch.tensor(q, dtype=torch.float32)) for q in initial]
        else:
            all_factors = [torch.nn.Parameter(torch.randn(6, rank) * scale),
                           torch.nn.Parameter(torch.randn(6, rank) * scale),
                           torch.nn.Parameter(torch.randn(rank, 6) * scale),
                           torch.nn.Parameter(torch.randn(rank, 6) * scale)]
        if share == "input": factors = [all_factors[0], all_factors[2], all_factors[3]]
        elif share == "output": factors = [all_factors[0], all_factors[1], all_factors[2]]
        elif share == "both": factors = [all_factors[0], all_factors[2]]
        else: factors = all_factors
        anchors = [q.detach().clone() for q in factors]
        lr = learning_rate if learning_rate is not None else (0.008 if (init_current or resume) else 0.035)
        opt = torch.optim.Adam(factors, lr=lr)
        for step in range(steps):
            opt.zero_grad()
            score = forward(factors)
            # run_network uses a literal >0 threshold.  Zero is therefore a
            # correct and numerically exact negative, not a failed margin.
            pos_loss = torch.relu(1.0 - score[positive]).square().mean()
            neg_loss = torch.relu(score[~positive] + neg_margin).square().mean()
            loss = pos_loss + neg_weight * neg_loss
            if anchor:
                loss = loss + anchor * sum((q - q0).square().mean() for q, q0 in zip(factors, anchors))
            loss = loss + 1e-8 * sum(q.square().mean() for q in factors)
            loss.backward()
            opt.step()
            if step % 250 == 249:
                bad = int(((score > 0) != positive).sum())
                if bad == 0:
                    break
        with torch.no_grad():
            score = forward(factors)
            bad = int(((score > 0) != positive).sum())
            min_positive = float(score[positive].min())
            max_negative = float(score[~positive].max())
            print(f"rank={rank} seed={seed} bad={bad} min_pos={min_positive:.6g} max_neg={max_negative:.6g}", flush=True)
            state = [q.detach().cpu().numpy().astype(np.float32) for q in unpack(factors)]
            key = (bad, -min_positive, max_negative)
            if best is None or key < best[0]:
                best = (key, state)
            if bad == 0 and min_positive > 1e-4 and max_negative <= 1e-5:
                break
    return best


def export(rank, factors, share):
    model, place, _, _ = arrays()
    node = model.graph.node[0]
    del model.graph.node[:]
    names = ["lr", "lc", "rr", "rc"]
    if share == "input": names[1] = names[0]
    if share == "output": names[3] = names[2]
    if share == "both": names = ["lr", "lr", "rr", "rr"]
    model.graph.node.append(helper.make_node(
        "Einsum", ["input", "place", "place", *names, "place", "place"],
        ["output"], equation="nchw,ah,dw,ab,db,be,bf,er,fs->ncrs"))
    del model.graph.initializer[:]
    tensors = [
        numpy_helper.from_array(place.astype(np.float32), "place"),
        numpy_helper.from_array(factors[0], "lr"),
    ]
    if share != "input" and share != "both": tensors.append(numpy_helper.from_array(factors[1], "lc"))
    tensors.append(numpy_helper.from_array(factors[2], "rr"))
    if share != "output" and share != "both": tensors.append(numpy_helper.from_array(factors[3], "rc"))
    model.graph.initializer.extend(tensors)
    model.graph.node[0].name = "output"
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, OUT)
    print(OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--share", choices=["none", "input", "output", "both"], default="none")
    ap.add_argument("--init-current", action="store_true")
    ap.add_argument("--neg-margin", type=float, default=0.0)
    ap.add_argument("--neg-weight", type=float, default=1.0)
    ap.add_argument("--anchor", type=float, default=0.0)
    ap.add_argument("--resume")
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--omit", help="comma-separated current-model term indices")
    args = ap.parse_args()
    explicit_omit = [int(x) for x in args.omit.split(',')] if args.omit else None
    best = train(args.rank, args.seeds, args.steps, args.share, args.init_current,
                 args.neg_margin, args.neg_weight, args.anchor, args.resume, args.learning_rate,
                 explicit_omit)
    print("best", best[0])
    export(args.rank, best[1], args.share)
