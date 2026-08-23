"""Train a binary ranker on outcome-discordant router action deviations."""
from __future__ import annotations

import argparse
import pickle
import random

import torch
import torch.nn.functional as F

from train_entity_advantage import EntityAdvantage
from train_state_ranker import collate, norms


def gate_metrics(prob: torch.Tensor, truth: torch.Tensor) -> dict:
    result = {"threshold": 1.0, "precision": 0.0, "coverage": 0.0, "positives": 0}
    for threshold in torch.linspace(0.5, 0.99, 50):
        use = prob >= threshold
        count = int(use.sum())
        if count < 25:
            continue
        precision = float(truth[use].float().mean())
        if precision >= 0.8 and count > result["positives"]:
            result = {
                "threshold": float(threshold),
                "precision": precision,
                "coverage": count / len(truth),
                "positives": count,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch", type=int, default=192)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()

    rows = []
    for path in args.data:
        rows.extend(pickle.load(open(path, "rb")))
    rows = [row for row in rows if int(row["advantage"]) != 0]
    stats = norms(rows)
    rng = random.Random(args.seed)
    groups = sorted({row["episode"] for row in rows})
    rng.shuffle(groups)
    hold = set(groups[: max(1, len(groups) // 5)])
    train = [row for row in rows if row["episode"] not in hold]
    valid = [row for row in rows if row["episode"] in hold]
    buckets = {
        value: [row for row in train if int(row["advantage"]) == value]
        for value in (-1, 1)
    }

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EntityAdvantage(layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    best_score = -1.0
    stale = 0
    for epoch in range(args.epochs):
        sample_count = max(map(len, buckets.values()))
        epoch_rows = []
        for bucket in buckets.values():
            epoch_rows.extend(rng.choices(bucket, k=sample_count))
        rng.shuffle(epoch_rows)
        model.train()
        for start in range(0, len(epoch_rows), args.batch):
            part = epoch_rows[start : start + args.batch]
            batch = [tensor.to(device) for tensor in collate(part, stats, maxs=80)]
            target = torch.tensor(
                [int(row["advantage"]) > 0 for row in part],
                dtype=torch.long,
                device=device,
            )
            logits3 = model(batch)
            logits = logits3[:, (0, 2)]
            loss = F.cross_entropy(logits, target, label_smoothing=0.02)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        probabilities = []
        truth = []
        with torch.no_grad():
            for start in range(0, len(valid), args.batch):
                part = valid[start : start + args.batch]
                batch = [tensor.to(device) for tensor in collate(part, stats, maxs=80)]
                logits = model(batch)[:, (0, 2)]
                probabilities.append(logits.softmax(-1)[:, 1].cpu())
                truth.extend(int(row["advantage"]) > 0 for row in part)
        probability = torch.cat(probabilities)
        target = torch.tensor(truth)
        accuracy = float(((probability >= 0.5) == target).float().mean())
        gate = gate_metrics(probability, target)
        score = gate["precision"] * gate["coverage"]
        receipt = {"accuracy": accuracy, "gate": gate, "rows": len(rows)}
        if score > best_score:
            best_score = score
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "stats": stats,
                    "gate": gate,
                    "accuracy": accuracy,
                    "layers": args.layers,
                    "binary_pairwise": True,
                    "rows": len(rows),
                },
                args.out,
            )
        else:
            stale += 1
        print(epoch + 1, receipt, flush=True)
        if stale >= 6:
            break


if __name__ == "__main__":
    main()
