#!/usr/bin/env python3
"""Append a row to the experiment ledger in EXPERIMENT_LOG.md.

Keeps the log consistent so it stays readable as history rather than drifting
into free text. Verdicts are constrained on purpose: the point of the ledger is
that a later reader can tell at a glance whether an idea is already settled.

    ./log_experiment.py --id E050 --experiment "10-fold CatBoost" \
        --metric "OOF 0.96801" --verdict REJECTED --note "below 5-fold"

With --auto-metrics it reads a metrics JSON written by an experiment script and
pulls oof_auc out of it, so the number in the log is the number the run produced
rather than one retyped by hand.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

LOG = Path(__file__).parent / "EXPERIMENT_LOG.md"
VERDICTS = ("ADOPTED", "REJECTED", "MARGINAL", "KEPT FOR BLEND", "PENDING")
LEDGER_END = "\n**Pattern across"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--metric", default="")
    ap.add_argument("--verdict", required=True, choices=VERDICTS)
    ap.add_argument("--note", default="")
    ap.add_argument("--date", default=str(date.today()))
    ap.add_argument("--auto-metrics", type=Path,
                    help="metrics JSON to read oof_auc from")
    args = ap.parse_args()

    metric = args.metric
    if args.auto_metrics:
        data = json.loads(args.auto_metrics.read_text())
        auc = data.get("oof_auc")
        if auc is None:
            raise SystemExit(f"no oof_auc in {args.auto_metrics}")
        metric = (f"{metric}, " if metric else "") + f"OOF {auc:.7f}"

    text = LOG.read_text()
    if re.search(rf"^\| {re.escape(args.id)} \|", text, flags=re.M):
        raise SystemExit(f"id {args.id} already in the ledger; pick another or edit by hand")

    for field in (args.experiment, metric, args.note):
        if "|" in field:
            raise SystemExit("fields must not contain '|', it breaks the table")

    row = f"| {args.id} | {args.date} | {args.experiment} | {metric} | {args.verdict} | {args.note} |\n"
    if LEDGER_END not in text:
        raise SystemExit("could not find the end of the ledger table in EXPERIMENT_LOG.md")
    text = text.replace(LEDGER_END, row + LEDGER_END, 1)
    LOG.write_text(text)
    print(row.strip())


if __name__ == "__main__":
    main()
