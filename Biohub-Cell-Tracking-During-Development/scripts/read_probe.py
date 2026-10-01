#!/usr/bin/env python
"""Pull a Kaggle kernel's output and print only the decision-relevant lines.

The forked public stack prints several thousand lines per run. Reading them by
hand is slow and, worse, it is easy to skim past the one line that matters. This
extracts the handful that actually decide whether to spend a submission:

* the held-out validator's proxy score and its division TP/FP/FN
* what the post-process sweep selected, and against which baseline
* the safe-division counters (how many candidates, how many survived each gate,
  how many the cap threw away)
* the submission's own shape (rows, datasets, node counts)
* any traceback

Usage
-----
    python scripts/read_probe.py 12_div_probe [--all] [--grep PATTERN]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from kaggle_ops import kaggle_username, resolve_notebook, slug_for  # noqa: E402

# lines worth surfacing, in the order they matter
PATTERNS = [
    r"Traceback|Error|error:|raise |FAILED|MISMATCH",
    r"VALIDATOR",
    r"proxy_score|adjusted_edge_jaccard|division_jaccard|div_tp|div_fp|div_fn",
    r"PPSWEEP|pp_sweep|selected|candidate .*score|baseline",
    r"safe_division|safe_divisions_added|cap_skipped|geometric_candidates",
    r"deepcenter_safe_div|divergence_rejected|symmetry_rejected|mutual_nn_rejected",
    r"submission|rows|datasets",
    r"Configuration guard|BIOHUB_PRESET|SCORE_AXIS",
]

def fetch(kernel_id: str, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", str(dest)],
                   capture_output=True, text=True)
    logs = sorted(dest.glob("*.log"))
    if not logs:
        raise SystemExit(f"no log downloaded for {kernel_id} - has it finished?")
    return logs[0]

def stdout_of(log: Path) -> str:
    """Kaggle logs are JSON records, not plain text."""
    try:
        rows = json.loads(log.read_text())
    except Exception:
        return log.read_text()
    return "".join(r.get("data", "") for r in rows if isinstance(r, dict))

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--all", action="store_true", help="print the whole log")
    ap.add_argument("--grep", help="extra regex to include")
    ap.add_argument("--tail", type=int, default=0, help="also print the last N lines")
    args = ap.parse_args()

    slug = slug_for(resolve_notebook(args.notebook))
    kernel_id = f"{kaggle_username()}/{slug}"

    with tempfile.TemporaryDirectory() as tmp:
        log = fetch(kernel_id, Path(tmp))
        text = stdout_of(log)
        lines = text.splitlines()

        if args.all:
            print(text)
            return

        pats = list(PATTERNS) + ([args.grep] if args.grep else [])
        rx = re.compile("|".join(f"(?:{p})" for p in pats), re.I)
        hits = [ln for ln in lines if rx.search(ln)]

        print(f"=== {kernel_id} :: {len(lines):,} log lines, {len(hits)} relevant ===\n")
        seen = set()
        for ln in hits:
            key = ln.strip()[:160]
            if key in seen:      # the stack repeats identical counters per film
                continue
            seen.add(key)
            print(ln.rstrip()[:200])

        if args.tail:
            print(f"\n=== last {args.tail} lines ===")
            for ln in lines[-args.tail:]:
                print(ln.rstrip()[:200])

if __name__ == "__main__":
    main()
