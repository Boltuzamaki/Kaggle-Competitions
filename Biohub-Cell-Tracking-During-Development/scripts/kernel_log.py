#!/usr/bin/env python
"""Download and pretty-print a Kaggle kernel's log.

The CLI stores logs as a JSON array of ``{"stream_name", "time", "data"}``
records, which is unreadable raw. This reassembles the stream.

    python scripts/kernel_log.py 03_train_unet --tail 4000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kaggle_ops import kaggle_username, resolve_notebook, slug_for  # noqa: E402

def fetch_log(kernel_id: str, dest: Path) -> Path | None:
    subprocess.run(["kaggle", "kernels", "output", kernel_id, "-p", str(dest)],
                   capture_output=True, text=True)
    logs = sorted(dest.glob("*.log"))
    return logs[0] if logs else None

def render(log_path: Path) -> str:
    raw = log_path.read_text(errors="replace")
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        # Kaggle sometimes emits a leading fragment before the array.
        start = raw.find("[")
        entries = json.loads(raw[start:]) if start >= 0 else []
    return "".join(e.get("data", "") for e in entries)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--slug")
    ap.add_argument("--tail", type=int, default=6000, help="characters from the end")
    ap.add_argument("--grep", help="only lines containing this substring")
    args = ap.parse_args()

    kernel_id = f"{kaggle_username()}/{args.slug or slug_for(resolve_notebook(args.notebook))}"
    with tempfile.TemporaryDirectory() as tmp:
        log = fetch_log(kernel_id, Path(tmp))
        if log is None:
            raise SystemExit(f"no log found for {kernel_id}")
        text = render(log)

    if args.grep:
        text = "\n".join(l for l in text.splitlines() if args.grep in l)
    print(text[-args.tail:] if args.tail else text)

if __name__ == "__main__":
    main()
