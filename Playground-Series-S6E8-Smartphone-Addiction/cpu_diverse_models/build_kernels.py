#!/usr/bin/env python3
"""Emit one Kaggle kernel directory per model in `experiment.py`.

A Kaggle script kernel uploads a single code file and runs it with no arguments,
so each model needs its own self-contained copy with the configuration baked in.
Those copies already existed here and had been maintained by hand; generating
them means a fix to the encoder reaches every kernel instead of whichever ones
somebody remembered to update.

    ./build_kernels.py et_deep          # one model
    ./build_kernels.py                  # all of them
"""
from pathlib import Path
import sys
import re
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = (HERE / "experiment.py").read_text()

# Read the model names out of the source rather than importing it, which would
# pull in pandas and sklearn just to list five strings. Keys are matched at a
# fixed indent inside the MODELS block, so an entry whose value wraps onto a
# second line is still found.
_BLOCK = SOURCE[SOURCE.index("MODELS = {"):]
_BLOCK = _BLOCK[:_BLOCK.index("\n}\n")]
MODELS = re.findall(r'^    "(\w+)":', _BLOCK, re.MULTILINE)

TAIL = '''

if __name__ == "__main__":
    # Kaggle runs this script with no arguments, so the configuration is fixed
    # here rather than passed on the command line.
    sys.argv = ["experiment", "--model", "{model}", "--threads", "4"]
    main()
'''


def main() -> None:
    wanted = sys.argv[1:] or MODELS
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; source defines {MODELS}")

    body = SOURCE
    if "\nimport sys\n" not in body:
        body = body.replace("\nimport argparse\n", "\nimport argparse\nimport sys\n", 1)
    # Drop the source's own entry point; each kernel gets its own.
    body = body[:body.index('\nif __name__ == "__main__":')]

    for model in wanted:
        out = ROOT / f"cpu_kernel_{model}"
        out.mkdir(exist_ok=True)
        (out / "experiment.py").write_text(body + TAIL.format(model=model))
        (out / "kernel-metadata.json").write_text(json.dumps({
            "id": f"boltuzamaki/s6e8-cpu-{model.replace('_', '-')}-private",
            "title": f"S6E8 CPU {model} Private",
            "code_file": "experiment.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "competition_sources": ["playground-series-s6e8"],
        }, indent=2) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
