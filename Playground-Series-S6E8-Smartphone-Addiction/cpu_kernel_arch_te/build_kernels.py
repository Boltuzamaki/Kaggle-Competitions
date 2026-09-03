#!/usr/bin/env python3
"""Emit one Kaggle kernel directory per entry in `VARIANTS`.

Same reason as the other generators here: a Kaggle script kernel uploads a
single file and runs it with no arguments, so every variant needs a
self-contained copy, and hand-maintained copies drift.

Unlike the mlp_te generator, the GPU flag is read from the variant's own config
rather than hard-coded, because `ftt_te` is the one architecture in this family
that cannot run on four CPU cores.

    ./build_kernels.py resnet_te cnn1d_te
    ./build_kernels.py                      # all variants
"""
from pathlib import Path
import sys
import re
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = (HERE / "experiment.py").read_text()
NEEDLE = 'VARIANT = "resnet_te"  # overwritten per-kernel by build_kernels.py'

_BLOCK = SOURCE[SOURCE.index("VARIANTS = {"):]
_BLOCK = _BLOCK[:_BLOCK.index("\n}\n")]
VARIANTS = re.findall(r'^    "(\w+)":', _BLOCK, re.MULTILINE)
GPU = {v: bool(re.search(rf'"{v}":.*?"gpu": (True|False)', _BLOCK, re.S).group(1) == "True")
       for v in VARIANTS}


def main() -> None:
    wanted = sys.argv[1:] or VARIANTS
    unknown = [v for v in wanted if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; source defines {VARIANTS}")
    assert SOURCE.count(NEEDLE) == 1, "variant marker missing or ambiguous"

    for variant in wanted:
        body = SOURCE.replace(
            NEEDLE, f'VARIANT = "{variant}"  # generated; edit the source, not this copy')
        out = ROOT / f"cpu_kernel_{variant}"
        out.mkdir(exist_ok=True)
        (out / "experiment.py").write_text(body)
        (out / "kernel-metadata.json").write_text(json.dumps({
            "id": f"boltuzamaki/s6e8-cpu-{variant.replace('_', '-')}-private",
            "title": f"S6E8 CPU {variant} Private",
            "code_file": "experiment.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": GPU[variant],
            "enable_internet": False,
            "competition_sources": ["playground-series-s6e8"],
        }, indent=2) + "\n")
        print(f"wrote {out}  (gpu={GPU[variant]})")


if __name__ == "__main__":
    main()
