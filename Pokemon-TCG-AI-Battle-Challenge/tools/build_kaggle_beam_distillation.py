"""Build the Kaggle GPU notebook for distilling our own beam policy."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, metadata, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_beam_distillation"

RUN = """\
from pathlib import Path
import json
import subprocess
import sys
import torch

print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
output = Path("/kaggle/working/exp51_beam_distillation")
command = [
    sys.executable,
    "automation/jobs/train_beam_distillation_gpu.py",
    "--games-per-opponent", "20",
    "--out", str(output),
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError(f"EXP-51 failed with code {completed.returncode}")
report = json.loads((output / "beam_distill_report.json").read_text())
print(json.dumps(report, indent=2))
"""


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    TARGET.joinpath("kernel-metadata.json").write_text(
        json.dumps(
            metadata(
                "ptcg-gpu-own-beam-distillation",
                "PTCG GPU own beam distillation",
                True,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    TARGET.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        "# EXP-51 distill our own beam policy\n\n"
                        "Teacher labels come only from our original beam agent. "
                        "Frozen opponents provide states, not action labels.",
                    ),
                    ("code", SETUP),
                    ("code", RUN),
                ],
                "GPU",
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
