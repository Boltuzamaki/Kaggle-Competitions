"""Build the Bolt calibrated-uncertainty GPU notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_uncertainty_gate_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "boltuzamaki/ptcg-gpu-calibrated-uncertainty-v1",
        "title": "PTCG GPU calibrated uncertainty V1",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["research"],
        "dataset_sources": [
            "boltuzamaki/ptcg-private-cpu-assets-bolt-v1",
            "kiyotah/cg-lib",
        ],
        "kernel_sources": [
            "boltuzamaki/ptcg-gpu-pairwise-preference-v1"
        ],
        "competition_sources": [],
        "model_sources": [],
    }
    output = "/kaggle/working/exp-76-uncertainty"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
command = [
    sys.executable,
    "automation/jobs/train_uncertainty_gate_gpu.py",
    "--models", "5",
    "--width", "256",
    "--epochs", "120",
    "--seed", "2026071905",
    "--out", {output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-76 failed with code " + str(completed.returncode))
report = json.loads(
    Path({(output + "/uncertainty_report.json")!r}).read_text(encoding="utf-8")
)
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
"""
    TARGET.joinpath("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    TARGET.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        "# EXP-76 Calibrated uncertainty gate\n\n"
                        "An original ensemble defers to the sequenced policy when "
                        "confidence or ensemble agreement is low.",
                    ),
                    ("code", SETUP),
                    ("code", run),
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
