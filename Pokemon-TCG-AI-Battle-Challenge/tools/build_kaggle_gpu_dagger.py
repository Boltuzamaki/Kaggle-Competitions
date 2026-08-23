"""Build the private Bolt DAgger notebook using our own beam oracle."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_dagger_own_oracle_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "boltuzamaki/ptcg-gpu-dagger-own-oracle-v1",
        "title": "PTCG GPU DAgger own oracle V1",
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
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    output = "/kaggle/working/exp-68-dagger"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
command = [
    sys.executable,
    "automation/jobs/train_dagger_own_oracle_gpu.py",
    "--seed-games", "20",
    "--dagger-games", "20",
    "--holdout-games", "40",
    "--rounds", "2",
    "--width", "768",
    "--epochs", "140",
    "--seed", "2026071903",
    "--out", {output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-68 failed with code " + str(completed.returncode))
report = json.loads(
    Path({(output + "/dagger_report.json")!r}).read_text(encoding="utf-8")
)
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
"""
    TARGET.joinpath("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    TARGET.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        "# EXP-68 DAgger with our own oracle\n\n"
                        "The student visits states, our clean-room beam policy "
                        "supplies labels, and public opponent actions are never labels.",
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
