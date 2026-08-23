"""Build the Bolt decision-complexity curriculum GPU notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_complexity_curriculum_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    trainer_source = (
        ROOT / "automation" / "jobs" / "train_complexity_curriculum_gpu.py"
    ).read_text(encoding="utf-8")
    metadata = {
        "id": "boltuzamaki/ptcg-gpu-complexity-curriculum-v1",
        "title": "PTCG GPU complexity curriculum V1",
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
    output = "/kaggle/working/exp-83-complexity-curriculum"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
trainer = Path(
    "/kaggle/working/ptcg/automation/jobs/train_complexity_curriculum_gpu.py"
)
trainer.write_text({trainer_source!r}, encoding="utf-8")
command = [
    sys.executable,
    str(trainer),
    "--width", "256",
    "--epochs", "120",
    "--seed", "2026071910",
    "--out", {output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-83 failed with code " + str(completed.returncode))
report = json.loads(
    Path({(output + "/complexity_curriculum_report.json")!r}).read_text(
        encoding="utf-8"
    )
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
                        "# EXP-83 Decision complexity curriculum\n\n"
                        "An easy-to-hard schedule is compared with a shuffled "
                        "control using the same clean-room decision labels.",
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
