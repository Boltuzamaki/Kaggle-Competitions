"""Build the Bolt novelty-policy-population GPU notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_novelty_population_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    trainer_source = (
        ROOT / "automation" / "jobs" / "train_novelty_population_gpu.py"
    ).read_text(encoding="utf-8")
    metadata = {
        "id": "boltuzamaki/ptcg-gpu-novelty-population-v1",
        "title": "PTCG GPU novelty population V1",
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
        "kernel_sources": ["boltuzamaki/ptcg-gpu-pairwise-preference-v1"],
        "competition_sources": [],
        "model_sources": [],
    }
    output = "/kaggle/working/exp-89-novelty-population"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
trainer = Path(
    "/kaggle/working/ptcg/automation/jobs/train_novelty_population_gpu.py"
)
trainer.write_text({trainer_source!r}, encoding="utf-8")
command = [
    sys.executable,
    str(trainer),
    "--population", "384",
    "--archive", "12",
    "--generations", "100",
    "--seed", "2026071912",
    "--out", {output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-89 failed with code " + str(completed.returncode))
report = json.loads(
    Path({(output + "/novelty_population_report.json")!r}).read_text(
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
                        "# EXP-89 Novelty based policy population\n\n"
                        "A diverse compact policy population is evolved from "
                        "our clean-room beam decisions. Oracle agreement is "
                        "reported only as selector headroom.",
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
