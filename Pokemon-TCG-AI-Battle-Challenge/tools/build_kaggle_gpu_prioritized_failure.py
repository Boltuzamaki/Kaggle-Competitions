"""Build the Bolt prioritized rare-failure replay GPU notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_prioritized_failure_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    trainer_source = (
        ROOT / "automation" / "jobs" / "train_prioritized_failure_gpu.py"
    ).read_text(encoding="utf-8")
    metadata = {
        "id": "boltuzamaki/ptcg-gpu-prioritized-failure-v1",
        "title": "PTCG GPU prioritized failure V1",
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
    output = "/kaggle/working/exp-82-prioritized-failure"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
trainer = Path(
    "/kaggle/working/ptcg/automation/jobs/train_prioritized_failure_gpu.py"
)
trainer.write_text({trainer_source!r}, encoding="utf-8")
command = [
    sys.executable,
    str(trainer),
    "--width", "256",
    "--epochs", "140",
    "--rare-repeat", "8",
    "--seed", "2026071909",
    "--out", {output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-82 failed with code " + str(completed.returncode))
report = json.loads(
    Path({(output + "/prioritized_failure_report.json")!r}).read_text(
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
                        "# EXP-82 Prioritized rare failure replay\n\n"
                        "Only our own sequenced-versus-beam disagreements are "
                        "oversampled. An unweighted control uses the same data.",
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
