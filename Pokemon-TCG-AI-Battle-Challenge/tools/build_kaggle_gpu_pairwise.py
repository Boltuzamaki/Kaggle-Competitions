"""Build the private Bolt pairwise action-preference GPU notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_pairwise_preference_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "boltuzamaki/ptcg-gpu-pairwise-preference-v1",
        "title": "PTCG GPU pairwise preference V1",
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
    output = "/kaggle/working/exp-73-pairwise"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
command = [
    sys.executable,
    "automation/jobs/train_beam_distillation_gpu.py",
    "--games-per-opponent", "60",
    "--width", "768",
    "--epochs", "180",
    "--loss", "pairwise",
    "--seed", "2026071904",
    "--out", {output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-73 failed with code " + str(completed.returncode))
report = json.loads(
    Path({(output + "/beam_distill_report.json")!r}).read_text(encoding="utf-8")
)
report["experiment"] = "EXP-73"
report["strategy_rank"] = 42
report["strategy"] = "Pairwise action preference model"
report["public_policy_code_used"] = False
report["public_policy_actions_used_as_labels"] = False
report["submission_performed"] = False
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
                        "# EXP-73 Pairwise action preference model\n\n"
                        "The scalar utility learns teacher-versus-hardest-alternative "
                        "preferences using only our clean-room beam labels.",
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
