"""Build the Bolt monotonic-rule constrained GPU notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "gpu_monotonic_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "boltuzamaki/ptcg-gpu-monotonic-policy-v1",
        "title": "PTCG GPU monotonic policy V1",
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
    output = "/kaggle/working/exp-77-monotonic"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
command = [
    sys.executable,
    "automation/jobs/train_monotonic_model_gpu.py",
    "--width", "512",
    "--epochs", "160",
    "--seed", "2026071906",
    "--out", {output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-77 failed with code " + str(completed.returncode))
report = json.loads(
    Path({(output + "/monotonic_report.json")!r}).read_text(encoding="utf-8")
)
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
"""
    setup = SETUP.replace(
        'matches = glob.glob("/kaggle/input/**/tools/train_rl.py", recursive=True)',
        'matches = [\n'
        '    candidate\n'
        '    for candidate in glob.glob("/kaggle/input/**/tools/train_rl.py", recursive=True)\n'
        '    if (Path(candidate).parents[1] / "automation/jobs/train_monotonic_model_gpu.py").exists()\n'
        ']',
        1,
    )
    TARGET.joinpath("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    TARGET.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        "# EXP-77 Monotonic rule constrained model\n\n"
                        "Original training adds tactical monotonic penalties while "
                        "using only our clean-room beam labels.",
                    ),
                    ("code", setup),
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
