"""Build private clean-room GPU scale-sweep notebooks for both Kaggle profiles."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"


def build(
    folder: str,
    owner: str,
    kernel_id: str,
    title: str,
    dataset: str,
    experiment: str,
    width: int,
    seed: int,
) -> None:
    target = KERNELS / folder
    target.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": f"{owner}/{kernel_id}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["research"],
        "dataset_sources": [dataset, "kiyotah/cg-lib"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    output = f"/kaggle/working/{experiment.lower()}_policy_scale"
    run = f"""\
from pathlib import Path
import json, subprocess, sys, torch

if not torch.cuda.is_available():
    raise RuntimeError("GPU lane requested but CUDA is unavailable")
print("gpu", torch.cuda.get_device_name(0))
output = Path({output!r})
command = [
    sys.executable,
    "automation/jobs/train_beam_distillation_gpu.py",
    "--games-per-opponent", "60",
    "--width", {str(width)!r},
    "--epochs", "220",
    "--seed", {str(seed)!r},
    "--out", str(output),
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError({experiment!r} + " failed with code " + str(completed.returncode))
report = json.loads((output / "beam_distill_report.json").read_text(encoding="utf-8"))
report["experiment"] = {experiment!r}
report["public_policy_code_used"] = False
report["public_policy_actions_used_as_labels"] = False
report["submission_performed"] = False
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
"""
    target.joinpath("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    target.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        f"# {experiment} original policy scale sweep\n\n"
                        "Teacher decisions come only from our clean-room beam "
                        "policy. Frozen opponents supply states, never labels.",
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
    print(target)


def main() -> None:
    build(
        "gpu_policy_scale_17m_bolt",
        "boltuzamaki",
        "ptcg-gpu-own-policy-scale-17m",
        "ptcg gpu own policy scale 17m",
        "boltuzamaki/ptcg-private-cpu-assets-bolt-v1",
        "EXP-64A",
        1280,
        2026071901,
    )
    build(
        "gpu_policy_scale_5m_uza",
        "divyanshuboltuzamaki",
        "ptcg-gpu-own-policy-scale-5m",
        "ptcg gpu own policy scale 5m",
        "divyanshuboltuzamaki/ptcg-research-assets-uza-v1",
        "EXP-64B",
        2240,
        2026071902,
    )


if __name__ == "__main__":
    main()
