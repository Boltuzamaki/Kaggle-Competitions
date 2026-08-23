"""Stage the iteration-70 model and build its private Kaggle CPU evaluation lane."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from build_kaggle_lanes import SETUP, metadata, notebook


ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "kaggle_remote"
DATASET = REMOTE / "ptcg_gpu_iter70_eval_v1"
KERNEL = REMOTE / "kernels" / "cpu_gpu_iter70_eval"
RESULTS = ROOT / "scratchpad" / "kaggle_results" / "gpu_iter70"


def main() -> None:
    DATASET.mkdir(parents=True, exist_ok=True)
    KERNEL.mkdir(parents=True, exist_ok=True)
    files = {
        RESULTS / "ptcg_gpu_v3only_iter70.pth": DATASET / "ptcg_gpu_v3only_iter70.pth",
        RESULTS / "ptcg_gpu_v3only_iter70.pth.arch.json": DATASET / "ptcg_gpu_v3only_iter70.pth.arch.json",
        ROOT / "automation" / "jobs" / "evaluate_rl_checkpoint.py": DATASET / "evaluate_rl_checkpoint.py",
    }
    for source, target in files.items():
        if not source.is_file():
            raise SystemExit(f"missing required artifact: {source}")
        shutil.copy2(source, target)

    (DATASET / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "PTCG GPU Iter70 Evaluation V1",
                "id": "boltuzamaki/ptcg-gpu-iter70-evaluation-v1",
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    evaluation = """\
from pathlib import Path
import glob, shutil, subprocess, sys

model_matches = glob.glob("/kaggle/input/**/ptcg_gpu_v3only_iter70.pth", recursive=True)
arch_matches = glob.glob("/kaggle/input/**/ptcg_gpu_v3only_iter70.pth.arch.json", recursive=True)
evaluator_matches = glob.glob("/kaggle/input/**/evaluate_rl_checkpoint.py", recursive=True)
if not model_matches or not arch_matches or not evaluator_matches:
    raise RuntimeError("iteration-70 evaluation assets are not attached")
model_dir = Path("/kaggle/working/ptcg/models/remote")
model_dir.mkdir(parents=True, exist_ok=True)
model = model_dir / "ptcg_gpu_v3only_iter70.pth"
shutil.copy2(model_matches[0], model)
shutil.copy2(arch_matches[0], Path(str(model) + ".arch.json"))
evaluator = Path("/kaggle/working/ptcg/automation/jobs/evaluate_rl_checkpoint.py")
shutil.copy2(evaluator_matches[0], evaluator)
subprocess.run([
    sys.executable, str(evaluator),
    "--model", str(model),
    "--games-per-opponent", "30",
    "--search", "8",
    "--time-budget", "0.75",
    "--out", "/kaggle/working/kaggle_gpu_iter70_holdout.json",
], check=True)
"""
    kernel_metadata = metadata(
        "ptcg-cpu-gpu-iter70-holdout-v1",
        "PTCG CPU GPU Iter70 Holdout V1",
        False,
    )
    kernel_metadata["dataset_sources"].append(
        "boltuzamaki/ptcg-gpu-iter70-evaluation-v1"
    )
    (KERNEL / "kernel-metadata.json").write_text(
        json.dumps(kernel_metadata, indent=2),
        encoding="utf-8",
    )
    (KERNEL / "notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        "# Iteration-70 held-out evaluation\n\n"
                        "Original model versus frozen local baselines. No submission.",
                    ),
                    ("code", SETUP),
                    ("code", evaluation),
                ]
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": "boltuzamaki/ptcg-gpu-iter70-evaluation-v1",
                "kernel": "boltuzamaki/ptcg-cpu-gpu-iter70-holdout-v1",
                "public_policy_code_included": False,
                "submission_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
