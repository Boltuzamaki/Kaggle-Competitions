"""Build Uza CPU notebooks for coupled chance sampling and adaptive budgets."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"
DATASET = "divyanshuboltuzamaki/ptcg-research-assets-uza-v1"
EXPERIMENTS = (
    (
        "cpu_chance_coupled_live_field",
        "divyanshuboltuzamaki/ptcg-cpu-chance-coupled-live-field",
        "PTCG CPU chance coupled live field",
        "EXP-69",
        "chancecoupled",
    ),
    (
        "cpu_adaptive_budget_live_field",
        "divyanshuboltuzamaki/ptcg-cpu-adaptive-budget-live-field",
        "PTCG CPU adaptive budget live field",
        "EXP-70",
        "adaptivebudget",
    ),
)


def main() -> None:
    for folder, kernel_id, title, experiment, candidate in EXPERIMENTS:
        target = KERNELS / folder
        target.mkdir(parents=True, exist_ok=True)
        metadata = {
            "id": kernel_id,
            "title": title,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["research"],
            "dataset_sources": [DATASET, "kiyotah/cg-lib"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
        }
        output = f"/kaggle/working/{experiment.lower()}-live-field"
        command = [
            "python",
            "automation/jobs/live_deck_gauntlet.py",
            "--candidate",
            candidate,
            "--games-per-deck",
            "12",
            "--max-decks",
            "20",
            "--deck-catalog",
            "data/live_decks.json",
            "--out",
            output,
        ]
        run = f"""\
from pathlib import Path
import json, subprocess, sys

command = {command!r}
command[0] = sys.executable
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError({experiment!r} + " failed with code " + str(completed.returncode))
summary = json.loads(Path({(output + ".json")!r}).read_text(encoding="utf-8"))
summary["experiment"] = {experiment!r}
summary["public_policy_code_used"] = False
summary["public_policy_actions_used_as_labels"] = False
summary["submission_performed"] = False
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
"""
        target.joinpath("kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        target.joinpath("notebook.ipynb").write_text(
            json.dumps(
                notebook(
                    [
                        (
                            "markdown",
                            f"# {experiment} {title}\n\n"
                            "Original-policy evaluation with frozen opponents. "
                            "No public policy code or public action labels are used.",
                        ),
                        ("code", SETUP),
                        ("code", run),
                    ]
                ),
                indent=1,
            ),
            encoding="utf-8",
        )
        print(target)


if __name__ == "__main__":
    main()
