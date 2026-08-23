"""Build four Uza CPU notebooks for confirmations and risk-width screens."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"
DATASET = "divyanshuboltuzamaki/ptcg-research-assets-uza-v1"
EXPERIMENTS = (
    (
        "cpu_chance_coupled_confirmation",
        "divyanshuboltuzamaki/ptcg-cpu-chance-coupled-confirmation",
        "PTCG CPU chance coupled confirmation",
        "EXP-69C",
        "chancecoupled",
        24,
        None,
        None,
    ),
    (
        "cpu_progressive_confirmation",
        "divyanshuboltuzamaki/ptcg-cpu-progressive-confirmation",
        "PTCG CPU progressive confirmation",
        "EXP-65C",
        "progressive",
        24,
        None,
        None,
    ),
    (
        "cpu_risk_width4",
        "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-width4",
        "PTCG CPU risk sensitive width 4",
        "EXP-74",
        "risksensitive",
        12,
        0.45,
        4,
    ),
    (
        "cpu_risk_width8",
        "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-width8",
        "PTCG CPU risk sensitive width 8",
        "EXP-75",
        "risksensitive",
        12,
        0.45,
        8,
    ),
)


def main() -> None:
    for (
        folder,
        kernel_id,
        title,
        experiment,
        candidate,
        games,
        weight,
        width,
    ) in EXPERIMENTS:
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
            str(games),
            "--max-decks",
            "20",
            "--deck-catalog",
            "data/live_decks.json",
            "--out",
            output,
        ]
        if weight is not None:
            command.extend(
                [
                    "--risk-weight",
                    str(weight),
                    "--risk-iterations",
                    "36",
                    "--flat-root-width",
                    str(width),
                ]
            )
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
summary["fresh_confirmation"] = {str(experiment in {"EXP-69C", "EXP-65C"})}
summary["exploratory_width_screen"] = {str(experiment in {"EXP-74", "EXP-75"})}
summary["public_policy_code_used"] = False
summary["public_policy_actions_used_as_labels"] = False
summary["submission_performed"] = False
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
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
                            f"# {experiment} {title}\n\n"
                            "Fresh original-policy evidence. Exploratory screens "
                            "cannot promote without independent confirmation.",
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
