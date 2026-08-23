"""Build fresh confirmation and exploratory risk-weight CPU notebooks."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"
DATASET = "divyanshuboltuzamaki/ptcg-research-assets-uza-v1"
EXPERIMENTS = (
    (
        "cpu_risk_sensitive_confirmation",
        "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-confirmation",
        "PTCG CPU risk sensitive confirmation",
        "EXP-66C",
        0.45,
        24,
    ),
    (
        "cpu_risk_sensitive_w020",
        "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-w020",
        "PTCG CPU risk sensitive weight 020",
        "EXP-71",
        0.20,
        12,
    ),
    (
        "cpu_risk_sensitive_w070",
        "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-w070",
        "PTCG CPU risk sensitive weight 070",
        "EXP-72",
        0.70,
        12,
    ),
)


def main() -> None:
    for folder, kernel_id, title, experiment, weight, games in EXPERIMENTS:
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
            "risksensitive",
            "--risk-weight",
            str(weight),
            "--risk-iterations",
            "36",
            "--flat-root-width",
            "6",
            "--games-per-deck",
            str(games),
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
summary["confirmation"] = {str(experiment == "EXP-66C")}
summary["exploratory_weight_screen"] = {str(experiment != "EXP-66C")}
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
                            "Fresh original-policy evaluation. Weight screens "
                            "are exploratory and require independent confirmation.",
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
