"""Build five independent private Uza CPU notebooks for the frontier wave."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"
DATASET = "divyanshuboltuzamaki/ptcg-research-assets-uza-v1"

EXPERIMENTS = (
    (
        "cpu_belief_6x8_live_field",
        "divyanshuboltuzamaki/ptcg-cpu-belief-6x8-live-field",
        "PTCG CPU belief 6x8 live field",
        "EXP-63D",
        "belief6x8",
    ),
    (
        "cpu_belief_12x4_live_field",
        "divyanshuboltuzamaki/ptcg-cpu-belief-12x4-live-field",
        "PTCG CPU belief 12x4 live field",
        "EXP-63E",
        "belief12x4",
    ),
    (
        "cpu_progressive_widening_live_field",
        "divyanshuboltuzamaki/ptcg-cpu-progressive-widening-live-field",
        "PTCG CPU progressive widening live field",
        "EXP-65",
        "progressive",
    ),
    (
        "cpu_risk_sensitive_live_field",
        "divyanshuboltuzamaki/ptcg-cpu-risk-sensitive-live-field",
        "PTCG CPU risk sensitive live field",
        "EXP-66",
        "risksensitive",
    ),
    (
        "cpu_disagreement_fallback_live_field",
        "divyanshuboltuzamaki/ptcg-cpu-disagreement-fallback-live-field",
        "PTCG CPU disagreement fallback live field",
        "EXP-67",
        "disagreement",
    ),
)


def metadata(kernel_id: str, title: str) -> dict:
    return {
        "id": kernel_id,
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["pokemon-tcg", "research", "clean-room"],
        "dataset_sources": [DATASET, "kiyotah/cg-lib"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "None",
    }


def run_cell(experiment: str, candidate: str) -> str:
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
    return f"""\
from pathlib import Path
import json, subprocess, sys

command = {command!r}
command[0] = sys.executable
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("experiment failed with code " + str(completed.returncode))
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


def main() -> None:
    for folder, kernel_id, title, experiment, candidate in EXPERIMENTS:
        target = KERNELS / folder
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath("kernel-metadata.json").write_text(
            json.dumps(metadata(kernel_id, title), indent=2),
            encoding="utf-8",
        )
        target.joinpath("notebook.ipynb").write_text(
            json.dumps(
                notebook(
                    [
                        (
                            "markdown",
                            f"# {experiment} {title}\n\n"
                            "Original-policy evaluation against frozen local v3. "
                            "No public policy code or public action labels are used.",
                        ),
                        ("code", SETUP),
                        ("code", run_cell(experiment, candidate)),
                    ]
                ),
                indent=1,
            ),
            encoding="utf-8",
        )
        print(target)


if __name__ == "__main__":
    main()
