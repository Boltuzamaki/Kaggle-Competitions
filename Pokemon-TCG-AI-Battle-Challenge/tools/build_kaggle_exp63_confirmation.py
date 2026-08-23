"""Build the private Uza EXP-63C particle-belief confirmation notebook."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_particle_belief_3x16_confirm"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "divyanshuboltuzamaki/ptcg-cpu-particle-belief-3x16-confirmation",
        "title": "PTCG CPU particle belief 3x16 confirmation",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["pokemon-tcg", "research", "clean-room"],
        "dataset_sources": [
            "divyanshuboltuzamaki/ptcg-research-assets-uza-v1",
            "kiyotah/cg-lib",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "None",
    }
    command = [
        "python",
        "automation/jobs/live_deck_gauntlet.py",
        "--candidate",
        "belief3x16",
        "--games-per-deck",
        "12",
        "--max-decks",
        "20",
        "--deck-catalog",
        "data/live_decks.json",
        "--out",
        "/kaggle/working/particle-belief-3x16-confirm",
    ]
    run = f"""\
from pathlib import Path
import json, subprocess, sys

command = {command!r}
command[0] = sys.executable
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("experiment failed with code " + str(completed.returncode))
summary = json.loads(
    Path("/kaggle/working/particle-belief-3x16-confirm.json").read_text(
        encoding="utf-8"
    )
)
summary["experiment"] = "EXP-63C"
summary["public_policy_code_used"] = False
summary["public_policy_actions_used_as_labels"] = False
summary["submission_performed"] = False
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
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
                        "# EXP-63C Particle belief search confirmation\n\n"
                        "Fresh clean-room confirmation. Public policy code and "
                        "public-policy action labels are excluded.",
                    ),
                    ("code", SETUP),
                    ("code", run),
                ]
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
