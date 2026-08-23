"""Build the Kaggle CPU screen for the recovery inventory policy."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, metadata, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_recovery_inventory"

RUN = """\
from pathlib import Path
import json
import subprocess
import sys

output = Path("/kaggle/working/exp48_recovery_inventory_live")
command = [
    sys.executable,
    "automation/jobs/live_deck_gauntlet.py",
    "--candidate", "recovery",
    "--games-per-deck", "12",
    "--max-decks", "20",
    "--deck-catalog", "data/live_decks.json",
    "--out", str(output),
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError(f"recovery inventory screen failed with code {completed.returncode}")

report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
summary = {
    "experiment": "EXP-48",
    "strategy_rank": 15,
    "strategy": "Recovery inventory planner",
    "candidate": report["candidate"],
    "opponent_policy": report["opponent_policy"],
    "games": report["total_games"],
    "wins": report["wins"],
    "losses": report["losses"],
    "draws": report["draws"],
    "winrate": report["winrate"],
    "failures": report["candidate_failures"] + report["game_errors"],
    "json_artifact": str(output.with_suffix(".json")),
    "csv_artifact": str(output.with_suffix(".csv")),
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}
Path("/kaggle/working/exp48_recovery_inventory_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
"""


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    TARGET.joinpath("kernel-metadata.json").write_text(
        json.dumps(
            metadata(
                "ptcg-cpu-recovery-inventory-live-field",
                "PTCG CPU recovery inventory live field",
                False,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    TARGET.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        "# EXP-48 recovery inventory planner\n\n"
                        "Original policy logic against frozen local v3. "
                        "No public policy code or action labels are used.",
                    ),
                    ("code", SETUP),
                    ("code", RUN),
                ]
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
