"""Build independent Kaggle CPU screens for small original-policy changes."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, metadata, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"

SCREENS = (
    {
        "folder": "cpu_end_turn_regret",
        "kernel": "ptcg-cpu-end-turn-regret-live-field",
        "title": "PTCG CPU end turn regret live field",
        "experiment": "EXP-49",
        "rank": 16,
        "strategy": "End turn regret check",
        "candidate": "endregret",
    },
    {
        "folder": "cpu_bench_capacity",
        "kernel": "ptcg-cpu-bench-capacity-live-field",
        "title": "PTCG CPU bench capacity live field",
        "experiment": "EXP-50",
        "rank": 17,
        "strategy": "Bench capacity planner",
        "candidate": "bench",
    },
    {
        "folder": "cpu_evolution_stack",
        "kernel": "ptcg-cpu-evolution-stack-live-field",
        "title": "PTCG CPU evolution stack live field",
        "experiment": "EXP-52",
        "rank": 18,
        "strategy": "Evolution stack preservation",
        "candidate": "stack",
    },
    {
        "folder": "cpu_damage_breakpoint",
        "kernel": "ptcg-cpu-damage-breakpoint-live-field",
        "title": "PTCG CPU damage breakpoint live field",
        "experiment": "EXP-53",
        "rank": 20,
        "strategy": "Damage breakpoint planner",
        "candidate": "damage",
    },
    {
        "folder": "cpu_stadium_timing",
        "kernel": "ptcg-cpu-stadium-timing-live-field",
        "title": "PTCG CPU stadium timing live field",
        "experiment": "EXP-54",
        "rank": 21,
        "strategy": "Full Metal Lab timing model",
        "candidate": "stadium",
    },
    {
        "folder": "cpu_healing_breakpoint",
        "kernel": "ptcg-cpu-healing-breakpoint-live-field",
        "title": "PTCG CPU healing breakpoint live field",
        "experiment": "EXP-55",
        "rank": 22,
        "strategy": "Healing value forecast",
        "candidate": "healing",
    },
)

RUN = """\
from pathlib import Path
import json
import subprocess
import sys

output = Path("/kaggle/working/{slug}")
command = [
    sys.executable,
    "automation/jobs/live_deck_gauntlet.py",
    "--candidate", "{candidate}",
    "--games-per-deck", "12",
    "--max-decks", "20",
    "--deck-catalog", "data/live_decks.json",
    "--out", str(output),
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("{experiment} failed with code " + str(completed.returncode))

report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
summary = {{
    "experiment": "{experiment}",
    "strategy_rank": {rank},
    "strategy": "{strategy}",
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
}}
Path("/kaggle/working/{slug}_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)
print(json.dumps(summary, indent=2))
"""


def main() -> None:
    for screen in SCREENS:
        target = KERNELS / screen["folder"]
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath("kernel-metadata.json").write_text(
            json.dumps(
                metadata(screen["kernel"], screen["title"], False),
                indent=2,
            ),
            encoding="utf-8",
        )
        slug = screen["kernel"].removeprefix("ptcg-cpu-")
        code = RUN.format(slug=slug, **screen)
        target.joinpath("notebook.ipynb").write_text(
            json.dumps(
                notebook(
                    [
                        (
                            "markdown",
                            f"# {screen['experiment']} {screen['strategy']}\n\n"
                            "Original policy logic against frozen local v3. "
                            "No public policy code or action labels are used.",
                        ),
                        ("code", SETUP),
                        ("code", code),
                    ]
                ),
                indent=1,
            ),
            encoding="utf-8",
        )
        print(target)


if __name__ == "__main__":
    main()
