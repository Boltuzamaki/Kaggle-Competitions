"""Build seven independent Kaggle CPU notebooks for the next strategy wave."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, metadata, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"


def command_cell(command, summary_path):
    return f"""\
from pathlib import Path
import json, subprocess, sys

command = {command!r}
command[0] = sys.executable
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("experiment failed with code " + str(completed.returncode))
summary = json.loads(Path({summary_path!r}).read_text(encoding="utf-8"))
summary["public_policy_code_used"] = False
summary["public_policy_actions_used_as_labels"] = False
summary["submission_performed"] = False
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
"""


def write(folder, kernel_id, title, experiment, strategy, code):
    target = KERNELS / folder
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("kernel-metadata.json").write_text(
        json.dumps(metadata(kernel_id, title, False), indent=2),
        encoding="utf-8",
    )
    target.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    (
                        "markdown",
                        f"# {experiment} {strategy}\n\n"
                        "Clean-room candidate or evaluation logic. No public policy "
                        "code and no public actions used as labels.",
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


def live_command(candidate, output, games=12, decks=20):
    return [
        "python",
        "automation/jobs/live_deck_gauntlet.py",
        "--candidate",
        candidate,
        "--games-per-deck",
        str(games),
        "--max-decks",
        str(decks),
        "--deck-catalog",
        "data/live_decks.json",
        "--out",
        output,
    ]


def comparison_cell(candidate, slug):
    control = f"/kaggle/working/{slug}_control"
    treatment = f"/kaggle/working/{slug}_treatment"
    return f"""\
from pathlib import Path
import json, subprocess, sys

commands = [
    {live_command("flatmc", control, 6, 6)!r},
    {live_command(candidate, treatment, 6, 6)!r},
]
for command in commands:
    command[0] = sys.executable
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError("comparison arm failed with code " + str(completed.returncode))
control = json.loads(Path({(control + ".json")!r}).read_text(encoding="utf-8"))
treatment = json.loads(Path({(treatment + ".json")!r}).read_text(encoding="utf-8"))
def p95(report):
    return max((float(row.get("p95_match_s") or 0.0) for row in report["pairings"]), default=0.0)
summary = {{
    "control": {{"candidate": control["candidate"], "games": control["total_games"],
                 "winrate": control["winrate"], "p95_match_s": p95(control),
                 "failures": control["candidate_failures"] + control["game_errors"]}},
    "treatment": {{"candidate": treatment["candidate"], "games": treatment["total_games"],
                   "winrate": treatment["winrate"], "p95_match_s": p95(treatment),
                   "failures": treatment["candidate_failures"] + treatment["game_errors"]}},
    "speed_gain": 1.0 - p95(treatment) / max(p95(control), 1e-9),
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}}
Path("/kaggle/working/final_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
"""


def main():
    write(
        "cpu_risk_retreat",
        "ptcg-cpu-risk-retreat-live-field",
        "PTCG CPU risk retreat live field",
        "EXP-56",
        "Risk aware retreat selector",
        command_cell(
            live_command("riskretreat", "/kaggle/working/risk-retreat-live-field"),
            "/kaggle/working/risk-retreat-live-field.json",
        ),
    )
    write(
        "cpu_hand_quality",
        "ptcg-cpu-hand-quality-live-field-v2",
        "PTCG CPU hand quality live field v2",
        "EXP-57",
        "Hand quality estimator",
        command_cell(
            live_command("handquality", "/kaggle/working/hand-quality-live-field"),
            "/kaggle/working/hand-quality-live-field.json",
        ),
    )
    write(
        "cpu_state_cache",
        "ptcg-cpu-state-cache-comparison-v2",
        "PTCG CPU state cache comparison v2",
        "EXP-58",
        "State transposition cache",
        comparison_cell("cacheflat", "state_cache"),
    )
    write(
        "cpu_selective_search",
        "ptcg-cpu-selective-search-comparison",
        "PTCG CPU selective search comparison",
        "EXP-59",
        "Selective search trigger",
        comparison_cell("gatedflat", "selective_search"),
    )
    write(
        "cpu_sequential_test",
        "ptcg-cpu-sequential-probability-test-v2",
        "PTCG CPU sequential probability test v2",
        "EXP-60",
        "Sequential probability testing",
        command_cell(
            [
                "python",
                "automation/jobs/sequential_probability_test.py",
                "--out",
                "/kaggle/working/sequential_probability.json",
            ],
            "/kaggle/working/sequential_probability.json",
        ),
    )
    write(
        "cpu_invariant_check",
        "ptcg-cpu-decision-invariant-check",
        "PTCG CPU decision invariant check",
        "EXP-61",
        "Decision invariant checker",
        command_cell(
            [
                "python",
                "automation/jobs/decision_invariant_check.py",
                "--deck-catalog",
                "data/live_decks.json",
                "--out",
                "/kaggle/working/decision_invariants.json",
            ],
            "/kaggle/working/decision_invariants.json",
        ),
    )
    write(
        "cpu_strategy_fusion",
        "ptcg-cpu-strategy-fusion-audit-v2",
        "PTCG CPU strategy fusion audit v2",
        "EXP-62",
        "Strategy fusion detector",
        command_cell(
            [
                "python",
                "automation/jobs/strategy_fusion_audit.py",
                "--games",
                "12",
                "--samples",
                "8",
                "--max-decisions",
                "40",
                "--out",
                "/kaggle/working/strategy_fusion.json",
            ],
            "/kaggle/working/strategy_fusion.json",
        ),
    )
    write(
        "cpu_particle_belief_1x48",
        "ptcg-cpu-particle-belief-1x48",
        "PTCG CPU particle belief 1x48",
        "EXP-63A",
        "Particle belief search allocation 1 by 48",
        command_cell(
            live_command(
                "belief1x48",
                "/kaggle/working/particle-belief-1x48",
                8,
                10,
            ),
            "/kaggle/working/particle-belief-1x48.json",
        ),
    )
    write(
        "cpu_particle_belief_3x16",
        "ptcg-cpu-particle-belief-3x16",
        "PTCG CPU particle belief 3x16",
        "EXP-63B",
        "Particle belief search allocation 3 by 16",
        command_cell(
            live_command(
                "belief3x16",
                "/kaggle/working/particle-belief-3x16",
                8,
                10,
            ),
            "/kaggle/working/particle-belief-3x16.json",
        ),
    )


if __name__ == "__main__":
    main()
