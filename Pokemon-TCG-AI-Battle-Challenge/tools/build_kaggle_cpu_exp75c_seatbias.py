"""Build the next two independent Uza CPU experiments."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"
DATASET = "divyanshuboltuzamaki/ptcg-research-assets-uza-v1"


def write_kernel(folder: str, metadata: dict, title: str, run: str) -> None:
    target = KERNELS / folder
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    target.joinpath("notebook.ipynb").write_text(
        json.dumps(
            notebook(
                [
                    ("markdown", title),
                    ("code", SETUP),
                    ("code", run),
                ]
            ),
            indent=1,
        ),
        encoding="utf-8",
    )
    print(target)


def metadata(kernel_id: str, title: str) -> dict:
    return {
        "id": f"divyanshuboltuzamaki/{kernel_id}",
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


def main() -> None:
    confirmation_output = "/kaggle/working/exp-75c-live-field"
    confirmation_run = f"""\
from pathlib import Path
import json, subprocess, sys

command = [
    sys.executable,
    "automation/jobs/live_deck_gauntlet.py",
    "--candidate", "risksensitive",
    "--risk-weight", "0.45",
    "--risk-iterations", "36",
    "--flat-root-width", "8",
    "--games-per-deck", "24",
    "--max-decks", "20",
    "--deck-catalog", "data/live_decks.json",
    "--out", {confirmation_output!r},
]
completed = subprocess.run(command, check=False)
if completed.returncode:
    raise RuntimeError("EXP-75C failed with code " + str(completed.returncode))
summary = json.loads(
    Path({(confirmation_output + ".json")!r}).read_text(encoding="utf-8")
)
summary["experiment"] = "EXP-75C"
summary["fresh_confirmation"] = True
summary["promotion_gate"] = "95 percent Wilson lower bound above 0.50"
summary["public_policy_code_used"] = False
summary["public_policy_actions_used_as_labels"] = False
summary["submission_performed"] = False
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
print(json.dumps(summary, indent=2))
"""
    write_kernel(
        "cpu_risk_width8_confirmation",
        metadata(
            "ptcg-cpu-risk-width-8-confirmation",
            "PTCG CPU risk width 8 confirmation",
        ),
        (
            "# EXP-75C Risk width 8 fresh confirmation\n\n"
            "This independent 480-game run is the only evidence that can promote "
            "the exploratory 126 to 114 opening."
        ),
        confirmation_run,
    )

    seat_bias_run = """\
from pathlib import Path
import json, statistics, time

from kaggle_environments import make
import archaludon_policy
import arena
import candidate_agents
import information_search
import search_agent

catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
decks = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        decks.append(record)
    if len(decks) == 8:
        break
if not decks:
    raise RuntimeError("no legal held-out decks")

def risk_width8(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        if information_search._eligible(obs_dict):
            return [
                information_search.risk_sensitive_choice(
                    obs_dict,
                    deck,
                    iterations=36,
                    root_width=8,
                    downside_weight=0.45,
                )
            ]
    except Exception:
        pass
    return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)

policies = {
    "sequenced": archaludon_policy.archaludon_sequenced_agent,
    "progressive": candidate_agents.arch_progressive_widening,
    "chance_coupled": candidate_agents.arch_chance_coupled,
    "risk_width8": risk_width8,
}
summary = {}
for policy_name, policy_fn in policies.items():
    candidate = arena.bind_deck_arg(policy_fn, archaludon_policy.ARCHALUDON_DECK)
    seats = {
        "first": {"games": 0, "wins": 0},
        "second": {"games": 0, "wins": 0},
    }
    failures = 0
    game_errors = 0
    times = []
    for record in decks:
        opponent = arena.bind_deck(
            search_agent.agent,
            record["deck"],
            search_module=search_agent,
        )
        for game in range(12):
            candidate_seat = game % 2
            started = time.time()
            if candidate_seat == 0:
                outcome = arena.play_game(make, candidate, opponent)
                candidate_failure = outcome.seat0_fail
                seat_name = "first"
            else:
                outcome = arena.play_game(make, opponent, candidate)
                candidate_failure = outcome.seat1_fail
                seat_name = "second"
            times.append(time.time() - started)
            seats[seat_name]["games"] += 1
            if outcome.winner == candidate_seat:
                seats[seat_name]["wins"] += 1
            failures += int(candidate_failure is not None)
            game_errors += int(outcome.game_error)
    for seat in seats.values():
        seat["winrate"] = seat["wins"] / max(seat["games"], 1)
        low, high = arena.wilson_interval(seat["wins"], seat["games"])
        seat["wilson_low"] = low
        seat["wilson_high"] = high
    summary[policy_name] = {
        "seats": seats,
        "absolute_gap": abs(
            seats["first"]["winrate"] - seats["second"]["winrate"]
        ),
        "candidate_failures": failures,
        "game_errors": game_errors,
        "mean_match_s": statistics.mean(times),
        "p95_match_s": arena.percentile(times, 0.95),
    }

report = {
    "experiment": "EXP-79",
    "strategy_rank": 66,
    "strategy": "Seat bias correction",
    "held_out_decks": len(decks),
    "games_per_policy": 96,
    "policies": summary,
    "candidate_failures": sum(
        row["candidate_failures"] for row in summary.values()
    ),
    "game_errors": sum(row["game_errors"] for row in summary.values()),
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
"""
    write_kernel(
        "cpu_seat_bias_correction",
        metadata(
            "ptcg-cpu-seat-bias-correction",
            "PTCG CPU seat bias correction",
        ),
        (
            "# EXP-79 Seat bias correction\n\n"
            "Four original policies are evaluated equally from first and second "
            "seat against the same held-out decks."
        ),
        seat_bias_run,
    )


if __name__ == "__main__":
    main()
