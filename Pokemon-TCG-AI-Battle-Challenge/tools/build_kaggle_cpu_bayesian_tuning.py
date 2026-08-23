"""Build the Uza Bayesian deck and policy tuning screen."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_bayesian_deck_policy"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "divyanshuboltuzamaki/ptcg-cpu-bayesian-deck-policy",
        "title": "PTCG CPU Bayesian deck policy",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["research"],
        "dataset_sources": [
            "divyanshuboltuzamaki/ptcg-research-assets-uza-v1",
            "kiyotah/cg-lib",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    run = """\
from pathlib import Path
import json, random, statistics, time

from kaggle_environments import make
import archaludon_policy
import arena
import candidate_agents
import deck_variants
import search_agent

rng = random.Random(2026071908)
catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
held_out = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        held_out.append(record)
    if len(held_out) == 20:
        break
if len(held_out) < 8:
    raise RuntimeError("insufficient legal held-out decks")

policy_functions = {
    "sequenced": archaludon_policy.archaludon_sequenced_agent,
    "progressive": candidate_agents.arch_progressive_widening,
    "chance_coupled": candidate_agents.arch_chance_coupled,
    "risk_sensitive": candidate_agents.arch_risk_sensitive,
}
deck_names = (
    "baseline",
    "no-relicanth",
    "relic-no-judge",
    "less-healing",
    "consistency",
)
arms = {}
for deck_name in deck_names:
    deck = deck_variants.VARIANTS[deck_name]
    legal, reason = arena.check_deck_legal(deck)
    if not legal:
        raise RuntimeError(deck_name + " is illegal: " + reason)
    for policy_name, policy_fn in policy_functions.items():
        key = deck_name + "|" + policy_name
        arms[key] = {
            "deck_name": deck_name,
            "policy_name": policy_name,
            "deck": deck,
            "agent": arena.bind_deck_arg(policy_fn, deck),
            "alpha": 1.0,
            "beta": 1.0,
            "games": 0,
            "wins": 0,
            "draws": 0,
            "failures": 0,
            "game_errors": 0,
            "times": [],
        }
opponents = [
    arena.bind_deck(
        search_agent.agent,
        record["deck"],
        search_module=search_agent,
    )
    for record in held_out
]

def play_arm(key, trial):
    arm = arms[key]
    opponent = opponents[trial % len(opponents)]
    candidate_seat = arm["games"] % 2
    started = time.time()
    if candidate_seat == 0:
        outcome = arena.play_game(make, arm["agent"], opponent)
        candidate_failure = outcome.seat0_fail
    else:
        outcome = arena.play_game(make, opponent, arm["agent"])
        candidate_failure = outcome.seat1_fail
    arm["times"].append(time.time() - started)
    arm["games"] += 1
    if outcome.winner == candidate_seat:
        arm["wins"] += 1
        arm["alpha"] += 1.0
    elif outcome.winner is None:
        arm["draws"] += 1
        arm["alpha"] += 0.5
        arm["beta"] += 0.5
    else:
        arm["beta"] += 1.0
    arm["failures"] += int(candidate_failure is not None)
    arm["game_errors"] += int(outcome.game_error)

trial = 0
for key in arms:
    for _ in range(2):
        play_arm(key, trial)
        trial += 1
while trial < 480:
    sampled = {
        key: rng.betavariate(arm["alpha"], arm["beta"])
        for key, arm in arms.items()
    }
    selected = max(sampled, key=sampled.get)
    play_arm(selected, trial)
    trial += 1

screen = {}
for key, arm in arms.items():
    screen[key] = {
        "deck": arm["deck_name"],
        "policy": arm["policy_name"],
        "games": arm["games"],
        "wins": arm["wins"],
        "draws": arm["draws"],
        "raw_winrate": arm["wins"] / max(arm["games"], 1),
        "posterior_mean": arm["alpha"] / (arm["alpha"] + arm["beta"]),
        "candidate_failures": arm["failures"],
        "game_errors": arm["game_errors"],
        "mean_match_s": statistics.mean(arm["times"]),
        "p95_match_s": arena.percentile(arm["times"], 0.95),
    }
top_keys = sorted(
    screen,
    key=lambda key: screen[key]["posterior_mean"],
    reverse=True,
)[:3]

confirmation = {}
for key in top_keys:
    arm = arms[key]
    wins = draws = failures = errors = 0
    times = []
    for game in range(64):
        opponent = opponents[(game + 8) % len(opponents)]
        candidate_seat = game % 2
        started = time.time()
        if candidate_seat == 0:
            outcome = arena.play_game(make, arm["agent"], opponent)
            candidate_failure = outcome.seat0_fail
        else:
            outcome = arena.play_game(make, opponent, arm["agent"])
            candidate_failure = outcome.seat1_fail
        times.append(time.time() - started)
        wins += int(outcome.winner == candidate_seat)
        draws += int(outcome.winner is None)
        failures += int(candidate_failure is not None)
        errors += int(outcome.game_error)
    low, high = arena.wilson_interval(wins, 64)
    confirmation[key] = {
        "games": 64,
        "wins": wins,
        "draws": draws,
        "winrate": wins / 64,
        "wilson_low": low,
        "wilson_high": high,
        "candidate_failures": failures,
        "game_errors": errors,
        "mean_match_s": statistics.mean(times),
        "p95_match_s": arena.percentile(times, 0.95),
    }

report = {
    "experiment": "EXP-81",
    "strategy_rank": 52,
    "strategy": "Bayesian deck and policy tuning",
    "screen_games": 480,
    "arms": len(arms),
    "screen": screen,
    "top_arms": top_keys,
    "holdout_confirmation": confirmation,
    "promotion_ready": False,
    "interpretation": (
        "Adaptive screening is selection biased. Any winning arm requires "
        "a separate predeclared 480-game confirmation."
    ),
    "candidate_failures": sum(
        row["candidate_failures"] for row in screen.values()
    ) + sum(
        row["candidate_failures"] for row in confirmation.values()
    ),
    "game_errors": sum(row["game_errors"] for row in screen.values()) + sum(
        row["game_errors"] for row in confirmation.values()
    ),
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}
Path("/kaggle/working/final_summary.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
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
                        "# EXP-81 Bayesian deck and policy tuning\n\n"
                        "Thompson sampling allocates games across our legal deck "
                        "variants and original policies. Top arms receive a "
                        "separate internal holdout, but cannot promote directly.",
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
