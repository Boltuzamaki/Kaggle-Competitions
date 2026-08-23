"""Build the Uza bounded exhaustive endgame-search experiment."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_endgame_solver"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "divyanshuboltuzamaki/ptcg-cpu-small-endgame-solver",
        "title": "PTCG CPU small endgame solver",
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
import json

from kaggle_environments import make
from cg.api import SelectContext, to_observation_class
import archaludon_policy
import arena
import candidate_agents
import information_search
import search_agent

class EndgameAgent:
    def __init__(self):
        self.activations = 0
        self.fallbacks = 0

    def agent(self, obs_dict, deck):
        if obs_dict.get("select") is None:
            return list(deck)
        try:
            obs = to_observation_class(obs_dict)
            override = archaludon_policy.hard_override(obs)
            if override is not None:
                return [override]
            policy = archaludon_policy.SequencedArchaludonPolicy(obs)
            prize_critical = (
                len(policy.mine.prize) <= 2
                or len(policy.theirs.prize) <= 2
            )
            option_count = len(obs.select.option) if obs.select else 0
            if (
                obs.select is not None
                and obs.select.context == SelectContext.MAIN
                and prize_critical
                and 1 < option_count <= 12
            ):
                self.activations += 1
                return information_search.flat_root_agent_config(
                    obs_dict,
                    deck,
                    iterations=max(96, option_count * 16),
                    root_width=option_count,
                )
        except Exception:
            self.fallbacks += 1
        return candidate_agents.arch_chance_coupled(obs_dict, deck)

catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
selected = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        selected.append(record)
    if len(selected) == 12:
        break

def screen(name, fn):
    candidate = arena.bind_deck_arg(fn, archaludon_policy.ARCHALUDON_DECK)
    rows = []
    for record in selected:
        opponent = arena.bind_deck(
            search_agent.agent,
            record["deck"],
            search_module=search_agent,
        )
        pairing = arena.run_round_robin(
            make,
            [
                {
                    "name": name,
                    "agent": candidate,
                    "deck": archaludon_policy.ARCHALUDON_DECK,
                },
                {
                    "name": "heldout-v3-" + record["signature"],
                    "agent": opponent,
                    "deck": record["deck"],
                },
            ],
            8,
        )[0]
        row = arena.pairing_row(pairing)
        row["live_deck_signature"] = record["signature"]
        rows.append(row)
    games = sum(row["games"] for row in rows)
    wins = sum(row["a_wins"] for row in rows)
    losses = sum(row["a_losses"] for row in rows)
    low, high = arena.wilson_interval(wins, games)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": games - wins - losses,
        "winrate": wins / games,
        "wilson_low": low,
        "wilson_high": high,
        "candidate_failures": sum(
            row["crashes"] + row["invalids"] + row["timeouts"]
            for row in rows
        ),
        "game_errors": sum(row["game_errors"] for row in rows),
        "worst_pairing_p95_s": max(row["p95_match_s"] for row in rows),
        "pairings": rows,
    }

solver = EndgameAgent()
treatment = screen("bounded-endgame", solver.agent)
control = screen("chance-control", candidate_agents.arch_chance_coupled)
report = {
    "experiment": "EXP-87",
    "strategy_rank": 57,
    "strategy": "Exact small endgame solver",
    "implementation": "root-action exhaustive bounded endgame screen",
    "activations": solver.activations,
    "fallbacks": solver.fallbacks,
    "treatment": treatment,
    "control": control,
    "winrate_change": treatment["winrate"] - control["winrate"],
    "ready_for_deeper_exact_solver": bool(
        solver.activations >= 20
        and treatment["winrate"] > control["winrate"]
        and treatment["candidate_failures"] == 0
    ),
    "promotion_ready": False,
    "interpretation": (
        "This first screen exhausts legal root actions in small prize-critical "
        "states. It is not a claim of solving the full remaining game tree."
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
                        "# EXP-87 Small endgame solver screen\n\n"
                        "Prize-critical states with at most twelve legal "
                        "actions receive exhaustive root coverage and a larger "
                        "bounded rollout budget.",
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
