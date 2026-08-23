"""Build the Uza within-match opponent adaptation experiment."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_within_match_adaptation"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "divyanshuboltuzamaki/ptcg-cpu-within-match-adaptation",
        "title": "PTCG CPU within match adaptation",
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
import archaludon_policy
import arena
import candidate_agents
import search_agent
from decks import MEGA_ABOMASNOW, MEGA_LUCARIO, DRAGAPULT, ALAKAZAM

policies = {
    "sequenced": archaludon_policy.archaludon_sequenced_agent,
    "progressive": candidate_agents.arch_progressive_widening,
    "risk_sensitive": candidate_agents.arch_risk_sensitive,
    "chance_coupled": candidate_agents.arch_chance_coupled,
}
archetype_decks = {
    "abomasnow": MEGA_ABOMASNOW,
    "lucario": MEGA_LUCARIO,
    "dragapult": DRAGAPULT,
    "alakazam": ALAKAZAM,
}
static_matrix = {}
mapping = {}
for archetype, opponent_deck in archetype_decks.items():
    opponent = arena.bind_deck(
        search_agent.agent,
        opponent_deck,
        search_module=search_agent,
    )
    static_matrix[archetype] = {}
    for policy_name, policy_fn in policies.items():
        candidate = arena.bind_deck_arg(
            policy_fn, archaludon_policy.ARCHALUDON_DECK
        )
        pairing = arena.run_round_robin(
            make,
            [
                {
                    "name": policy_name,
                    "agent": candidate,
                    "deck": archaludon_policy.ARCHALUDON_DECK,
                },
                {
                    "name": "v3-" + archetype,
                    "agent": opponent,
                    "deck": opponent_deck,
                },
            ],
            12,
        )[0]
        static_matrix[archetype][policy_name] = arena.pairing_row(pairing)
    mapping[archetype] = max(
        policies,
        key=lambda name: static_matrix[archetype][name]["a_wins"],
    )

def adaptive_agent(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    archetype = candidate_agents._visible_archetype(obs_dict)
    policy_name = mapping.get(archetype, "chance_coupled")
    return policies[policy_name](obs_dict, deck)

catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
selected = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        selected.append(record)
    if len(selected) == 20:
        break

def live_screen(name, policy_fn):
    candidate = arena.bind_deck_arg(
        policy_fn, archaludon_policy.ARCHALUDON_DECK
    )
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
                    "name": "live-v3-" + record["signature"],
                    "agent": opponent,
                    "deck": record["deck"],
                },
            ],
            6,
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

adaptive = live_screen("adaptive-selector", adaptive_agent)
control = live_screen("chance-control", policies["chance_coupled"])
report = {
    "experiment": "EXP-84",
    "strategy_rank": 55,
    "strategy": "Within match opponent adaptation",
    "static_mapping": mapping,
    "static_matrix": static_matrix,
    "adaptive_live": adaptive,
    "chance_control_live": control,
    "winrate_change": adaptive["winrate"] - control["winrate"],
    "promotion_ready": bool(
        adaptive["wilson_low"] > 0.50
        and adaptive["winrate"] > control["winrate"]
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
                        "# EXP-84 Within-match opponent adaptation\n\n"
                        "A policy map is learned on four frozen archetype decks, "
                        "then compared with chance-coupled control on held-out "
                        "live decks.",
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
