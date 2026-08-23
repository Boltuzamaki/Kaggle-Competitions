"""Build the Bolt resource-shadow-pricing CPU screen."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_resource_pricing_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "boltuzamaki/ptcg-cpu-resource-shadow-pricing",
        "title": "PTCG CPU resource shadow pricing",
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["research"],
        "dataset_sources": [
            "boltuzamaki/ptcg-private-cpu-assets-bolt-v1",
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
from cg.api import AreaType, OptionType, to_observation_class
import archaludon_policy
import arena
import search_agent

class ShadowPricePolicy(archaludon_policy.SequencedArchaludonPolicy):
    price = 0.0

    def _main_score(self, option):
        score = super()._main_score(option)
        remaining_prizes = len(self.mine.prize)
        early_factor = max(0.0, (remaining_prizes - 2) / 4.0)
        if option.type == OptionType.ATTACH:
            target = archaludon_policy._card(
                self.obs, option.inPlayArea, option.inPlayIndex, self.me
            )
            energy = (
                archaludon_policy._energy_count(target) if target else 0
            )
            if energy >= 3:
                score -= self.price * (energy - 2)
        elif option.type == OptionType.PLAY:
            card = archaludon_policy._card(
                self.obs, AreaType.HAND, option.index, self.me
            )
            scarce = {
                archaludon_policy.BOSS,
                archaludon_policy.JUDGE,
                archaludon_policy.ULTRA_BALL,
                archaludon_policy.POKEGEAR,
            }
            if card is not None and card.id in scarce:
                score -= self.price * early_factor
        return score

def make_agent(price):
    def agent(obs_dict, deck):
        if obs_dict.get("select") is None:
            return list(deck)
        try:
            obs = to_observation_class(obs_dict)
            override = archaludon_policy.hard_override(obs)
            if override is not None:
                return [override]
            policy = ShadowPricePolicy(obs)
            policy.price = price
            return policy.choose()
        except Exception:
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
    return agent

catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
selected = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        selected.append(record)
    if len(selected) == 12:
        break

prices = {
    "control": 0.0,
    "price-20k": 20000.0,
    "price-60k": 60000.0,
    "price-120k": 120000.0,
}
results = {}
for name, price in prices.items():
    candidate = arena.bind_deck_arg(
        make_agent(price), archaludon_policy.ARCHALUDON_DECK
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
    results[name] = {
        "price": price,
        "games": games,
        "wins": wins,
        "losses": losses,
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
best = max(results, key=lambda name: results[name]["winrate"])
report = {
    "experiment": "EXP-88",
    "strategy_rank": 58,
    "strategy": "Resource shadow pricing",
    "results": results,
    "best_arm": best,
    "best_gain_over_control": (
        results[best]["winrate"] - results["control"]["winrate"]
    ),
    "ready_for_fresh_confirmation": bool(
        best != "control"
        and results[best]["winrate"] > results["control"]["winrate"]
        and results[best]["candidate_failures"] == 0
    ),
    "promotion_ready": False,
    "interpretation": (
        "This is a multi-arm exploratory screen. A selected price requires "
        "a separate predeclared fresh confirmation."
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
                        "# EXP-88 Resource shadow pricing\n\n"
                        "Three prices penalize premature use of visible scarce "
                        "resources while preserving unconditional tactical "
                        "overrides. A zero-price control uses the same policy.",
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
