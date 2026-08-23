"""Build the fresh 480-game no-Judge balanced-deck confirmation."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_balanced_confirmation"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "divyanshuboltuzamaki/ptcg-cpu-balanced-no-judge-confirmation",
        "title": "PTCG CPU balanced no Judge confirmation",
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
import deck_variants
import search_agent

deck = list(deck_variants.RELIC_NO_JUDGE)
legal, reason = arena.check_deck_legal(deck)
if not legal:
    raise RuntimeError("balanced deck is illegal: " + reason)
candidate = arena.bind_deck_arg(
    archaludon_policy.archaludon_sequenced_agent, deck
)
catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
selected = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        selected.append(record)
    if len(selected) == 20:
        break
rows = []
for record in selected:
    opponent = arena.bind_deck(
        search_agent.agent, record["deck"], search_module=search_agent
    )
    pairing = arena.run_round_robin(
        make,
        [
            {"name": "ours-balanced-no-judge", "agent": candidate, "deck": deck},
            {
                "name": "live-v3-" + record["signature"],
                "agent": opponent,
                "deck": record["deck"],
            },
        ],
        24,
    )[0]
    row = arena.pairing_row(pairing)
    row["live_deck_signature"] = record["signature"]
    rows.append(row)
games = sum(row["games"] for row in rows)
wins = sum(row["a_wins"] for row in rows)
losses = sum(row["a_losses"] for row in rows)
low, high = arena.wilson_interval(wins, games)
report = {
    "experiment": "EXP-96",
    "candidate": "balanced no-Judge Archaludon",
    "opponent_policy": "frozen local v3 search",
    "total_games": games,
    "wins": wins,
    "losses": losses,
    "draws": games - wins - losses,
    "winrate": wins / games,
    "wilson_low": low,
    "wilson_high": high,
    "candidate_failures": sum(
        row["crashes"] + row["invalids"] + row["timeouts"] for row in rows
    ),
    "game_errors": sum(row["game_errors"] for row in rows),
    "worst_pairing_p95_s": max(row["p95_match_s"] for row in rows),
    "pairings": rows,
    "promotion_ready": bool(low > 0.50),
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
                        "# EXP-96 Balanced no-Judge confirmation\n\n"
                        "The exact audited deck and original sequenced policy "
                        "receive a fresh 480-game broad live-field gate.",
                    ),
                    ("code", SETUP),
                    ("code", run),
                ]
            ),
            indent=1,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
