"""Build EXP-97 option-order invariance repair and fresh arena gate."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_exp97_order_invariance"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    policy_source = (
        ROOT / "agent" / "archaludon_policy.py"
    ).read_text(encoding="utf-8")
    metadata = {
        "id": "divyanshuboltuzamaki/ptcg-cpu-option-order-invariance",
        "title": "PTCG CPU option order invariance",
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
    run = f"""\
from copy import deepcopy
from pathlib import Path
import json

policy_path = Path("/kaggle/working/ptcg/agent/archaludon_policy.py")
policy_path.write_text({policy_source!r}, encoding="utf-8")

from kaggle_environments import make
import archaludon_policy
import arena
import deck_variants
import search_agent
from decks import ALAKAZAM

checks = 0
mismatches = []
deck = list(deck_variants.NO_RELICANTH)

def semantic(option):
    return json.dumps(option, sort_keys=True, default=str)

def checked(obs):
    global checks
    base = archaludon_policy.archaludon_sequenced_agent(obs, deck)
    select = obs.get("select") or {{}}
    options = select.get("option") or []
    if (
        len(options) > 1
        and (select.get("maxCount") or 1) == 1
        and len(base) == 1
    ):
        transformed = deepcopy(obs)
        transformed["select"]["option"] = list(reversed(options))
        other = archaludon_policy.archaludon_sequenced_agent(
            transformed, deck
        )
        checks += 1
        original_choice = semantic(options[base[0]])
        reversed_options = transformed["select"]["option"]
        transformed_choice = (
            semantic(reversed_options[other[0]]) if len(other) == 1 else None
        )
        if original_choice != transformed_choice:
            mismatches.append(
                {{
                    "turn": (obs.get("current") or {{}}).get("turn"),
                    "options": len(options),
                }}
            )
    return base

opponent = arena.bind_deck(
    search_agent.agent, ALAKAZAM, search_module=search_agent
)
metamorphic_pairing = arena.run_round_robin(
    make,
    [
        {{"name": "order-invariant", "agent": checked, "deck": deck}},
        {{"name": "v3-alakazam", "agent": opponent, "deck": ALAKAZAM}},
    ],
    24,
)[0]
mismatch_rate = len(mismatches) / max(checks, 1)

rows = []
if mismatch_rate <= 0.01:
    catalog = json.loads(
        Path("data/live_decks.json").read_text(encoding="utf-8")
    )
    records = catalog.get("decks", catalog)
    selected = []
    for record in records:
        legal, _ = arena.check_deck_legal(record["deck"])
        if legal:
            selected.append(record)
        if len(selected) == 20:
            break
    candidate = arena.bind_deck_arg(
        archaludon_policy.archaludon_sequenced_agent, deck
    )
    for record in selected:
        live_opponent = arena.bind_deck(
            search_agent.agent,
            record["deck"],
            search_module=search_agent,
        )
        pairing = arena.run_round_robin(
            make,
            [
                {{
                    "name": "ours-order-invariant-no-relic",
                    "agent": candidate,
                    "deck": deck,
                }},
                {{
                    "name": "live-v3-" + record["signature"],
                    "agent": live_opponent,
                    "deck": record["deck"],
                }},
            ],
            12,
        )[0]
        row = arena.pairing_row(pairing)
        row["live_deck_signature"] = record["signature"]
        rows.append(row)

games = sum(row["games"] for row in rows)
wins = sum(row["a_wins"] for row in rows)
losses = sum(row["a_losses"] for row in rows)
low, high = arena.wilson_interval(wins, games) if games else (0.0, 0.0)
report = {{
    "experiment": "EXP-97",
    "candidate": "option-order-invariant no-Relicanth Archaludon",
    "metamorphic_checks": checks,
    "semantic_mismatches": len(mismatches),
    "mismatch_rate": mismatch_rate,
    "mismatch_gate_passed": mismatch_rate <= 0.01,
    "metamorphic_pairing": arena.pairing_row(metamorphic_pairing),
    "total_games": games,
    "wins": wins,
    "losses": losses,
    "draws": games - wins - losses,
    "winrate": wins / games if games else 0.0,
    "wilson_low": low,
    "wilson_high": high,
    "candidate_failures": sum(
        row["crashes"] + row["invalids"] + row["timeouts"] for row in rows
    ),
    "game_errors": sum(row["game_errors"] for row in rows),
    "worst_pairing_p95_s": max(
        (row["p95_match_s"] for row in rows), default=0.0
    ),
    "pairings": rows,
    "promotion_ready": bool(mismatch_rate <= 0.01 and low > 0.50),
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}}
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
                        "# EXP-97 Option-order invariance\n\n"
                        "This clean-room repair replaces list-position tie "
                        "breaks with stable semantic option keys. It must first "
                        "reduce EXP-93 mismatches from 31.29% to at most 1%.",
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
