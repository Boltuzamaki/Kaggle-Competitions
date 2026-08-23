"""Build independent Kaggle CPU kernels for the original v3 policy deck portfolio."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, metadata, notebook


ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kaggle_remote" / "kernels"

DECKS = {
    "abomasnow": ("MEGA_ABOMASNOW", "Abomasnow"),
    "lucario": ("MEGA_LUCARIO", "Lucario"),
    "alakazam": ("ALAKAZAM", "Alakazam"),
    "dragapult": ("DRAGAPULT", "Dragapult"),
}

RUN = """\
from pathlib import Path
import json

import arena
import search_agent
from decks import {deck_constant}
from kaggle_environments import make

catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
field_decks = [item for item in catalog["decks"] if len(item.get("deck", [])) == 60][:20]
candidate_deck = list({deck_constant})
candidate = {{
    "name": "our-v3-{slug}",
    "agent": arena.bind_deck(search_agent.agent, candidate_deck, search_module=search_agent),
    "deck": candidate_deck,
}}

rows = []
for index, field in enumerate(field_decks, 1):
    opponent_deck = list(field["deck"])
    opponent = {{
        "name": f"frozen-v3-field-{{index:02d}}",
        "agent": arena.bind_deck(search_agent.agent, opponent_deck, search_module=search_agent),
        "deck": opponent_deck,
    }}
    pairing = arena.run_round_robin(make, [candidate, opponent], games=16)[0]
    row = arena.pairing_row(pairing)
    row["field_signature"] = field.get("signature")
    row["field_appearances"] = field.get("appearances")
    rows.append(row)

wins = sum(row["a_wins"] for row in rows)
losses = sum(row["a_losses"] for row in rows)
draws = sum(row["draws"] for row in rows)
games = sum(row["games"] for row in rows)
failures = sum(
    row["crashes"] + row["invalids"] + row["timeouts"] + row["game_errors"]
    for row in rows
)
result = {{
    "experiment": "EXP-47{suffix}",
    "candidate_policy": "our frozen v3 shallow deterministic search",
    "candidate_deck": "{deck_name}",
    "games_per_field_deck": 16,
    "field_decks": len(rows),
    "games": games,
    "wins": wins,
    "losses": losses,
    "draws": draws,
    "winrate": wins / games if games else 0.0,
    "failures": failures,
    "pairings": rows,
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}}
output = Path("/kaggle/working/exp47_{slug}_live_decks.json")
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps({{key: value for key, value in result.items() if key != "pairings"}}, indent=2))
"""


def main() -> None:
    for index, (slug, (constant, deck_name)) in enumerate(DECKS.items()):
        suffix = chr(ord("A") + index)
        kernel_id = f"ptcg-cpu-original-v3-{slug}-live-field"
        target = KERNELS / f"cpu_v3_deck_{slug}"
        target.mkdir(parents=True, exist_ok=True)
        target.joinpath("kernel-metadata.json").write_text(
            json.dumps(
                metadata(
                    kernel_id,
                    f"PTCG CPU original v3 {deck_name} live field",
                    False,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        code = RUN.format(
            deck_constant=constant,
            slug=slug,
            suffix=suffix,
            deck_name=deck_name,
        )
        target.joinpath("notebook.ipynb").write_text(
            json.dumps(
                notebook(
                    [
                        (
                            "markdown",
                            "Original frozen v3 policy deck comparison. "
                            "No public policy code or public action labels are used.",
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
