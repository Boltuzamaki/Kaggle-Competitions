"""Build a fresh live-field screen for the prioritized failure model."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_exp82_arena_bolt"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "boltuzamaki/ptcg-cpu-prioritized-model-screen",
        "title": "PTCG CPU prioritized model screen",
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
        "kernel_sources": [
            "boltuzamaki/ptcg-gpu-prioritized-failure-v1"
        ],
        "competition_sources": [],
        "model_sources": [],
    }
    run = """\
from pathlib import Path
import glob, json, sys, torch

from cg.api import SelectContext, to_observation_class
from kaggle_environments import make
import archaludon_policy
import arena
import candidate_agents
import search_agent
from automation.jobs import train_beam_distillation_gpu as base

matches = glob.glob(
    "/kaggle/input/**/prioritized_failure_student.pth", recursive=True
)
if not matches:
    raise RuntimeError("EXP-82 checkpoint was not attached")
checkpoint = torch.load(matches[0], map_location="cpu", weights_only=False)
model = base.BeamStudent(width=int(checkpoint["width"]))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

def prioritized_agent(obs_dict, deck):
    if obs_dict.get("select") is None:
        return list(deck)
    try:
        obs = to_observation_class(obs_dict)
        if (
            obs.select is None
            or obs.select.context != SelectContext.MAIN
            or not (1 < len(obs.select.option) <= base.MAX_OPTIONS)
        ):
            return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)
        override = archaludon_policy.hard_override(obs)
        if override is not None:
            return [override]
        policy = archaludon_policy.SequencedArchaludonPolicy(obs)
        features = torch.tensor(
            [
                base._option_features(policy, option)
                for option in obs.select.option
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            choice = int(model(features).argmax().item())
        return [choice]
    except Exception:
        return archaludon_policy.archaludon_sequenced_agent(obs_dict, deck)

catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
selected = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        selected.append(record)
    if len(selected) == 20:
        break
candidate = arena.bind_deck_arg(
    prioritized_agent, archaludon_policy.ARCHALUDON_DECK
)
rows = []
for record in selected:
    opponent = arena.bind_deck(
        search_agent.agent,
        record["deck"],
        search_module=search_agent,
    )
    pairings = arena.run_round_robin(
        make,
        [
            {
                "name": "ours-prioritized-model",
                "agent": candidate,
                "deck": archaludon_policy.ARCHALUDON_DECK,
            },
            {
                "name": "live-v3-" + record["signature"],
                "agent": opponent,
                "deck": record["deck"],
            },
        ],
        12,
    )
    row = arena.pairing_row(pairings[0])
    row["live_deck_signature"] = record["signature"]
    rows.append(row)
games = sum(row["games"] for row in rows)
wins = sum(row["a_wins"] for row in rows)
losses = sum(row["a_losses"] for row in rows)
low, high = arena.wilson_interval(wins, games)
report = {
    "experiment": "EXP-82A",
    "candidate": "prioritized rare failure model",
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
    "hard_tactical_overrides_preserved": True,
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
                        "# EXP-82A Prioritized model live-field screen\n\n"
                        "The model keeps unconditional lethal and escape "
                        "overrides, then handles only main-phase choices.",
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
