"""Build the Uza state-cluster opening-book CPU experiment."""

from __future__ import annotations

import json
from pathlib import Path

from build_kaggle_lanes import SETUP, notebook


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "kaggle_remote" / "kernels" / "cpu_opening_book"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": "divyanshuboltuzamaki/ptcg-cpu-opening-state-book",
        "title": "PTCG CPU opening state book",
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
from collections import defaultdict
import json, statistics, time

from kaggle_environments import make
from cg.api import SelectContext, to_observation_class
import archaludon_policy
import arena
import candidate_agents
import search_agent

catalog = json.loads(Path("data/live_decks.json").read_text(encoding="utf-8"))
records = catalog.get("decks", catalog)
legal_records = []
for record in records:
    legal, _ = arena.check_deck_legal(record["deck"])
    if legal:
        legal_records.append(record)
    if len(legal_records) == 20:
        break
if len(legal_records) < 16:
    raise RuntimeError("at least 16 legal live decks are required")
train_records = legal_records[:8]
test_records = legal_records[8:20]

policies = {
    "sequenced": archaludon_policy.archaludon_sequenced_agent,
    "progressive": candidate_agents.arch_progressive_widening,
    "chance_coupled": candidate_agents.arch_chance_coupled,
    "risk_sensitive": candidate_agents.arch_risk_sensitive,
}

def opening_cluster(obs_dict):
    obs = to_observation_class(obs_dict)
    policy = archaludon_policy.SequencedArchaludonPolicy(obs)
    active = policy.active()
    energy = archaludon_policy._energy_count(active) if active else 0
    turn = min(int(getattr(policy.state, "turn", 0) or 0), 3)
    hand = min(int(getattr(policy.mine, "handCount", 0) or 0) // 3, 4)
    bench = min(len(policy.mine.bench), 3)
    options = min(len(policy.select.option) // 4, 5)
    return f"t{turn}|h{hand}|b{bench}|e{min(energy,3)}|o{options}"

class TrainingRecorder:
    def __init__(self, policy_fn):
        self.policy_fn = policy_fn
        self.cluster = None

    def reset(self):
        self.cluster = None

    def agent(self, obs_dict, deck):
        try:
            obs = to_observation_class(obs_dict)
            if (
                self.cluster is None
                and obs.select is not None
                and obs.select.context == SelectContext.MAIN
            ):
                self.cluster = opening_cluster(obs_dict)
        except Exception:
            pass
        return self.policy_fn(obs_dict, deck)

cluster_stats = defaultdict(
    lambda: defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0})
)
training_games = []
for policy_name, policy_fn in policies.items():
    recorder = TrainingRecorder(policy_fn)
    candidate = arena.bind_deck_arg(
        recorder.agent, archaludon_policy.ARCHALUDON_DECK
    )
    for deck_index, record in enumerate(train_records):
        opponent = arena.bind_deck(
            search_agent.agent,
            record["deck"],
            search_module=search_agent,
        )
        for game in range(8):
            recorder.reset()
            candidate_seat = game % 2
            if candidate_seat == 0:
                outcome = arena.play_game(make, candidate, opponent)
                candidate_failure = outcome.seat0_fail
            else:
                outcome = arena.play_game(make, opponent, candidate)
                candidate_failure = outcome.seat1_fail
            cluster = recorder.cluster or "unobserved"
            row = cluster_stats[cluster][policy_name]
            row["games"] += 1
            row["wins"] += int(outcome.winner == candidate_seat)
            row["draws"] += int(outcome.winner is None)
            training_games.append(
                {
                    "policy": policy_name,
                    "deck": record["signature"],
                    "game": game,
                    "cluster": cluster,
                    "candidate_seat": candidate_seat,
                    "candidate_win": outcome.winner == candidate_seat,
                    "candidate_failure": candidate_failure,
                    "game_error": outcome.game_error,
                    "seconds": outcome.seconds,
                }
            )

book = {}
book_evidence = {}
for cluster, rows in cluster_stats.items():
    eligible = {
        name: row for name, row in rows.items() if row["games"] >= 4
    }
    if not eligible or cluster == "unobserved":
        continue
    selected = max(
        eligible,
        key=lambda name: (
            (eligible[name]["wins"] + 1) / (eligible[name]["games"] + 2),
            eligible[name]["games"],
            name == "chance_coupled",
        ),
    )
    book[cluster] = selected
    book_evidence[cluster] = {
        "selected": selected,
        "policies": dict(rows),
    }

class OpeningBookAgent:
    def __init__(self):
        self.selected = None
        self.matched = False
        self.games = 0
        self.covered_games = 0

    def agent(self, obs_dict, deck):
        if obs_dict.get("select") is None:
            self.selected = None
            self.matched = False
            self.games += 1
            return list(deck)
        if self.selected is None:
            try:
                obs = to_observation_class(obs_dict)
                if (
                    obs.select is not None
                    and obs.select.context == SelectContext.MAIN
                    and int(getattr(obs.state, "turn", 0) or 0) <= 3
                ):
                    cluster = opening_cluster(obs_dict)
                    self.selected = book.get(cluster, "chance_coupled")
                    self.matched = cluster in book
                    if self.matched:
                        self.covered_games += 1
            except Exception:
                self.selected = "chance_coupled"
        policy_name = self.selected or "chance_coupled"
        return policies[policy_name](obs_dict, deck)

def screen(name, policy_fn):
    candidate = arena.bind_deck_arg(
        policy_fn, archaludon_policy.ARCHALUDON_DECK
    )
    rows = []
    for record in test_records:
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
        "mean_pairing_s": statistics.mean(
            row["mean_match_s"] for row in rows
        ),
        "worst_pairing_p95_s": max(row["p95_match_s"] for row in rows),
        "pairings": rows,
    }

book_agent = OpeningBookAgent()
book_result = screen("opening-book", book_agent.agent)
control_result = screen("chance-control", policies["chance_coupled"])
coverage = book_agent.covered_games / max(book_agent.games, 1)
ready_for_confirmation = bool(
    coverage >= 0.30
    and book_result["winrate"] > control_result["winrate"]
    and book_result["candidate_failures"] == 0
)
report = {
    "experiment": "EXP-85",
    "strategy_rank": 56,
    "strategy": "State cluster opening book",
    "training_decks": len(train_records),
    "training_games": len(training_games),
    "heldout_decks": len(test_records),
    "book_clusters": len(book),
    "heldout_coverage": coverage,
    "book_result": book_result,
    "chance_control_result": control_result,
    "winrate_change": book_result["winrate"] - control_result["winrate"],
    "ready_for_fresh_confirmation": ready_for_confirmation,
    "promotion_ready": False,
    "interpretation": (
        "The book uses only our visible opening states and game outcomes. "
        "This exploratory screen cannot promote without a fresh confirmation."
    ),
    "public_policy_code_used": False,
    "public_policy_actions_used_as_labels": False,
    "submission_performed": False,
}
Path("/kaggle/working/opening_book.json").write_text(
    json.dumps({"book": book, "evidence": book_evidence}, indent=2),
    encoding="utf-8",
)
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
                        "# EXP-85 State cluster opening book\n\n"
                        "Opening clusters are learned from our own visible "
                        "states and game outcomes on eight decks, then tested "
                        "against chance-coupled control on twelve disjoint decks.",
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
