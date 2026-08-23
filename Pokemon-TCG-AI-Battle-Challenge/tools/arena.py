"""
Local "gauntlet arena" for the Pokémon TCG AI Battle project (Phase 0 of PLAN.md
— repair/extend evaluation).

A round-robin tournament between a set of *competitors*, where

    competitor = (name, agent_fn, deck)

An agent plays a given deck by *returning that deck* during the deck-selection
phase (when ``obs["select"] is None``). The helper ``bind_deck`` wraps any base
agent so it returns the assigned deck at deck-selection and otherwise delegates
to the base agent — and, for our ``search_agent``, also points
``search_agent.DECK`` at that deck so its internal determinization is consistent.

Every game is wrapped in try/except: an agent that raises, a game that errors,
or an agent the engine marks ERROR/INVALID/TIMEOUT is *recorded* and counted, and
never crashes the whole arena. The loser of such a game is the failing agent.

Outputs (map to PLAN.md Phase 0 acceptance criteria):
  * per-pairing W/L/D with a 95% Wilson confidence interval,
  * first-player vs second-player win-rate,
  * mean and p95 per-match wall-clock time,
  * counts of crashes / exceptions / timeouts / invalid-actions,
  * a readable row-vs-col matchup matrix + an overall rating per competitor
    (simple mean win-rate across opponents),
  * a machine-readable ``arena_results.csv`` (+ ``.json``) with all columns.

Usage
-----
    KAGGLE_CONFIG_DIR=C:/Users/chand/.kaggle \
      .venv/Scripts/python.exe tools/arena.py --games 20

    # a subset of competitors
    ... tools/arena.py --games 40 --competitors v3-abomasnow,v3-lucario

    # list the registered competitors and exit
    ... tools/arena.py --list

Add more competitors in the clearly-marked COMPETITORS block near the top.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback

# ---------------------------------------------------------------------------
# Path setup — make agent/ and references/top_rankers/ importable regardless of
# where this script is launched from.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "agent"), os.path.join(_ROOT, "references", "top_rankers")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The gauntlet decks (each a flat list of exactly 60 card IDs).
from decks import MEGA_ABOMASNOW, MEGA_LUCARIO, DRAGAPULT, ALAKAZAM  # noqa: E402

# Aug-2026 live-meta decks mined from the official top-episode replay dump.
import meta_decks  # noqa: E402

# The engine's built-in agents + default deck.
import kaggle_environments.envs.cabt.cabt as cabt  # noqa: E402

# Our v3 search agent module (imported as a module so bind_deck can set DECK).
import search_agent  # noqa: E402

# Our own domain-knowledge policy + the hybrid (domain policy driving search).
# Both take (obs, deck) directly -- see bind_deck_arg below.
import domain_policy  # noqa: E402
import hybrid_agent  # noqa: E402
import candidate_agents  # noqa: E402
import archaludon_policy  # noqa: E402
import grimmsnarl_policy  # noqa: E402
import grimmsnarl_variants  # noqa: E402
import elite_grim_override  # noqa: E402
import search_scaled  # noqa: E402
import search_belief  # noqa: E402

try:  # Public references are optional and never required by our candidate path.
    import public_agents  # noqa: E402
except Exception:
    public_agents = None


# ---------------------------------------------------------------------------
# bind_deck — wrap a base agent so it plays a *specific* deck.
# ---------------------------------------------------------------------------
def bind_deck(agent_fn, deck, search_module=None):
    """Return a new agent that returns ``deck`` at deck-selection and otherwise
    delegates to ``agent_fn``.

    If ``search_module`` is given (our ``search_agent``), its module-level
    ``DECK`` is set to ``deck`` before every delegated call, so the agent's
    internal determinization draws from the deck it is actually piloting.
    """
    frozen = list(deck)

    def wrapped(obs):
        if search_module is not None:
            # Keep the search agent's embedded deck in sync with what it pilots.
            search_module.DECK = frozen
        if obs.get("select") is None:  # deck-selection phase
            return list(frozen)
        return agent_fn(obs)

    return wrapped


def bind_deck_arg(agent_fn2, deck):
    """Wrap an agent whose signature is fn(obs, deck) -> list[int] (our
    domain_policy.domain_agent / hybrid_agent.hybrid_agent) into the plain
    fn(obs) -> list[int] shape the arena expects."""
    frozen = list(deck)

    def wrapped(obs):
        return agent_fn2(obs, frozen)

    return wrapped


# ===========================================================================
# COMPETITORS — edit this block to add/remove entrants.
# Each entry: (name, base_agent_fn, deck, kind)
#   * name           : unique short label used on the CLI and in the matrix.
#   * base_agent_fn  : callable obs -> list[int] (the policy).
#   * deck           : 60-card ID list the competitor pilots.
#   * kind           : "search" (search_agent, needs .DECK synced),
#                       "arg" (domain_policy/hybrid_agent, takes (obs, deck)),
#                       or "plain" (builtin agents: random/first).
# ===========================================================================
COMPETITORS = [
    ("random",        cabt.random_agent,      MEGA_ABOMASNOW, "plain"),
    ("first",         cabt.first_agent,       MEGA_ABOMASNOW, "plain"),
    ("v3-abomasnow",  search_agent.agent,     MEGA_ABOMASNOW, "search"),
    ("v3-lucario",    search_agent.agent,     MEGA_LUCARIO,   "search"),
    ("v3-dragapult",  search_agent.agent,     DRAGAPULT,      "search"),
    ("v3-alakazam",   search_agent.agent,     ALAKAZAM,       "search"),
    ("domain-lucario", domain_policy.domain_agent, MEGA_LUCARIO, "arg"),
    ("hybrid-lucario", hybrid_agent.hybrid_agent,  MEGA_LUCARIO, "arg"),
    ("hybrid-active-threat", candidate_agents.hybrid_active_threat, MEGA_LUCARIO, "arg"),
    ("hybrid-no-retreat", candidate_agents.hybrid_no_retreat, MEGA_LUCARIO, "arg"),
    ("ours-archaludon", archaludon_policy.archaludon_agent, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("ours-arch-sequenced", archaludon_policy.archaludon_sequenced_agent, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("ours-arch-boss", archaludon_policy.archaludon_boss_gated_agent, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("ours-arch-seq-boss", archaludon_policy.archaludon_seq_boss_agent, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("ours-arch-search", candidate_agents.archaludon_search, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-search-k3d6", candidate_agents.arch_search_k3_d6, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-search-k6d3", candidate_agents.arch_search_k6_d3, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-search-k9d2", candidate_agents.arch_search_k9_d2, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-search-k6d3-correct", candidate_agents.arch_search_k6_d3_correct, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-search-k9d2-correct", candidate_agents.arch_search_k9_d2_correct, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-belief-1x48", candidate_agents.arch_belief_d1_r48, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-belief-3x16", candidate_agents.arch_belief_d3_r16, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-belief-6x8", candidate_agents.arch_belief_d6_r8, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-belief-12x4", candidate_agents.arch_belief_d12_r4, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-beam-2", candidate_agents.arch_beam_2, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-beam-4", candidate_agents.arch_beam_4, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-beam-8", candidate_agents.arch_beam_8, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-beam-2-response", candidate_agents.arch_beam_2_response, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-flat-root", candidate_agents.arch_flat_root, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-ismcts", candidate_agents.arch_ismcts, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-gated-ismcts", candidate_agents.arch_gated_ismcts, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("ours-arch-adaptive", candidate_agents.arch_adaptive, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-rl16-local", candidate_agents.arch_rl_16, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-rl24-local", candidate_agents.arch_rl_24, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-rl32-local", candidate_agents.arch_rl_32, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-rl48-local", candidate_agents.arch_rl_48, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("rl-exp18-8", candidate_agents.rl_exp18_8, MEGA_ABOMASNOW, "arg"),
    ("arch-context-015", candidate_agents.arch_context_015, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-context-035", candidate_agents.arch_context_035, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("arch-context-070", candidate_agents.arch_context_070, archaludon_policy.ARCHALUDON_DECK, "arg"),
    ("ours-arch-learned", candidate_agents.arch_learned_adaptive, archaludon_policy.ARCHALUDON_DECK, "arg"),
    # --- Aug-2026 live-meta field: our deck-agnostic domain policy piloting the
    # actual decks the current leaderboard plays. These are the opponents that
    # matter now; the Archaludon/Abomasnow entries above are historical.
    ("meta-grimmsnarl", domain_policy.domain_agent, meta_decks.GRIMMSNARL, "arg"),
    ("meta-dragapult", domain_policy.domain_agent, meta_decks.DRAGAPULT, "arg"),
    ("meta-alakazam", domain_policy.domain_agent, meta_decks.ALAKAZAM, "arg"),
    ("meta-crustle", domain_policy.domain_agent, meta_decks.CRUSTLE, "arg"),
    ("meta-garchomp", domain_policy.domain_agent, meta_decks.GARCHOMP, "arg"),
    ("meta-dudunsparce", domain_policy.domain_agent, meta_decks.DUDUNSPARCE, "arg"),
    ("meta-slowking", domain_policy.domain_agent, meta_decks.SLOWKING, "arg"),
    ("meta-lucario", domain_policy.domain_agent, meta_decks.LUCARIO, "arg"),
    ("v3-grimmsnarl", search_agent.agent, meta_decks.GRIMMSNARL, "search"),
    # Our candidate: original Grimmsnarl-aware policy on the mined meta list.
    ("ours-grimmsnarl", grimmsnarl_policy.grimmsnarl_agent, meta_decks.GRIMMSNARL, "arg"),
    ("elite-grim-gated", elite_grim_override.make_agent(meta_decks.GRIMMSNARL), meta_decks.GRIMMSNARL, "plain"),
    ("elite-grim-strict", elite_grim_override.make_agent(meta_decks.GRIMMSNARL, 3.0), meta_decks.GRIMMSNARL, "plain"),
    ("elite-grim-ultra", elite_grim_override.make_agent(meta_decks.GRIMMSNARL, 4.5), meta_decks.GRIMMSNARL, "plain"),
    # Ablations: each drops exactly one deck-specific override.
    ("abl-no-snipe", grimmsnarl_variants.no_snipe_agent, meta_decks.GRIMMSNARL, "arg"),
    ("abl-no-energy", grimmsnarl_variants.no_energy_agent, meta_decks.GRIMMSNARL, "arg"),
    ("abl-no-trainer", grimmsnarl_variants.no_trainer_agent, meta_decks.GRIMMSNARL, "arg"),
    ("abl-no-trainer-only", grimmsnarl_variants.no_trainer_only_agent, meta_decks.GRIMMSNARL, "arg"),
    ("abl-no-setup-lead", grimmsnarl_variants.no_setup_lead_agent, meta_decks.GRIMMSNARL, "arg"),
    ("energy-only", grimmsnarl_variants.energy_only_agent, meta_decks.GRIMMSNARL, "arg"),
    # Strong local baselines on other meta decks, for deck-vs-policy separation.
    ("v3-garchomp", search_agent.agent, meta_decks.GARCHOMP, "search"),
    ("v3-dragapult-meta", search_agent.agent, meta_decks.DRAGAPULT, "search"),
    # Time-budgeted search: same algorithm, many more determinizations.
    ("scaled-garchomp", search_scaled.make_agent(meta_decks.GARCHOMP), meta_decks.GARCHOMP, "plain"),
    ("scaled-grimmsnarl", search_scaled.make_agent(meta_decks.GRIMMSNARL), meta_decks.GRIMMSNARL, "plain"),
    # Same search, same budget -- only the opponent model changes.
    ("belief-garchomp", search_belief.make_agent(meta_decks.GARCHOMP), meta_decks.GARCHOMP, "plain"),
    ("belief-grimmsnarl", search_belief.make_agent(meta_decks.GRIMMSNARL), meta_decks.GRIMMSNARL, "plain"),
]
if public_agents is not None:
    COMPETITORS += [
        ("public-naoto-1027", public_agents.NAOTO_1027_AGENT,
         public_agents.NAOTO_1027_DECK, "plain"),
        ("public-arch-1030", public_agents.ARCH_1030_AGENT,
         public_agents.ARCH_1030_DECK, "plain"),
        ("public-garchomp-v28", public_agents.GARCHOMP_V28_AGENT,
         public_agents.GARCHOMP_V28_DECK, "plain"),
    ]
# v3 search on every mined meta deck: policy held constant, deck is the variable.
COMPETITORS += [
    (f"sweep-{name}", search_agent.agent, deck, "search")
    for name, deck in sorted(meta_decks.META_DECKS.items())
]
# Domain-policy-driven search (hybrid v5) on the current meta decks.
COMPETITORS += [
    ("hyb-garchomp", hybrid_agent.hybrid_agent, meta_decks.GARCHOMP, "arg"),
    ("hyb-other", hybrid_agent.hybrid_agent, meta_decks.OTHER, "arg"),
    ("hyb-grimmsnarl", hybrid_agent.hybrid_agent, meta_decks.GRIMMSNARL, "arg"),
]
# Baseline: always take the engine's first-listed option. Public analysis
# (discussion 713608) reports the engine enumerates options best->worst, making
# this a strong and hard-to-beat local optimum.
COMPETITORS += [
    ("first-garchomp", cabt.first_agent, meta_decks.GARCHOMP, "plain"),
    ("first-other", cabt.first_agent, meta_decks.OTHER, "plain"),
]
# Card-identity ranker trained on official replay data (host-sanctioned).
import ranker_agent  # noqa: E402
COMPETITORS += [
    ("rank-prior-garchomp", ranker_agent.make_agent(meta_decks.GARCHOMP, "prior"), meta_decks.GARCHOMP, "plain"),
    ("rank-policy-garchomp", ranker_agent.make_agent(meta_decks.GARCHOMP, "policy"), meta_decks.GARCHOMP, "plain"),
]
# Hybrid on every mined meta deck: the vs-public benchmark showed hybrid > v3
# against strong opponents, so deck choice must be re-measured under hybrid.
COMPETITORS += [
    (f"hybs-{name}", hybrid_agent.hybrid_agent, deck, "arg")
    for name, deck in sorted(meta_decks.META_DECKS.items())
]
# Exact public Starmie/Cinderace/Budew list (the notebook discloses the list and
# strategy but not its private skill modules).  These three policy controls test
# whether our original domain/search stack can exploit the archetype before we
# invest in a dedicated implementation or replay-distilled policy.
_STARMIE_TEMPO_DECK = (
    [1030] * 4 + [1031] * 3 + [666] * 4 + [235]
    + [1086] * 4 + [1145] * 4 + [1122] * 4 + [1121]
    + [1120] * 4 + [1097] * 2 + [1189] * 4 + [1182] * 2
    + [1229] * 3 + [1225] + [1227] * 4 + [1129] + [1159]
    + [3] * 9 + [17] * 4
)
COMPETITORS += [
    ("starmie-domain", domain_policy.domain_agent, _STARMIE_TEMPO_DECK, "arg"),
    ("starmie-hybrid", hybrid_agent.hybrid_agent, _STARMIE_TEMPO_DECK, "arg"),
    ("starmie-search", search_agent.agent, _STARMIE_TEMPO_DECK, "search"),
]
# Matchup-gated router between hybrid and v3 (complementary matchup profiles).
import gated_agent  # noqa: E402
COMPETITORS += [
    ("gated-garchomp", gated_agent.gated_agent, meta_decks.GARCHOMP, "arg"),
]
# Exact 60-card lists mined from the highest-scoring leaderboard teams.
import top_decks  # noqa: E402
COMPETITORS += [
    (f"td-{name}", hybrid_agent.hybrid_agent, deck, "arg")
    for name, deck in sorted(top_decks.TOP_DECKS.items())
]
# Hybrid search with the learned board-value critic as its leaf evaluator.
import critic_policy  # noqa: E402
import tuned_policy  # noqa: E402
def _tuned_hybrid(obs, deck):
    return hybrid_agent.hybrid_agent(obs, deck, policy_module=tuned_policy)
def _critic_hybrid(obs, deck):
    return hybrid_agent.hybrid_agent(obs, deck, policy_module=critic_policy)
COMPETITORS += [
    ("critic-ogerpon", _critic_hybrid, meta_decks.OTHER, "arg"),
    ("critic-garchomp", _critic_hybrid, meta_decks.GARCHOMP, "arg"),
    # SPSA-tuned priority weights (policy_w.json), hybrid search, live deck.
    ("tuned-ogerpon", _tuned_hybrid, meta_decks.OTHER, "arg"),
    ("tuned-garchomp", _tuned_hybrid, meta_decks.GARCHOMP, "arg"),
]
# Card-specific Ogerpon weights feeding hybrid search (the LB-950 mechanism).
import ogerpon_policy  # noqa: E402
def _oger_hybrid(obs, deck):
    return hybrid_agent.hybrid_agent(obs, deck, policy_module=ogerpon_policy)
COMPETITORS += [
    ("oger-cards", _oger_hybrid, meta_decks.OTHER, "arg"),
    ("oger-flat", ogerpon_policy.ogerpon_agent, meta_decks.OTHER, "arg"),
]
# Correct damage for scaling attacks (Ogerpon reads 270 not 30; Alakazam's
# Powerful Hand reads real damage instead of 0).
import scaled_policy  # noqa: E402
def _scaled_hybrid(obs, deck):
    return hybrid_agent.hybrid_agent(obs, deck, policy_module=scaled_policy)
COMPETITORS += [
    ("scaled-ogerpon", _scaled_hybrid, meta_decks.OTHER, "arg"),
    ("scaled-garchomp", _scaled_hybrid, meta_decks.GARCHOMP, "arg"),
]
# Defence-only variant: correct incoming damage, keep the static offence estimate.
import scaled_defence  # noqa: E402
def _scdef_hybrid(obs, deck):
    return hybrid_agent.hybrid_agent(obs, deck, policy_module=scaled_defence)
COMPETITORS += [
    ("scdef-ogerpon", _scdef_hybrid, meta_decks.OTHER, "arg"),
]
# 2-ply minimax: score our move by the opponent's best reply (the technique the
# LB-950 agent uses and our 1-ply search lacks).
import minimax_agent, ogerpon_policy as _ogp  # noqa: E402
COMPETITORS += [
    ("mm-ogerpon", minimax_agent.minimax_agent, meta_decks.OTHER, "arg"),
    ("mm-garchomp", minimax_agent.minimax_agent, meta_decks.GARCHOMP, "arg"),
]
# Track B: the public baseline we submitted, loaded as a local competitor so its
# external weight table can be re-tuned with our CRN/SPRT infrastructure.
try:
    import importlib.util as _ilu, os as _os
    _fp = _os.path.join(_ROOT, "agent", "fork", "fork_main.py")
    _spec = _ilu.spec_from_file_location("fork_main", _fp)
    _fm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_fm)
    _FORK_DECK = [int(x) for x in open(_os.path.join(_ROOT, "agent", "fork", "deck.csv")) if x.strip()]
    COMPETITORS += [("fork-v22", _fm.agent, _FORK_DECK, "plain")]
except Exception as _e:
    print("fork competitor unavailable:", _e)
try:
    _rfp = _os.path.join(_ROOT, "agent", "router_v12", "main.py")
    _rspec = _ilu.spec_from_file_location("router_v12_main", _rfp)
    _rm = _ilu.module_from_spec(_rspec); _rspec.loader.exec_module(_rm)
    _ROUTER_V12_DECK = [int(x) for x in open(
        _os.path.join(_ROOT, "agent", "router_v12", "deck.csv")) if x.strip()]
    COMPETITORS += [("router-v12", _rm.agent, _ROUTER_V12_DECK, "plain")]
    import v12_neural_override as _v12nn
    COMPETITORS += [("v12-neural", _v12nn.agent, _ROUTER_V12_DECK, "plain")]
    import v12_pairwise_override as _v12pair
    COMPETITORS += [("v12-pairwise", _v12pair.agent, _ROUTER_V12_DECK, "plain")]
except Exception as _e:
    print("router v12 competitor unavailable:", _e)
try:
    _gfp = _os.path.join(_ROOT, "agent", "grim_candy_v2", "main.py")
    # Grim Candy is a multi-file submission whose entry point uses Kaggle-style
    # sibling imports (``import policy_features``).  Loading only main.py via a
    # file spec does not add its directory to sys.path, especially in spawned
    # counterfactual workers, so make the packaged directory importable first.
    _gdir = _os.path.dirname(_gfp)
    if _gdir not in sys.path:
        sys.path.insert(0, _gdir)
    _gspec = _ilu.spec_from_file_location("grim_candy_v2_main", _gfp)
    _gm = _ilu.module_from_spec(_gspec); _gspec.loader.exec_module(_gm)
    _GRIM_CANDY_V2_DECK = [int(x) for x in open(
        _os.path.join(_ROOT, "agent", "grim_candy_v2", "deck.csv")) if x.strip()]
    COMPETITORS += [("grim-candy-v2", _gm.agent, _GRIM_CANDY_V2_DECK, "plain")]
    def _grim_damage_route_v1(obs):
        """One causal rule: prefer benched Grim ex for DAMAGE_COUNTER.

        Five independent counterfactual episodes (including held-out seeds)
        favored this exact active->bench substitution.  Every other decision is
        delegated byte-for-byte to frozen Grim Candy v2.
        """
        baseline = _gm.agent(obs)
        try:
            sel = obs.get("select") or {}
            if int(sel.get("context", -1)) != 13 or len(baseline) != 1:
                return baseline
            options = sel.get("option") or []
            base_i = int(baseline[0])
            if not (0 <= base_i < len(options)):
                return baseline
            cur = obs.get("current") or {}
            me = int(cur.get("yourIndex", 0) or 0)
            players = cur.get("players") or []

            def target(option):
                area = int(option.get("area", 0) or 0)
                index = int(option.get("index", -1) if option.get("index") is not None else -1)
                pidx = int(option.get("playerIndex", me) if option.get("playerIndex") is not None else me)
                if not (0 <= pidx < len(players)):
                    return 0, area
                zone = players[pidx].get("active" if area == 4 else "bench") or []
                card = zone[index] if area in (4, 5) and 0 <= index < len(zone) else None
                return int((card or {}).get("id", 0) or 0), area

            if target(options[base_i]) != (648, 4):
                return baseline
            for i, option in enumerate(options):
                if target(option) == (648, 5):
                    return [i]
        except Exception:
            pass
        return baseline

    COMPETITORS += [("grim-damage-route-v1", _grim_damage_route_v1,
                     _GRIM_CANDY_V2_DECK, "plain")]
    import grim_search_hybrid as _gsh
    COMPETITORS += [
        ("grim-search-m0", _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=0.0),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-m1k", _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=1000.0),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-m5k", _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=5000.0),
         _GRIM_CANDY_V2_DECK, "plain"),
    ]
    _grim_search_m5k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=5000.0)
    _grim_search_m10k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=10000.0)
    _grim_search_m20k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=20000.0)
    _grim_search_m30k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=30000.0)
    _grim_search_m40k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=40000.0)
    _grim_search_m50k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=50000.0)
    _grim_search_m75k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK, margin=75000.0)
    _grim_reply_m5k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK,
                                      margin=5000.0, opponent_roll=20)
    _grim_reply_m10k = _gsh.make_agent(_gm.agent, _GRIM_CANDY_V2_DECK,
                                       margin=10000.0, opponent_roll=20)
    _garchomp_ids = {341, 342, 379, 380, 381, 387}
    _archaludon_ids = {57, 169, 190, 666}
    _grim_garch_d4 = _gsh.make_agent(
        _gm.agent, _GRIM_CANDY_V2_DECK, margin=10000.0, determinizations=4)
    _grim_garch_d6 = _gsh.make_agent(
        _gm.agent, _GRIM_CANDY_V2_DECK, margin=10000.0, determinizations=6)
    _grim_garch_k3d4 = _gsh.make_agent(
        _gm.agent, _GRIM_CANDY_V2_DECK, margin=10000.0,
        top_k=3, determinizations=4)
    _grim_garch_roll40 = _gsh.make_agent(
        _gm.agent, _GRIM_CANDY_V2_DECK, margin=10000.0, max_roll=40)
    _grim_garch_m7500 = _gsh.make_agent(
        _gm.agent, _GRIM_CANDY_V2_DECK, margin=7500.0)
    _grim_garch_m12500 = _gsh.make_agent(
        _gm.agent, _GRIM_CANDY_V2_DECK, margin=12500.0)
    # Strictly conservative composite: outside a visible, recognized family it
    # calls frozen Grim directly.  Each family threshold was selected on
    # discovery seeds and then validated on fresh seeds and independent policy
    # implementations.
    _grim_search_arch_gated = _gsh.visible_family_gate(
        _grim_search_m20k, _gm.agent, _archaludon_ids)
    _grim_search_family_v1 = _gsh.visible_family_gate(
        _grim_search_m10k, _grim_search_arch_gated, _garchomp_ids)
    _router_crustle_ids = {344, 345, 607}
    _grim_search_family_v2 = _gsh.visible_family_gate(
        _grim_search_m20k, _grim_search_family_v1, _router_crustle_ids)
    def _phase_family(garch_late, arch_late, router_late, late_prizes=2):
        garch = _gsh.make_agent(
            _gm.agent, _GRIM_CANDY_V2_DECK, margin=10000.0,
            margin_late=garch_late, late_prizes=late_prizes)
        arch = _gsh.make_agent(
            _gm.agent, _GRIM_CANDY_V2_DECK, margin=20000.0,
            margin_late=arch_late, late_prizes=late_prizes)
        router = _gsh.make_agent(
            _gm.agent, _GRIM_CANDY_V2_DECK, margin=20000.0,
            margin_late=router_late, late_prizes=late_prizes)
        arch_gate = _gsh.visible_family_gate(arch, _gm.agent, _archaludon_ids)
        family_v1 = _gsh.visible_family_gate(garch, arch_gate, _garchomp_ids)
        return _gsh.visible_family_gate(router, family_v1, _router_crustle_ids)
    _grim_phase_late_off = _phase_family(1e12, 1e12, 1e12)
    _grim_phase_late_strict = _phase_family(20000.0, 40000.0, 40000.0)
    _grim_phase_last_off = _phase_family(1e12, 1e12, 1e12, late_prizes=1)
    COMPETITORS += [
        # Ungated variants are discovery probes.  A selected margin is only
        # promoted behind a visible opponent-family gate after held-out tests.
        ("grim-search-m10k", _grim_search_m10k,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-m20k", _grim_search_m20k,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-m30k", _grim_search_m30k,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-m40k", _grim_search_m40k,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-m50k", _grim_search_m50k,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-m75k", _grim_search_m75k,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-garchomp-v1",
         _gsh.visible_family_gate(_grim_search_m5k, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-garchomp-m10k",
         _gsh.visible_family_gate(_grim_search_m10k, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-garchomp-m20k",
         _gsh.visible_family_gate(_grim_search_m20k, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-arch-m20k", _grim_search_arch_gated,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-family-v1", _grim_search_family_v1,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-search-family-v2", _grim_search_family_v2,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-phase-late-off", _grim_phase_late_off,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-phase-late-strict", _grim_phase_late_strict,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-phase-last-off", _grim_phase_last_off,
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-garch-d4", _gsh.visible_family_gate(
            _grim_garch_d4, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-garch-d6", _gsh.visible_family_gate(
            _grim_garch_d6, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-garch-k3d4", _gsh.visible_family_gate(
            _grim_garch_k3d4, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-garch-roll40", _gsh.visible_family_gate(
            _grim_garch_roll40, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-garch-m7500", _gsh.visible_family_gate(
            _grim_garch_m7500, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-garch-m12500", _gsh.visible_family_gate(
            _grim_garch_m12500, _gm.agent, _garchomp_ids),
         _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-reply-m5k", _grim_reply_m5k, _GRIM_CANDY_V2_DECK, "plain"),
        ("grim-reply-m10k", _grim_reply_m10k, _GRIM_CANDY_V2_DECK, "plain"),
    ]
except Exception as _e:
    print("grim candy v2 competitor unavailable:", _e)

# Public high-performing notebook candidates pulled for a controlled battery.
# They are multi-file Kaggle submissions with sibling imports, so load each
# while its own directory is first on sys.path.  Purging sibling module names
# prevents a previous submission's generic modules (for example ``policy``)
# from being silently reused by the next candidate.
def _load_packaged_candidate(short_name):
    import importlib.util as ilu
    from pathlib import Path

    directory = Path(_ROOT) / "agent" / "public_candidates" / short_name
    sibling_names = {path.stem for path in directory.glob("*.py")}
    sibling_names.update(path.name for path in directory.iterdir()
                         if path.is_dir() and (path / "__init__.py").is_file())
    for module_name in sibling_names:
        sys.modules.pop(module_name, None)
    old_cwd = _os.getcwd()
    sys.path.insert(0, str(directory))
    try:
        _os.chdir(directory)
        spec = ilu.spec_from_file_location(f"public_candidate_{short_name}", directory / "main.py")
        module = ilu.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        deck = [int(value) for value in (directory / "deck.csv").read_text().split()]
        if len(deck) != 60:
            raise ValueError(f"{short_name}: expected 60 cards, found {len(deck)}")
        entrypoint = getattr(module, "agent", None)
        if entrypoint is None:
            entrypoint = getattr(module, "mega_lopunny_cleanroom_entrypoint", None)
        if not callable(entrypoint):
            raise AttributeError(f"{short_name}: no callable agent entry point")
        return entrypoint, deck
    finally:
        _os.chdir(old_cwd)
        sys.path.remove(str(directory))


_loaded_public_candidates = {}
for _short, _label in (
    ("battlecore", "public-battlecore"),
    ("grimmsnarl_control", "public-grim-control"),
    ("mega_lucario", "public-mega-lucario"),
    ("probability_v2", "public-probability-v2"),
    ("meta_a", "public-meta-a"),
    ("a-better-hand-alakazam-rising-tide-v21", "current-better-hand-v21"),
    ("codex-sol-eclipse-alakazam", "current-sol-eclipse"),
    ("ptcg-ai-battle-static-deck-tusk-1208-v24", "current-great-tusk"),
    ("ptcg-ai-battle-visible-grim-belief-alakazam-v21", "current-visible-grim-v21"),
    ("current-crustle-counter-v29", "current-crustle-counter-v29"),
    ("current-fixed-metal-v15", "current-fixed-metal-v15"),
    ("current-visible-field-router-v3", "current-visible-field-router-v3"),
    ("current-souta-1208", "current-souta-1208"),
    ("current-public-915", "current-public-915"),
    ("current-elo-1050", "current-elo-1050"),
    ("current-advanced-alakazam", "current-advanced-alakazam"),
    ("current-mega-lucario-v62", "current-mega-lucario-v62"),
    ("current-frostwall", "current-frostwall"),
    ("current-naoto-froslass", "current-naoto-froslass"),
    ("current-baseline-1084", "current-baseline-1084"),
):
    try:
        _candidate_agent, _candidate_deck = _load_packaged_candidate(_short)
        _loaded_public_candidates[_short] = (_candidate_agent, _candidate_deck)
        COMPETITORS.append((_label, _candidate_agent, _candidate_deck, "plain"))
    except Exception as _e:
        print(f"{_label} unavailable:", _e)

# Two-by-two decomposition of the strongest public candidate and Grim v2.
# Their lists differ by exactly one slot (Rare Candy vs Tool Scrapper), so these
# crosses distinguish a deck improvement from a policy improvement.
if "grimmsnarl_control" in _loaded_public_candidates and "_gm" in globals():
    _control_agent, _control_deck = _loaded_public_candidates["grimmsnarl_control"]
    COMPETITORS.extend([
        ("grim-v2-scrapper", _gm.agent, _control_deck, "plain"),
        ("grim-control-candy", _control_agent, _GRIM_CANDY_V2_DECK, "plain"),
    ])

# Deck/policy decomposition for the forward-search Alakazam candidate.  This
# identifies whether its positive Alakazam/Archaludon lanes come from the list
# or the action policy before any hybridization is attempted.
if "current-advanced-alakazam" in _loaded_public_candidates and "_gm" in globals():
    _advanced_agent, _advanced_deck = _loaded_public_candidates["current-advanced-alakazam"]
    COMPETITORS.extend([
        ("grim-policy-advanced-deck", _gm.agent, _advanced_deck, "plain"),
        ("advanced-policy-grim-deck", _advanced_agent, _GRIM_CANDY_V2_DECK, "plain"),
    ])
# Our policy piloting the PUBLIC agents' own deck lists (deck lists are permitted;
# no policy code is used). Never tested before.
if public_agents is not None:
    COMPETITORS += [
        ("hyb-romanB", hybrid_agent.hybrid_agent, public_agents.ROMAN_B_DECK, "arg"),
        ("hyb-romanA", hybrid_agent.hybrid_agent, public_agents.ROMAN_A_DECK, "arg"),
        ("scdef-romanB", _scdef_hybrid, public_agents.ROMAN_B_DECK, "arg"),
    ]
if public_agents is not None:
    COMPETITORS.extend(
        [
            ("public-archaludon", public_agents.ROMAN_A_AGENT, public_agents.ROMAN_A_DECK, "plain"),
            ("public-alakazam", public_agents.ROMAN_B_AGENT, public_agents.ROMAN_B_DECK, "plain"),
        ]
    )

# Human-readable deck names for reporting (identity-based lookup).
_DECK_NAMES = [
    ("MEGA_ABOMASNOW", MEGA_ABOMASNOW),
    ("MEGA_LUCARIO", MEGA_LUCARIO),
    ("DRAGAPULT", DRAGAPULT),
    ("ALAKAZAM", ALAKAZAM),
    ("OUR_ARCHALUDON", archaludon_policy.ARCHALUDON_DECK),
]
if public_agents is not None:
    _DECK_NAMES.extend(
        [
            ("PUBLIC_ARCHALUDON", public_agents.ROMAN_A_DECK),
            ("PUBLIC_ALAKAZAM", public_agents.ROMAN_B_DECK),
        ]
    )


def deck_name(deck) -> str:
    for name, d in _DECK_NAMES:
        if d is deck:
            return name
    return "custom"


# ---------------------------------------------------------------------------
# Statistics helpers.
# ---------------------------------------------------------------------------
def wilson_interval(wins: int, n: int, z: float = 1.96):
    """95% Wilson score confidence interval for a win-rate."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def percentile(values, q: float) -> float:
    """Nearest-rank percentile (q in [0,1]). Empty -> 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = int(math.ceil(q * len(s))) - 1
    idx = min(max(idx, 0), len(s) - 1)
    return s[idx]


# ---------------------------------------------------------------------------
# Deck legality (via battle_start) — checked once at startup.
# ---------------------------------------------------------------------------
_ERR_DESC = {1: "bad card id", 2: ">4 cards same name", 3: "no basic Pokemon", 4: ">1 ace spec"}


def check_deck_legal(deck):
    """Return (legal: bool, reason: str). Uses the engine's battle_start."""
    try:
        from kaggle_environments.envs.cabt.cg.game import battle_start, battle_finish
    except Exception as e:  # engine not available -> assume legal, note why
        return True, f"legality check unavailable ({e})"
    try:
        _, sd = battle_start(list(deck), list(deck))
        try:
            battle_finish()
        except Exception:
            pass
        if getattr(sd, "errorPlayer", -1) >= 0:
            et = getattr(sd, "errorType", 0)
            return False, _ERR_DESC.get(et, f"errorType {et}")
        return True, "ok"
    except Exception as e:
        return False, f"battle_start raised: {e}"


# ---------------------------------------------------------------------------
# Single game.
# ---------------------------------------------------------------------------
_FAIL_STATUSES = {"ERROR", "INVALID", "TIMEOUT"}


class GameOutcome:
    """Result of one game. ``winner`` is 0 (seat0), 1 (seat1), or None (draw)."""

    __slots__ = ("winner", "seconds", "seat0_fail", "seat1_fail", "game_error")

    def __init__(self, winner, seconds, seat0_fail=None, seat1_fail=None, game_error=False):
        self.winner = winner
        self.seconds = seconds
        self.seat0_fail = seat0_fail  # None or one of ERROR/INVALID/TIMEOUT
        self.seat1_fail = seat1_fail
        self.game_error = game_error


def play_game(make, agent0, agent1) -> GameOutcome:
    """Run one game seat0=agent0 vs seat1=agent1. Never raises."""
    t0 = time.time()
    try:
        env = make("cabt")
        env.run([agent0, agent1])
    except Exception:
        # Game-level failure (engine blew up). Record as a no-winner draw.
        return GameOutcome(None, time.time() - t0, game_error=True)

    seconds = time.time() - t0
    final = env.steps[-1]
    try:
        s0 = final[0].get("status")
        s1 = final[1].get("status")
        r0 = final[0].get("reward")
        r1 = final[1].get("reward")
    except Exception:
        return GameOutcome(None, seconds, game_error=True)

    f0 = s0 if s0 in _FAIL_STATUSES else None
    f1 = s1 if s1 in _FAIL_STATUSES else None

    # An agent the engine flagged (crash/invalid/timeout) loses outright.
    if f0 and f1:
        winner = None
    elif f0:
        winner = 1
    elif f1:
        winner = 0
    else:
        # Both finished cleanly — decide on reward (+1 win / -1 loss / equal draw).
        if r0 is None and r1 is None:
            winner = None
        elif r0 is None:
            winner = 1
        elif r1 is None:
            winner = 0
        elif r0 > r1:
            winner = 0
        elif r1 > r0:
            winner = 1
        else:
            winner = None
    return GameOutcome(winner, seconds, f0, f1, game_error=False)


# ---------------------------------------------------------------------------
# Pairing result (all stats from competitor A's perspective).
# ---------------------------------------------------------------------------
class Pairing:
    def __init__(self, a_name, b_name, a_deck, b_deck):
        self.a_name = a_name
        self.b_name = b_name
        self.a_deck = a_deck
        self.b_deck = b_deck
        self.games = 0
        self.a_wins = 0
        self.b_wins = 0
        self.draws = 0
        self.first_wins = 0   # games won by whoever went first (seat 0)
        self.second_wins = 0  # games won by whoever went second (seat 1)
        self.crashes = 0      # ERROR statuses (agent raised)
        self.invalids = 0     # INVALID statuses (illegal action)
        self.timeouts = 0     # TIMEOUT statuses
        self.game_errors = 0  # engine-level failures
        self.times = []       # per-game wall-clock seconds

    # --- derived stats -----------------------------------------------------
    @property
    def a_winrate(self):
        return self.a_wins / self.games if self.games else 0.0

    @property
    def wilson(self):
        return wilson_interval(self.a_wins, self.games)

    @property
    def first_winrate(self):
        return self.first_wins / self.games if self.games else 0.0

    @property
    def second_winrate(self):
        return self.second_wins / self.games if self.games else 0.0

    @property
    def mean_time(self):
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def p95_time(self):
        return percentile(self.times, 0.95)

    def record(self, a_seat, outcome: GameOutcome):
        """Record one game. ``a_seat`` is which seat competitor A occupied."""
        self.games += 1
        self.times.append(outcome.seconds)

        # Tally failure statuses (from either seat).
        for fail in (outcome.seat0_fail, outcome.seat1_fail):
            if fail == "ERROR":
                self.crashes += 1
            elif fail == "INVALID":
                self.invalids += 1
            elif fail == "TIMEOUT":
                self.timeouts += 1
        if outcome.game_error:
            self.game_errors += 1

        # Seat-based (first/second player) tally.
        if outcome.winner == 0:
            self.first_wins += 1
        elif outcome.winner == 1:
            self.second_wins += 1

        # A-perspective tally.
        if outcome.winner is None:
            self.draws += 1
        elif outcome.winner == a_seat:
            self.a_wins += 1
        else:
            self.b_wins += 1


# ---------------------------------------------------------------------------
# Round-robin driver.
# ---------------------------------------------------------------------------
def run_round_robin(make, competitors, games, include_self=False):
    """competitors: list of dicts {name, agent, deck}. Returns list[Pairing]."""
    pairings = []
    n = len(competitors)
    for i in range(n):
        for j in range(n):
            if j <= i and not (include_self and i == j):
                continue
            if i == j and not include_self:
                continue
            a = competitors[i]
            b = competitors[j]
            p = Pairing(a["name"], b["name"], deck_name(a["deck"]), deck_name(b["deck"]))
            print(f"\n[pairing] {a['name']} vs {b['name']}  ({games} games)")
            for g in range(games):
                # Alternate who goes first each game to remove seat bias.
                a_seat = g % 2  # game 0: A is seat0(first); game 1: A is seat1; ...
                if a_seat == 0:
                    outcome = play_game(make, a["agent"], b["agent"])
                else:
                    outcome = play_game(make, b["agent"], a["agent"])
                p.record(a_seat, outcome)
                fails = p.crashes + p.invalids + p.timeouts + p.game_errors
                print(
                    f"  game {g + 1}/{games}: {a['name']} "
                    f"W{p.a_wins}-L{p.b_wins}-D{p.draws}  fails={fails}",
                    end="\r",
                )
            print()  # newline after the \r progress line
            pairings.append(p)
    return pairings


# ---------------------------------------------------------------------------
# Reporting: matchup matrix + overall ratings.
# ---------------------------------------------------------------------------
def build_winrate_lookup(pairings):
    """Map (row_name, col_name) -> row's win-rate vs col (row perspective)."""
    wr = {}
    for p in pairings:
        wr[(p.a_name, p.b_name)] = p.a_winrate
        # B-vs-A win-rate is symmetric: B wins / games.
        wr[(p.b_name, p.a_name)] = (p.b_wins / p.games) if p.games else 0.0
    return wr


def print_matchup_matrix(names, wr):
    """Print a readable row-vs-col win-rate matrix + overall rating column."""
    col_w = max(12, max((len(n) for n in names), default=12) + 1)
    header = " " * col_w + "".join(f"{n[:col_w-1]:>{col_w}}" for n in names) + f"{'RATING':>{col_w}}"
    print("\n" + "=" * len(header))
    print("MATCHUP MATRIX  (row win-rate vs column)")
    print("=" * len(header))
    print(header)
    ratings = {}
    for r in names:
        cells = []
        opp_rates = []
        for c in names:
            if r == c:
                cells.append(f"{'--':>{col_w}}")
                continue
            if (r, c) in wr:
                rate = wr[(r, c)]
                opp_rates.append(rate)
                cells.append(f"{rate * 100:>{col_w - 1}.0f}%")
            else:
                cells.append(f"{'.':>{col_w}}")
        rating = sum(opp_rates) / len(opp_rates) if opp_rates else 0.0
        ratings[r] = rating
        print(f"{r[:col_w-1]:<{col_w}}" + "".join(cells) + f"{rating * 100:>{col_w - 1}.0f}%")
    print("=" * len(header))
    print("RATING = simple mean win-rate across all opponents (higher is better).")

    ranked = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    print("\nOverall ranking:")
    for rank, (name, rating) in enumerate(ranked, 1):
        print(f"  {rank}. {name:<16} {rating * 100:5.1f}%")
    return ratings


# ---------------------------------------------------------------------------
# Output files.
# ---------------------------------------------------------------------------
_CSV_FIELDS = [
    "competitor_a", "competitor_b", "deck_a", "deck_b", "games",
    "a_wins", "a_losses", "draws", "a_winrate", "wilson_low", "wilson_high",
    "first_player_winrate", "second_player_winrate",
    "mean_match_s", "p95_match_s",
    "crashes", "invalids", "timeouts", "game_errors",
]


def pairing_row(p: Pairing):
    lo, hi = p.wilson
    return {
        "competitor_a": p.a_name,
        "competitor_b": p.b_name,
        "deck_a": p.a_deck,
        "deck_b": p.b_deck,
        "games": p.games,
        "a_wins": p.a_wins,
        "a_losses": p.b_wins,
        "draws": p.draws,
        "a_winrate": round(p.a_winrate, 4),
        "wilson_low": round(lo, 4),
        "wilson_high": round(hi, 4),
        "first_player_winrate": round(p.first_winrate, 4),
        "second_player_winrate": round(p.second_winrate, 4),
        "mean_match_s": round(p.mean_time, 3),
        "p95_match_s": round(p.p95_time, 3),
        "crashes": p.crashes,
        "invalids": p.invalids,
        "timeouts": p.timeouts,
        "game_errors": p.game_errors,
    }


def write_outputs(pairings, ratings, csv_path, json_path):
    rows = [pairing_row(p) for p in pairings]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(json_path, "w") as f:
        json.dump({"pairings": rows, "ratings": {k: round(v, 4) for k, v in ratings.items()}}, f, indent=2)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def build_competitors(selected_names):
    """Build the (bound) competitor list, validating deck legality."""
    chosen = []
    for name, base_fn, deck, kind in COMPETITORS:
        if selected_names and name not in selected_names:
            continue
        legal, reason = check_deck_legal(deck)
        tag = "LEGAL" if legal else "ILLEGAL"
        print(f"  [{tag}] {name:<16} deck={deck_name(deck):<16} ({reason})")
        if not legal:
            print(f"    -> skipping {name}: illegal deck ({reason})")
            continue
        if kind == "search":
            bound = bind_deck(base_fn, deck, search_module=search_agent)
        elif kind == "arg":
            bound = bind_deck_arg(base_fn, deck)
        else:
            bound = bind_deck(base_fn, deck, search_module=None)
        chosen.append({"name": name, "agent": bound, "deck": deck})
    return chosen


def main():
    ap = argparse.ArgumentParser(description="Local round-robin gauntlet arena for cabt agents.")
    ap.add_argument("--games", type=int, default=20, help="games per pairing (default 20)")
    ap.add_argument("--competitors", default=None,
                    help="comma-separated subset of competitor names (default: all)")
    ap.add_argument("--include-self", action="store_true",
                    help="also play each competitor against itself (default off)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "arena_results"),
                    help="output path prefix (writes <prefix>.csv and <prefix>.json)")
    ap.add_argument("--list", action="store_true", help="list registered competitors and exit")
    args = ap.parse_args()

    if args.list:
        print("Registered competitors:")
        for name, _fn, deck, kind in COMPETITORS:
            print(f"  {name:<16} deck={deck_name(deck):<16} ({kind})")
        return

    try:
        from kaggle_environments import make
    except ImportError:
        sys.exit("kaggle-environments not installed. Run: pip install kaggle-environments==1.30.1")

    selected = None
    if args.competitors:
        selected = {s.strip() for s in args.competitors.split(",") if s.strip()}

    print("=" * 60)
    print("ARENA - deck legality check")
    print("=" * 60)
    competitors = build_competitors(selected)
    if len(competitors) < 2:
        sys.exit("Need at least 2 legal competitors to run a round-robin.")

    names = [c["name"] for c in competitors]
    print("\n" + "=" * 60)
    print(f"ROUND-ROBIN: {len(competitors)} competitors, {args.games} games/pairing, "
          f"self-play {'ON' if args.include_self else 'OFF'}")
    print("Competitors:", ", ".join(names))
    print("=" * 60)

    t0 = time.time()
    try:
        pairings = run_round_robin(make, competitors, args.games, include_self=args.include_self)
    except Exception:
        print("FATAL: round-robin loop raised:")
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.time() - t0

    # ---- per-pairing summary ----
    print("\n" + "=" * 60)
    print("PER-PAIRING RESULTS")
    print("=" * 60)
    for p in pairings:
        lo, hi = p.wilson
        fails = p.crashes + p.invalids + p.timeouts + p.game_errors
        print(f"\n{p.a_name} vs {p.b_name}  ({p.games} games)")
        print(f"  record (A persp.): {p.a_wins}W-{p.b_wins}L-{p.draws}D  "
              f"winrate {p.a_winrate:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
        print(f"  first-player winrate: {p.first_winrate:.1%}   "
              f"second-player winrate: {p.second_winrate:.1%}")
        print(f"  match time: mean {p.mean_time:.2f}s  p95 {p.p95_time:.2f}s")
        print(f"  fails: crashes={p.crashes} invalids={p.invalids} "
              f"timeouts={p.timeouts} game_errors={p.game_errors} (total {fails})")

    # ---- matchup matrix + ratings ----
    wr = build_winrate_lookup(pairings)
    ratings = print_matchup_matrix(names, wr)

    # ---- outputs ----
    write_outputs(pairings, ratings, args.out + ".csv", args.out + ".json")

    total_games = sum(p.games for p in pairings)
    print(f"\nDone: {total_games} games across {len(pairings)} pairings in {elapsed:.1f}s "
          f"({elapsed / max(total_games, 1):.2f}s/game).")


if __name__ == "__main__":
    main()
