"""
Entry-point agent (v4) for the Pokémon TCG AI Battle Challenge (cabt).

Contract: agent(obs) -> list[int]. `obs["select"] is None` = deck-selection ->
return the 60 card IDs; otherwise return chosen index/indices into
`obs["select"]["option"]`.

Layered policy (strongest available wins; each layer falls back to the next):
  1. RL net + MCTS  — a value/policy Transformer guiding determinized MCTS
                      (agent/rlnet.py). Used only if torch + cg + trained weights
                      (model.pth) are all present.
  2. Shallow search — v3 determinized 1-turn lookahead (agent/search_agent.py).
  3. Heuristic      — type-priority fallback (inside search_agent).

Robustness: any import/runtime failure at any layer degrades to the next, so the
agent ALWAYS returns a legal move — never crash, never time out (both = loss).
Kaggle loads this file with exec() (no __file__), so the deck is embedded in
search_agent and weights are located via the imported rlnet module's __file__.
"""

from __future__ import annotations

import os

# --- Layer 2/3: proven shallow-search + heuristic (self-contained, always works).
from search_agent import agent as _search_agent, DECK  # noqa: E402

# --- Layer 1: optional RL net + MCTS.
_RL = None
try:
    import rlnet  # normal import -> has __file__, unlike the exec'd main.py

    if getattr(rlnet, "NN_AVAILABLE", False):
        _bundle_dir = os.path.dirname(os.path.abspath(rlnet.__file__))
        _model_path = None
        for cand in (os.path.join(_bundle_dir, "model.pth"),):
            if os.path.exists(cand):
                _model_path = cand
                break
        if _model_path is not None:
            # Inference MCTS depth: bundled rl_config.py sets the shipped value;
            # env RL_SEARCH overrides for local testing.
            _search, _budget = 32, 8.0
            _enabled = False
            try:
                import rl_config  # optional bundled file
                _enabled = bool(getattr(rl_config, "ENABLED", False))
                _search = int(getattr(rl_config, "SEARCH", _search))
                _budget = float(getattr(rl_config, "BUDGET", _budget))
            except Exception:
                pass
            if os.environ.get("RL_ENABLED"):
                _enabled = os.environ["RL_ENABLED"].strip().lower() in {
                    "1", "true", "yes", "on"
                }
            if os.environ.get("RL_SEARCH"):
                _search = int(os.environ["RL_SEARCH"])
            if _enabled:
                _rl = rlnet.RLAgent(model_path=_model_path, search_count=_search,
                                    time_budget_s=_budget, deck=DECK)
                if _rl.available:
                    _RL = _rl
except Exception:
    _RL = None


def agent(obs_dict: dict) -> list[int]:
    try:
        select = obs_dict.get("select")
        if select is None:
            return list(DECK)

        # Layer 1: RL net + MCTS (returns None to defer to the search layer).
        if _RL is not None:
            try:
                move = _RL.act(obs_dict)
                if move:
                    return move
            except Exception:
                pass

        # Layer 2/3: shallow search + heuristic.
        return _search_agent(obs_dict)

    except Exception:
        # Absolute safety net.
        try:
            select = obs_dict.get("select")
            if select is None:
                return list(DECK)
            n = len(select.get("option") or [])
            mc = select.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []
        except Exception:
            return list(DECK) if DECK else [0]


if __name__ == "__main__":
    print(f"Embedded deck: {len(DECK)} cards | RL net active: {_RL is not None}")
    print("Deck-select returns", len(agent({"select": None})), "card IDs")
    print("Move:", agent({"current": {}, "select": {"option": [{"type": 13}, {"type": 14}], "maxCount": 1}}))
