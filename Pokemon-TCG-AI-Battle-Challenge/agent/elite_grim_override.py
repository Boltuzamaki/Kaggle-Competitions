"""Conservative elite-BC overrides on top of the local Grim policy."""
from __future__ import annotations
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "tools", "rank")
if RANK not in sys.path:
    sys.path.insert(0, RANK)

import torch
import grimmsnarl_policy
from extract_dataset import board_context, option_features, state_tokens
from train_state_ranker import StateRanker, collate

_MODEL = _CK = None


def _load():
    global _MODEL, _CK
    if _MODEL is not None:
        return
    path = os.environ.get("ELITE_GRIM_CKPT",
                          os.path.join(ROOT, "agent", "elite_grim_gated_w1.pt"))
    _CK = torch.load(path, map_location="cpu", weights_only=False)
    _MODEL = StateRanker()
    _MODEL.load_state_dict(_CK["model"])
    _MODEL.eval()


def _row(obs):
    sel = obs["select"]; cur = obs["current"]; me = cur["yourIndex"]
    feats = [option_features(o, cur, sel, me) for o in sel["option"]]
    return {"episode": "live", "ctx": board_context(cur, me),
            "state": state_tokens(cur, me), "cids": [f[0] for f in feats],
            "types": [f[1] for f in feats], "nums": [f[2] for f in feats],
            "y": 0, "ctxid": int(sel.get("context", 0) or 0)}


def make_agent(deck, threshold=None):
    frozen = list(deck)
    def run(obs):
        baseline = grimmsnarl_policy.grimmsnarl_agent(obs, frozen)
        try:
            sel = obs.get("select")
            if sel is None or len(sel.get("option") or []) < 2 \
                    or (sel.get("maxCount", 1) or 1) != 1:
                return baseline
            _load()
            batch = collate([_row(obs)], _CK["stats"])
            with torch.no_grad():
                scores = _MODEL(*batch[:-1])[0]
            pred = int(scores.argmax())
            gate = float(_CK["gate"]["threshold"] if threshold is None else threshold)
            margin = float(scores[pred] - scores[0])
            if pred != 0 and margin >= gate:
                return [pred]
        except Exception:
            pass
        return baseline
    return run
