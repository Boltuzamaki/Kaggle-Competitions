"""Sparse learned corrections on top of the frozen router-v12 policy."""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = os.path.join(ROOT, "tools", "rank")
if RANK not in sys.path:
    sys.path.insert(0, RANK)

from extract_dataset import board_context, option_features, state_tokens
from train_state_ranker import StateRanker, collate

_SPEC = importlib.util.spec_from_file_location(
    "v12_neural_frozen_base", os.path.join(ROOT, "agent", "router_v12", "main.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_MODEL = _CK = None


def _load():
    global _MODEL, _CK
    if _MODEL is not None:
        return
    path = os.environ.get("V12_NEURAL_CKPT", os.path.join(ROOT, "agent", "v12_causal.pt"))
    _CK = torch.load(path, map_location="cpu", weights_only=False)
    _MODEL = StateRanker()
    _MODEL.load_state_dict(_CK["model"])
    _MODEL.eval()


def _row(obs):
    sel = obs["select"]
    cur = obs["current"]
    me = cur["yourIndex"]
    feats = [option_features(option, cur, sel, me) for option in sel["option"]]
    return {
        "episode": "live", "ctx": board_context(cur, me),
        "state": state_tokens(cur, me), "cids": [f[0] for f in feats],
        "types": [f[1] for f in feats], "nums": [f[2] for f in feats],
        "y": 0, "ctxid": int(sel.get("context", 0) or 0),
    }


def agent(obs, configuration=None):
    baseline = _BASE.agent(obs, configuration)
    try:
        sel = obs.get("select")
        if sel is None or len(sel.get("option") or []) < 2 or (sel.get("maxCount", 1) or 1) != 1:
            return baseline
        _load()
        batch = collate([_row(obs)], _CK["stats"])
        with torch.no_grad():
            scores = _MODEL(*batch[:-1])[0]
        pred = int(scores.argmax())
        margin = float(scores[pred] - scores[0])
        threshold = float(os.environ.get("V12_NEURAL_MARGIN", _CK["gate"]["threshold"]))
        if pred != 0 and margin >= threshold:
            return [pred]
    except Exception:
        pass
    return baseline
