"""Conservative pairwise causal corrections over frozen router-v12.

The training data rebases the policy's baseline action to option zero and asks
whether one particular alternative was better.  Live inference must reproduce
that transformation; treating the model as an ordinary option scorer is wrong.
"""
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
from train_entity_advantage import EntityAdvantage
from train_state_ranker import collate

_SPEC = importlib.util.spec_from_file_location(
    "v12_pairwise_frozen_base", os.path.join(ROOT, "agent", "router_v12", "main.py")
)
_BASE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BASE)
_MODEL = _CK = None


def _load():
    global _MODEL, _CK
    if _MODEL is not None:
        return
    path = os.environ.get(
        "V12_PAIRWISE_CKPT", os.path.join(ROOT, "agent", "v12_pairwise_rebased_s43_l3.pt")
    )
    _CK = torch.load(path, map_location="cpu", weights_only=False)
    _MODEL = EntityAdvantage(layers=int(_CK["layers"]))
    _MODEL.load_state_dict(_CK["model"])
    _MODEL.eval()


def _rows(obs, baseline_index):
    sel = obs["select"]
    cur = obs["current"]
    me = cur["yourIndex"]
    feats = [option_features(option, cur, sel, me) for option in sel["option"]]
    order = [baseline_index] + [i for i in range(len(feats)) if i != baseline_index]
    inverse = {original: rebased for rebased, original in enumerate(order)}
    reordered = [feats[i] for i in order]
    common = {
        "episode": "live",
        "ctx": board_context(cur, me),
        "state": state_tokens(cur, me),
        "cids": [f[0] for f in reordered],
        "types": [f[1] for f in reordered],
        "nums": [f[2] for f in reordered],
        "ctxid": int(sel.get("context", 0) or 0),
    }
    rows, originals = [], []
    for original in order[1:]:
        rows.append({**common, "y": inverse[original]})
        originals.append(original)
    return rows, originals


def agent(obs, configuration=None):
    baseline = _BASE.agent(obs, configuration)
    try:
        sel = obs.get("select")
        if (sel is None or len(sel.get("option") or []) < 2
                or (sel.get("maxCount", 1) or 1) != 1 or len(baseline) != 1):
            return baseline
        base_index = int(baseline[0])
        if not 0 <= base_index < len(sel["option"]):
            return baseline
        _load()
        rows, originals = _rows(obs, base_index)
        batch = [tensor for tensor in collate(rows, _CK["stats"], maxs=80)]
        with torch.no_grad():
            logits = _MODEL(batch)[:, (0, 2)]
            positive = logits.softmax(-1)[:, 1]
        best = int(positive.argmax())
        threshold = float(os.environ.get("V12_PAIRWISE_THRESHOLD", _CK["gate"]["threshold"]))
        if float(positive[best]) >= threshold:
            return [originals[best]]
    except Exception:
        pass
    return baseline
