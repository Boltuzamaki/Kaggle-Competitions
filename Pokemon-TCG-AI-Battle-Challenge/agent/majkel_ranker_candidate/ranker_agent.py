"""Card-identity ranker used as a prior over legal options, refined by v3 search.

Two lessons from the public post-mortem (discussion 713608) shape this design:

  * Pure imitation plateaus BELOW its teacher -- their DAgger clone topped out at
    41% head-to-head against the very heuristic it was cloning, because small
    errors compound into states the teacher never visited. So the ranker does not
    act alone by default.
  * "Search should refine a strong heuristic, not replace it", and beam value is
    inversely proportional to base-policy quality. So the ranker supplies the
    candidate ordering / prior and v3's determinized search adjudicates the
    shortlist, rather than either one overriding the other outright.

MODE selects the behaviour:
  "prior"  -- rank options, hand the top-K to v3 search, play the search's pick
  "policy" -- play the ranker's top option directly (ablation)
  "off"    -- plain v3 (control)

Every path degrades to v3 and then to a legal index on any failure: a crash or a
timeout is an instant loss, and the model is the least trustworthy component here.
"""
from __future__ import annotations

import os

import search_agent

MODE = os.environ.get("RANKER_MODE", "prior")
TOPK = int(os.environ.get("RANKER_TOPK", "5"))
_MODEL = None
_STATS = None
_TORCH = None
_READY = False

_CTX_DIM, _NUM_DIM = 21, 8
_N_CARDS, _N_TYPES, _N_CTXTYPE = 1400, 20, 64


def _load():
    """Load the checkpoint once. Any failure permanently disables the ranker."""
    global _MODEL, _STATS, _TORCH, _READY
    if _READY:
        return _MODEL is not None
    _READY = True
    try:
        import torch
        import torch.nn as nn
        _TORCH = torch

        here = os.path.dirname(os.path.abspath(__file__))
        path = None
        for cand in (os.path.join(here, "ranker.pt"),
                     "/kaggle_simulations/agent/ranker.pt", "ranker.pt"):
            if os.path.exists(cand):
                path = cand
                break
        if path is None:
            return False
        ck = torch.load(path, map_location="cpu", weights_only=False)
        d_card = ck.get("d_card", 64)
        d_hidden = 256

        class Ranker(nn.Module):
            def __init__(self):
                super().__init__()
                self.card = nn.Embedding(_N_CARDS, d_card)
                self.otype = nn.Embedding(_N_TYPES, 16)
                self.sctx = nn.Embedding(_N_CTXTYPE, 16)
                self.ctx_mlp = nn.Sequential(
                    nn.Linear(_CTX_DIM, d_hidden), nn.ReLU(),
                    nn.Linear(d_hidden, d_hidden), nn.ReLU())
                self.opt_mlp = nn.Sequential(
                    nn.Linear(d_card + 16 + _NUM_DIM, d_hidden), nn.ReLU(),
                    nn.Linear(d_hidden, d_hidden), nn.ReLU())
                self.score = nn.Sequential(
                    nn.Linear(d_hidden * 2 + 16, d_hidden), nn.ReLU(),
                    nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                    nn.Linear(d_hidden, 1))

            def forward(self, ctx, ctxid, cids, types, nums):
                O = cids.shape[1]
                c = self.ctx_mlp(ctx)
                sc = self.sctx(ctxid)
                opt = self.opt_mlp(
                    _TORCH.cat([self.card(cids), self.otype(types), nums], dim=-1))
                joint = _TORCH.cat([opt,
                                    c.unsqueeze(1).expand(-1, O, -1),
                                    sc.unsqueeze(1).expand(-1, O, -1)], dim=-1)
                return self.score(joint).squeeze(-1)

        m = Ranker()
        m.load_state_dict(ck["model"])
        m.eval()
        _MODEL, _STATS = m, ck["stats"]
        return True
    except Exception:
        _MODEL = None
        return False


def _ctx_feats(cur, me):
    try:
        mp = cur["players"][me]
        op = cur["players"][1 - me]

        def mons(p):
            return [m for m in list(p.get("active") or []) + list(p.get("bench") or []) if m]

        my, oo = mons(mp), mons(op)
        f = [float(cur.get("turn", 0)), float(len(mp.get("prize") or [])),
             float(len(op.get("prize") or [])), float(mp.get("handCount", 0) or 0),
             float(op.get("handCount", 0) or 0), float(mp.get("deckCount", 0) or 0),
             float(op.get("deckCount", 0) or 0), float(len(my)), float(len(oo)),
             float(sum((m.get("hp") or 0) for m in my)),
             float(sum((m.get("hp") or 0) for m in oo)),
             float(sum(len(m.get("energies") or []) for m in my)),
             float(sum(len(m.get("energies") or []) for m in oo)),
             float(1 if cur.get("energyAttached") else 0),
             float(1 if cur.get("supporterPlayed") else 0),
             float(1 if cur.get("retreated") else 0),
             float(1 if cur.get("firstPlayer") == me else 0)]
        a0 = (mp.get("active") or [None])[0]
        a1 = (op.get("active") or [None])[0]
        f += [float(a0["id"]) if a0 else 0.0, float(a0.get("hp") or 0) if a0 else 0.0,
              float(a1["id"]) if a1 else 0.0, float(a1.get("hp") or 0) if a1 else 0.0]
        return f
    except Exception:
        return [0.0] * _CTX_DIM


def rank(obs_dict):
    """Return option indices best-first, or None if the model is unavailable."""
    if not _load():
        return None
    try:
        sel = obs_dict.get("select")
        cur = obs_dict.get("current")
        if sel is None or not cur:
            return None
        opts = sel.get("option") or []
        n = len(opts)
        if n < 2:
            return None
        me = cur.get("yourIndex")
        if me is None:
            return None

        s = _STATS
        ctx = _ctx_feats(cur, me)
        ctx = [(v - s["ctx_mu"][i]) / s["ctx_sd"][i] for i, v in enumerate(ctx)]

        cids, types, nums = [], [], []
        for o in opts:
            cids.append(min(max(int(o.get("cardId", 0) or 0), 0), _N_CARDS - 1))
            types.append(min(max(int(o.get("type", 0) or 0), 0), _N_TYPES - 1))
            raw = [float(o.get("number", 0) or 0), float(o.get("area", 0) or 0),
                   float(o.get("index", 0) or 0),
                   float(o.get("playerIndex", -1) if o.get("playerIndex") is not None else -1),
                   float(o.get("count", 0) or 0), float(o.get("inPlayArea", 0) or 0),
                   float(o.get("inPlayIndex", 0) or 0), float(o.get("attackId", 0) or 0)]
            nums.append([(v - s["num_mu"][i]) / s["num_sd"][i] for i, v in enumerate(raw)])

        t = _TORCH
        with t.no_grad():
            sc = _MODEL(t.tensor([ctx], dtype=t.float),
                        t.tensor([min(int(sel.get("context", 0) or 0), _N_CTXTYPE - 1)]),
                        t.tensor([cids]), t.tensor([types]),
                        t.tensor([nums], dtype=t.float))[0]
        return t.argsort(sc, descending=True).tolist()
    except Exception:
        return None


def agent(obs_dict: dict) -> list[int]:
    try:
        select = obs_dict.get("select")
        if select is None:
            return list(search_agent.DECK)
        opts = select.get("option") or []
        n = len(opts)
        if n == 0:
            return []
        max_count = select.get("maxCount", 1) or 1

        if MODE != "off" and max_count == 1 and n > 1:
            order = rank(obs_dict)
            if order:
                if MODE == "policy":
                    return [order[0]]
                # "prior": let v3 adjudicate the ranker's shortlist. If the search
                # is unavailable the ranker's own top choice still stands.
                try:
                    pick = search_agent._search_choice(obs_dict)
                    short = order[:max(2, TOPK)]
                    return [pick] if pick in short else [short[0]]
                except Exception:
                    return [order[0]]
        return search_agent.agent(obs_dict)
    except Exception:
        try:
            return search_agent.agent(obs_dict)
        except Exception:
            sel = obs_dict.get("select")
            if sel is None:
                return list(search_agent.DECK)
            n = len(sel.get("option") or [])
            mc = sel.get("maxCount", 1) or 1
            return list(range(min(mc, n))) if n else []


def make_agent(deck, mode=None, topk=None):
    frozen = list(deck)

    def run(obs_dict):
        global MODE, TOPK
        if mode:
            MODE = mode
        if topk:
            TOPK = topk
        search_agent.DECK = frozen
        return agent(obs_dict)
    return run
