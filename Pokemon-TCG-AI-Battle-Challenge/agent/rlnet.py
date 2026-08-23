"""
Shared neural-net + MCTS module for the PTCG AI Battle agent.

Used by BOTH the training notebook (self-play) and the inference agent (main.py),
so the state encoding is guaranteed identical. Architecture, feature builders,
and MCTS are adapted from the official kiyotah sample
(references/reinforcement-learning-and-mcts-sample-code.ipynb).

Design for the competition:
  * `RLAgent(model_path).act(obs_dict) -> list[int] | None`
    Runs determinized MCTS guided by the value/policy net and returns the most-
    visited action. Returns None on ANY failure so the caller can fall back to
    the search/heuristic agent (never crash, never time out).
  * SEARCH_COUNT and a wall-clock budget bound the per-decision cost so a match
    stays under the 10-minute limit.

Requires: torch, and the cg engine (cg.api) importable. If either is missing the
module still imports but RLAgent.available is False.
"""

from __future__ import annotations

import math
import os
import random
import time

try:
    import torch
    import torch.nn
    import torch.nn.functional
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from cg.api import (
        AreaType, OptionType, SelectContext,
        all_attack, all_card_data, to_observation_class,
        search_begin, search_step, search_end,
    )
    _HAS_CG = True
except Exception:
    _HAS_CG = False


NN_AVAILABLE = _HAS_TORCH and _HAS_CG

# ---------------------------------------------------------------------------
# Dimensions derived from the card catalog (identical in training & inference
# because both load the same cg-lib).
# ---------------------------------------------------------------------------
if _HAS_CG:
    _ALL_CARD = all_card_data()
    CARD_COUNT = max(_ALL_CARD, key=lambda c: c.cardId).cardId + 1
    ATTACK_COUNT = max(all_attack(), key=lambda a: a.attackId).attackId + 1
else:
    CARD_COUNT = 2000
    ATTACK_COUNT = 1600

NUM_WORDS_ENCODER = 24
ENCODER_SIZE = 22000
DECODER_MAIN_FEATURE = 8
DECODER_ATTACK_OFFSET = 14
DECODER_CARD_OFFSET = DECODER_ATTACK_OFFSET + ATTACK_COUNT
_RECOVER_SC = SelectContext.RECOVER_SPECIAL_CONDITION if _HAS_CG else 48
DECODER_SIZE = DECODER_CARD_OFFSET + (1 + DECODER_MAIN_FEATURE + _RECOVER_SC) * CARD_COUNT

# Default model hyperparameters (must match the trained checkpoint).
D_MODEL = 128
NUM_HEADS = 2
D_FF = 256
N_ENC = 1
N_DEC = 1


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if _HAS_TORCH:
    class DecoderLayer(torch.nn.Module):
        def __init__(self, d_model, num_heads, d_ff):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(d_model, num_heads)
            self.fc1 = torch.nn.Linear(d_model, d_ff)
            self.fc2 = torch.nn.Linear(d_ff, d_model)
            self.norm1 = torch.nn.LayerNorm(d_model)
            self.norm2 = torch.nn.LayerNorm(d_model)

        def forward(self, x, encoder_out):
            y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
            res = self.norm1(x + y)
            y = self.fc1(res)
            y = torch.nn.functional.relu(y)
            y = self.fc2(y)
            return self.norm2(res + y)

    class MyModel(torch.nn.Module):
        def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
                     n_enc=N_ENC, n_dec=N_DEC):
            super().__init__()
            self.d_model = d_model
            self.encoder_bag = torch.nn.EmbeddingBag(ENCODER_SIZE, d_model, mode="sum")
            enc_layer = torch.nn.TransformerEncoderLayer(d_model, num_heads, d_ff, 0)
            self.encoder = torch.nn.TransformerEncoder(enc_layer, n_enc, enable_nested_tensor=False)
            self.encoder_fc = torch.nn.Linear(d_model, 1)
            self.decoder_bag = torch.nn.EmbeddingBag(DECODER_SIZE, d_model, mode="sum")
            self.decoder = torch.nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(n_dec)])
            self.decoder_fc = torch.nn.Linear(d_model, 1)

        def forward(self, ie, ve, oe, idc, vdc, odc):
            v = self.encoder_bag(ie, oe, ve)
            v = v.reshape(-1, NUM_WORDS_ENCODER, self.d_model).transpose(0, 1)
            batch = v.size(1)
            enc_out = self.encoder(v)
            vv = self.encoder_fc(enc_out)
            vv = torch.tanh(vv.mean(0))
            p = self.decoder_bag(idc, odc, vdc)
            p = p.reshape(batch, -1, self.d_model).transpose(0, 1)
            for layer in self.decoder:
                p = layer(p, enc_out)
            p = self.decoder_fc(p)
            p = p.transpose(0, 1).view(batch, -1)
            p = torch.tanh(p)
            return vv, p


# ---------------------------------------------------------------------------
# Sparse feature vector + encoders (verbatim logic from the sample)
# ---------------------------------------------------------------------------
class SparseVector:
    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []
        self.pos = 0

    def add(self, index, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos + index)
            self.value.append(value)

    def add_pos(self, pos):
        self.pos += pos

    def add_single(self, value):
        value = float(value)
        if value != 0.0:
            self.index.append(self.pos)
            self.value.append(value)
        self.pos += 1

    def word_start(self):
        self.offset.append(len(self.index))


def _add_card(sv, card):
    if card is not None:
        sv.add(card.id, 1)
    sv.add_pos(CARD_COUNT)


def _add_cards(sv, cards, value):
    if cards is not None:
        for card in cards:
            sv.add(card.id, value)
    sv.add_pos(CARD_COUNT)


def _add_pokemon(sv, poke):
    if poke is None:
        sv.add_single(1)
        sv.add_pos(1 + 3 * CARD_COUNT)
    else:
        sv.add_single(0)
        sv.add_single(poke.hp / 400)
        _add_card(sv, poke)
        _add_cards(sv, poke.tools, 1.0)
        _add_cards(sv, poke.energyCards, 0.5)


def _add_player(sv, ps):
    sv.add_single(ps.deckCount / 60)
    sv.add_single(len(ps.discard) / 60)
    sv.add_single(ps.handCount / 8)
    sv.add_single(len(ps.bench) / 5)
    sv.add(len(ps.prize), 1)
    sv.add_pos(7)
    sv.add_single(ps.poisoned)
    sv.add_single(ps.burned)
    sv.add_single(ps.asleep)
    sv.add_single(ps.paralyzed)
    sv.add_single(ps.confused)
    _add_cards(sv, ps.discard, 0.25)


def get_encoder_input(obs, your_deck):
    your_index = obs.current.yourIndex
    state = obs.current
    sv = SparseVector()
    for i in range(2):
        ps = state.players[i ^ your_index]
        for j in range(8):
            sv.word_start()
            pos = sv.pos
            if j < len(ps.bench):
                _add_pokemon(sv, ps.bench[j])
            else:
                _add_pokemon(sv, None)
            if j != 7:
                sv.pos = pos
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        if 0 < len(ps.active):
            _add_pokemon(sv, ps.active[0])
        else:
            _add_pokemon(sv, None)
    for i in range(2):
        ps = state.players[i ^ your_index]
        sv.word_start()
        _add_player(sv, ps)
    sv.word_start()
    _add_cards(sv, state.players[your_index].hand, 0.25)
    sv.word_start()
    for cid in your_deck:
        sv.add(cid, 0.25)
    sv.add_pos(CARD_COUNT)
    sv.word_start()
    _add_cards(sv, state.stadium, 1.0)
    sv.word_start()
    sv.add_single(1)
    sv.add_single(state.turn / 10)
    sv.add_single(state.firstPlayer == your_index)
    return sv


def _get_card(obs, area, index, player_index):
    ps = obs.current.players[player_index]
    if area == AreaType.DECK:
        return obs.select.deck[index]
    if area == AreaType.HAND:
        return ps.hand[index]
    if area == AreaType.DISCARD:
        return ps.discard[index]
    if area == AreaType.ACTIVE:
        return ps.active[index]
    if area == AreaType.BENCH:
        return ps.bench[index]
    if area == AreaType.PRIZE:
        return ps.prize[index]
    if area == AreaType.STADIUM:
        return obs.current.stadium[index]
    if area == AreaType.LOOKING:
        return obs.current.looking[index]
    return None


def _decoder_main(sv, feature_index, card):
    if card is not None:
        sv.add(DECODER_CARD_OFFSET + feature_index * CARD_COUNT + card.id, 1)


def _decoder_card_id(sv, context, card_id):
    sv.add(DECODER_CARD_OFFSET + (DECODER_MAIN_FEATURE + context) * CARD_COUNT + card_id, 1)


def _decoder_card(sv, context, card):
    if card is not None:
        _decoder_card_id(sv, context, card.id)


def get_decoder_input(obs, actions):
    sv = SparseVector()
    your_index = obs.current.yourIndex
    ps = obs.current.players[your_index]
    context = obs.select.context
    for action in actions:
        sv.word_start()
        if len(action) == 0:
            sv.add(0, 1)
            continue
        for i in action:
            o = obs.select.option[i]
            t = o.type
            if t == OptionType.END:
                sv.add(1, 1)
            elif t == OptionType.YES:
                sv.add(2, 1)
            elif t == OptionType.NO:
                sv.add(3, 1)
            elif t == OptionType.SPECIAL_CONDITION:
                sv.add(4 + o.specialConditionType, 1)
            elif t == OptionType.NUMBER:
                sv.add(9 + min(o.number, 4), 1)
            elif t == OptionType.ATTACK:
                sv.add(DECODER_ATTACK_OFFSET + o.attackId, 1)
            elif t == OptionType.PLAY:
                _decoder_main(sv, 0, ps.hand[o.index])
            elif t == OptionType.ATTACH:
                _decoder_main(sv, 1, _get_card(obs, o.area, o.index, your_index))
                _decoder_main(sv, 2, _get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
            elif t == OptionType.EVOLVE:
                _decoder_main(sv, 3, _get_card(obs, o.area, o.index, your_index))
                _decoder_main(sv, 4, _get_card(obs, o.inPlayArea, o.inPlayIndex, your_index))
            elif t == OptionType.ABILITY:
                _decoder_main(sv, 5, _get_card(obs, o.area, o.index, your_index))
            elif t == OptionType.DISCARD:
                _decoder_main(sv, 6, _get_card(obs, o.area, o.index, your_index))
            elif t == OptionType.RETREAT:
                _decoder_main(sv, 7, ps.active[0])
            elif t == OptionType.CARD:
                _decoder_card(sv, context, _get_card(obs, o.area, o.index, o.playerIndex))
            elif t == OptionType.TOOL_CARD:
                card = _get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.tools[o.toolIndex])
            elif t in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                card = _get_card(obs, o.area, o.index, o.playerIndex)
                _decoder_card(sv, context, card.energyCards[o.energyIndex])
            elif t == OptionType.SKILL:
                _decoder_card_id(sv, context, o.cardId)
    return sv


def eval_nn(sv_enc, sv_dec, model):
    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
    )
    return value.tolist()[0][0], policy.tolist()[0]


# ---------------------------------------------------------------------------
# MCTS (inference-only: no training-sample generation)
# ---------------------------------------------------------------------------
def _enum_actions(obs, cap=64):
    actions = []
    indices = list(range(obs.select.maxCount))
    for _ in range(cap):
        actions.append(indices.copy())
        for i in range(len(indices)):
            index = len(indices) - i - 1
            if indices[index] < len(obs.select.option) - i - 1:
                indices[index] += 1
                for j in range(index + 1, len(indices)):
                    indices[j] = indices[j - 1] + 1
                break
        else:
            break
    return actions


class _Child:
    __slots__ = ("node", "select", "prob")

    def __init__(self, select, prob):
        self.node = None
        self.select = select
        self.prob = prob


class _Node:
    __slots__ = ("value", "total", "visit", "parent", "children", "state")

    def __init__(self, parent, state):
        self.value = -2.0
        self.total = 0.0
        self.visit = 0
        self.parent = parent
        self.children = []
        self.state = state

    def backprop(self, value):
        node = self
        while node is not None:
            node.total += value
            node.visit += 1
            node = node.parent


class RLAgent:
    """Determinized MCTS guided by the value/policy net. act() returns a
    list[int] action, or None on any failure (so caller can fall back)."""

    def __init__(self, model_path=None, search_count=16, time_budget_s=8.0, deck=None):
        self.available = False
        self.search_count = search_count
        self.time_budget_s = time_budget_s
        self.deck = list(deck) if deck else None
        self.model = None
        if not NN_AVAILABLE:
            return
        try:
            # Architecture must match the trained checkpoint. Prefer an .arch.json
            # sidecar (written by the trainer); else infer d_model from the
            # weights and fall back to defaults for the rest.
            arch = {}
            if model_path:
                import json
                side = model_path + ".arch.json"
                if os.path.exists(side):
                    try:
                        with open(side) as f:
                            arch = json.load(f)
                    except Exception:
                        arch = {}
            self.model = MyModel(
                d_model=arch.get("d_model", D_MODEL),
                num_heads=arch.get("num_heads", NUM_HEADS),
                d_ff=arch.get("d_ff", D_FF),
                n_enc=arch.get("n_enc", N_ENC),
                n_dec=arch.get("n_dec", N_DEC),
            )
            if model_path:
                sd = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(sd)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False

    def _create_node(self, parent, state, your_index, your_deck):
        node = _Node(parent, state)
        obs = state.observation
        cur = obs.current
        if cur.result >= 0:
            node.value = 0 if cur.result == 2 else (1 if cur.result == your_index else -1)
            node.backprop(node.value)
            return node
        actions = _enum_actions(obs)
        sv_enc = get_encoder_input(obs, your_deck)
        sv_dec = get_decoder_input(obs, actions)
        value, policy = eval_nn(sv_enc, sv_dec, self.model)
        v = value if cur.yourIndex == your_index else -value
        node.value = v
        node.backprop(v)
        s = 0.0
        for i in range(len(policy)):
            p = math.exp(policy[i] * 10.0)
            node.children.append(_Child(actions[i], p))
            s += p
        for ch in node.children:
            ch.prob /= s if s else 1.0
        return node

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def act(self, obs_dict):
        if not self.available:
            return None
        try:
            sel = obs_dict.get("select")
            if sel is None:
                return list(self.deck) if self.deck else None
            if (sel.get("maxCount", 1) or 1) != 1 or len(sel.get("option") or []) <= 1:
                return None  # let the caller's fast path handle trivial/multi picks
            O = to_observation_class(obs_dict)
            cur = O.current
            me = cur.yourIndex
            mp, op = cur.players[me], cur.players[1 - me]
            deck = self.deck or []
            det = dict(
                your_deck=random.sample(deck, mp.deckCount) if deck and mp.deckCount <= len(deck) else [random.choice(deck) for _ in range(mp.deckCount)] if deck else [],
                your_prize=[random.choice(deck) for _ in range(len(mp.prize))] if deck else [],
                opponent_deck=[1072] * op.deckCount,
                opponent_prize=[1] * len(op.prize),
                opponent_hand=[1] * op.handCount,
                opponent_active=[1072] if (op.active and op.active[0] is None) else [],
            )
            ss = search_begin(O, **det)
            root = self._create_node(None, ss, me, det["your_deck"])
            deadline = time.time() + self.time_budget_s
            for _ in range(self.search_count):
                if time.time() > deadline:
                    break
                cur_node = root
                while True:
                    best_v, nxt = -1e18, None
                    c = 0.4 * math.sqrt(cur_node.visit)
                    for child in cur_node.children:
                        if child.node is None:
                            q = cur_node.total / max(cur_node.visit, 1)
                            visit = 0
                        else:
                            q = child.node.total / max(child.node.visit, 1)
                            visit = child.node.visit
                        if cur_node.state.observation.current.yourIndex != me:
                            q = -q
                        u = q + c * child.prob / (1 + visit)
                        if u > best_v:
                            best_v, nxt = u, child
                    if nxt is None:
                        break
                    if nxt.node is None:
                        st = search_step(cur_node.state.searchId, nxt.select)
                        nxt.node = self._create_node(cur_node, st, me, det["your_deck"])
                        break
                    cur_node = nxt.node
                    if cur_node.state.observation.current.result >= 0:
                        cur_node.backprop(cur_node.value)
                        break
            best, best_visit = None, -1
            for child in root.children:
                if child.node is not None and child.node.visit > best_visit:
                    best_visit, best = child.node.visit, child
            search_end()
            return best.select if best is not None else None
        except Exception:
            try:
                search_end()
            except Exception:
                pass
            return None
