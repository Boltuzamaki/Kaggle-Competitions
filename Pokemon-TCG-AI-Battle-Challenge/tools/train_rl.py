"""
Self-play RL trainer for the PTCG AI Battle net (AlphaZero-style: value/policy
Transformer + MCTS), adapted from the official kiyotah sample. Reuses the SHARED
encoders/model in agent/rlnet.py so the trained weights load 1:1 at inference.

Run locally (tiny, CPU, just to smoke-test the pipeline):
    python tools/train_rl.py --iterations 1 --games 2 --search 6 --out agent/model.pth

Run on KAGGLE (real training, GPU) — see notebooks/train_rl notebook, or:
    # attach dataset kiyotah/cg-lib, enable GPU, then:
    python train_rl.py --iterations 20 --games 60 --search 16 --out /kaggle/working/model.pth

Deck rules validated by battle_start (errorType): 1 bad id / 2 >4 same name /
3 no Basic Pokémon / 4 >1 Ace Spec.
"""

from __future__ import annotations

import argparse
from collections import Counter
import math
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "agent"))
# On Kaggle, add the attached cg-lib so `import cg` works:
import glob as _glob
for _p in _glob.glob("/kaggle/input/**/cg-lib", recursive=True):
    sys.path.append(_p)

import torch

import rlnet
from rlnet import (MyModel, SparseVector, get_encoder_input, get_decoder_input,
                   eval_nn, _enum_actions)
from cg.game import battle_start, battle_select, battle_finish
from cg.api import (
    all_card_data,
    to_observation_class,
    search_begin,
    search_step,
    search_end,
)

# Default training deck = the tuned sample deck (same as the agent's).
import kaggle_environments.envs.cabt.cabt as _cabt  # noqa: E402
from search_agent import agent as frozen_v3_agent  # noqa: E402
from search_agent import _heuristic as frozen_v1_agent  # noqa: E402
SAMPLE_DECK = list(_cabt.deck)

_CARD_DATA = {card.cardId: card for card in all_card_data()}
_NAME_TO_IDS = {}
for _card_data in _CARD_DATA.values():
    _NAME_TO_IDS.setdefault(_card_data.name, []).append(_card_data.cardId)


def _remove_visible(counts, card_id, deck_ids, include_evolution=False):
    if card_id is None:
        return
    if counts[card_id] > 0:
        counts[card_id] -= 1
    if not include_evolution:
        return
    data = _CARD_DATA.get(card_id)
    evolves_from = getattr(data, "evolvesFrom", None) if data else None
    if not evolves_from:
        return
    for previous_id in _NAME_TO_IDS.get(evolves_from, []):
        if previous_id in deck_ids and counts[previous_id] > 0:
            _remove_visible(counts, previous_id, deck_ids, include_evolution=True)
            break


def _joint_own_hidden(player, deck):
    """Sample deck and prizes jointly after removing every visible own card."""
    counts = Counter(deck)
    deck_ids = set(counts)
    for card in list(player.hand or []) + list(player.discard or []):
        _remove_visible(counts, getattr(card, "id", None), deck_ids)
    for pokemon in list(player.active or []) + list(player.bench or []):
        if pokemon is None:
            continue
        _remove_visible(
            counts, getattr(pokemon, "id", None), deck_ids,
            include_evolution=True,
        )
        energies = (
            getattr(pokemon, "energies", None)
            or getattr(pokemon, "energyCards", None)
            or []
        )
        for energy in energies:
            _remove_visible(counts, getattr(energy, "id", None), deck_ids)
        for tool in getattr(pokemon, "tools", None) or []:
            _remove_visible(counts, getattr(tool, "id", None), deck_ids)

    needed = player.deckCount + len(player.prize)
    unknown = list(counts.elements())
    random.shuffle(unknown)
    if len(unknown) < needed:
        # Defensive fallback for malformed/incomplete observations. Normal games
        # should have exactly enough cards after visible zones are removed.
        unknown.extend(random.choices(deck, k=needed - len(unknown)))
    unknown = unknown[:needed]
    return unknown[:player.deckCount], unknown[player.deckCount:]


class LearnSample:
    def __init__(self, value, policy, sv_enc, sv_dec):
        self.value = value
        self.policy = policy
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec


class _Child:
    def __init__(self, select, prob):
        self.node = None
        self.select = select
        self.prob = prob


class _Node:
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


def _create_node(parent, state, your_index, your_deck, model):
    node = _Node(parent, state)
    obs = state.observation
    cur = obs.current
    if cur.result >= 0:
        node.value = 0 if cur.result == 2 else (1 if cur.result == your_index else -1)
        node.backprop(node.value)
        return node, None
    actions = _enum_actions(obs)
    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    value, policy = eval_nn(sv_enc, sv_dec, model)
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
    return node, LearnSample(value, policy, sv_enc, sv_dec)


def mcts_agent(obs_dict, your_deck, model, search_count):
    obs = to_observation_class(obs_dict)
    your_index = obs.current.yourIndex
    st = obs.current
    own_hidden_deck, own_hidden_prize = _joint_own_hidden(
        st.players[your_index], your_deck
    )
    active = st.players[1 - your_index].active
    ss = search_begin(
        obs,
        your_deck=own_hidden_deck,
        your_prize=own_hidden_prize,
        opponent_deck=[1072] * st.players[1 - your_index].deckCount,
        opponent_prize=[1] * len(st.players[1 - your_index].prize),
        opponent_hand=[1] * st.players[1 - your_index].handCount,
        opponent_active=[1072] if (len(active) > 0 and active[0] is None) else [],
    )
    root, sample = _create_node(None, ss, your_index, your_deck, model)
    for _ in range(search_count):
        cur = root
        while True:
            best, nxt = -1e18, None
            c = 0.4 * math.sqrt(cur.visit)
            for child in cur.children:
                if child.node is None:
                    v = cur.total / max(cur.visit, 1)
                    visit = 0
                else:
                    v = child.node.total / max(child.node.visit, 1)
                    visit = child.node.visit
                if cur.state.observation.current.yourIndex != your_index:
                    v = -v
                v += c * child.prob / (1 + visit)
                if v > best:
                    best, nxt = v, child
            if nxt is None:
                break
            if nxt.node is None:
                st2 = search_step(cur.state.searchId, nxt.select)
                nxt.node, _ = _create_node(cur, st2, your_index, your_deck, model)
                break
            cur = nxt.node
            if cur.state.observation.current.result >= 0:
                cur.backprop(cur.value)
                break
    max_child, max_visit = None, -1
    for child in root.children:
        if child.node is not None:
            if child.node.visit > max_visit:
                max_child, max_visit = child, child.node.visit
    if sample is not None:
        visits = [
            child.node.visit if child.node is not None else 0
            for child in root.children
        ]
        total_visits = sum(visits)
        if total_visits <= 0:
            target = 1.0 / max(len(visits), 1)
            sample.policy = [target for _ in visits]
        else:
            sample.policy = [visit / total_visits for visit in visits]
        if abs(sum(sample.policy) - 1.0) > 1e-6:
            raise RuntimeError("MCTS visit target is not normalized")
    search_end()
    sel = max_child.select if max_child is not None else [0]
    return sel, sample


class LearnInput:
    def __init__(self):
        self.index = []
        self.value = []
        self.offset = []

    def add(self, sv):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)


def random_agent_sel(obs_dict):
    obs = to_observation_class(obs_dict)
    return random.sample(range(len(obs.select.option)), obs.select.maxCount)


_FROZEN_LEAGUE = (
    ("random", random_agent_sel),
    ("v1", frozen_v1_agent),
    ("v3", frozen_v3_agent),
)


def train(
    iterations,
    games,
    search,
    out_path,
    deck,
    batch=128,
    resume=True,
    arch=None,
    replay_capacity=50000,
    league_mix=0.0,
    league_opponents=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = arch or {}
    a_dmodel = arch.get("d_model", rlnet.D_MODEL)
    a_heads = arch.get("num_heads", rlnet.NUM_HEADS)
    a_dff = arch.get("d_ff", rlnet.D_FF)
    a_nenc = arch.get("n_enc", rlnet.N_ENC)
    a_ndec = arch.get("n_dec", rlnet.N_DEC)
    selected_names = set(league_opponents or ())
    frozen_league = tuple(
        item for item in _FROZEN_LEAGUE
        if not selected_names or item[0] in selected_names
    )
    if not frozen_league:
        raise ValueError(
            f"no frozen league opponents matched {sorted(selected_names)}"
        )
    print(f"device={device} | iters={iterations} games/iter={games} search={search} | "
          f"model d={a_dmodel} heads={a_heads} ff={a_dff} enc={a_nenc} dec={a_ndec} | "
          f"league={','.join(name for name, _ in frozen_league)}")
    # Persist the architecture so inference (RLAgent) builds a matching model.
    import json
    with open(out_path + ".arch.json", "w") as f:
        json.dump({"d_model": a_dmodel, "num_heads": a_heads, "d_ff": a_dff,
                   "n_enc": a_nenc, "n_dec": a_ndec}, f)
    model = MyModel(d_model=a_dmodel, num_heads=a_heads, d_ff=a_dff,
                    n_enc=a_nenc, n_dec=a_ndec).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_enc = torch.nn.HuberLoss(delta=0.2)

    # --- checkpoint / auto-resume -------------------------------------------
    ckpt_path = out_path + ".ckpt"
    start_iter = 0
    replay = []
    if resume and os.path.exists(ckpt_path):
        try:
            try:
                # This checkpoint is written by this trainer and includes local
                # replay objects, so it is intentionally not a weights-only load.
                ck = torch.load(
                    ckpt_path, map_location=device, weights_only=False
                )
            except TypeError:
                # Compatibility with older PyTorch releases that do not expose
                # the weights_only keyword.
                ck = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            start_iter = ck["iter"] + 1
            replay = ck.get("replay", [])
            try:
                random.setstate(ck["py_rng"])
                torch.set_rng_state(ck["torch_rng"].cpu())
            except Exception:
                pass
            print(f"RESUMED from {ckpt_path}: continuing at iter {start_iter}/{iterations}", flush=True)
        except Exception as e:
            print(f"checkpoint load failed ({e}); starting fresh", flush=True)
            start_iter = 0
    if start_iter >= iterations:
        print(f"already at iter {start_iter} >= {iterations}; nothing to do.", flush=True)
        return

    for it in range(start_iter, iterations):
        # --- quick eval vs random ---
        model.eval()
        wins = losses = 0
        with torch.inference_mode():
            for gi in range(max(2, games // 5)):
                obs, sd = battle_start(deck, deck)
                if sd.errorPlayer >= 0:
                    raise SystemExit(f"Deck error type {sd.errorType}")
                me = gi % 2
                while obs["current"]["result"] < 0:
                    if obs["current"]["yourIndex"] == me:
                        sel, _ = mcts_agent(obs, deck, model, search)
                    else:
                        sel = random_agent_sel(obs)
                    obs = battle_select(sel)
                battle_finish()
                r = obs["current"]["result"]
                if r == me:
                    wins += 1
                elif r != 2:
                    losses += 1
        wr = 100 * wins // max(wins + losses, 1)
        print(f"[iter {it}] eval vs random: {wins}W-{losses}L  ({wr}%)", flush=True)

        # --- self-play data collection ---
        iteration_samples = []
        league_games = Counter()
        league_samples = 0
        model.eval()
        with torch.inference_mode():
            for game_index in range(games):
                obs, _ = battle_start(deck, deck)
                per = [[], []]
                use_league = random.random() < league_mix
                learner_seat = game_index % 2
                frozen_name, frozen_policy = frozen_league[
                    game_index % len(frozen_league)
                ]
                if use_league:
                    league_games[frozen_name] += 1
                while obs["current"]["result"] < 0:
                    current_seat = obs["current"]["yourIndex"]
                    learner_turn = not use_league or current_seat == learner_seat
                    if learner_turn:
                        sel, sample = mcts_agent(obs, deck, model, search)
                        if sample is not None:
                            per[current_seat].append(sample)
                            if use_league:
                                league_samples += 1
                    else:
                        sel = frozen_policy(obs)
                    obs = battle_select(sel)
                battle_finish()
                result = obs["current"]["result"]
                for i in range(2):
                    if use_league and i != learner_seat:
                        continue
                    value = 0.0 if result == 2 else (1.0 if i == result else -1.0)
                    for s in per[i]:
                        s.value = value
                        iteration_samples.append(s)

        if league_games:
            mix_report = ", ".join(
                f"{name}={league_games[name]}" for name, _ in frozen_league
                if league_games[name]
            )
            print(
                f"[iter {it}] league games: {mix_report}; "
                f"learner samples={league_samples}",
                flush=True,
            )

        replay.extend(iteration_samples)
        if len(replay) > replay_capacity:
            replay = replay[-replay_capacity:]
        samples = list(replay)

        # --- train ---
        if len(samples) >= batch:
            model.train()
            random.shuffle(samples)
            nb = len(samples) // batch
            for bi in range(nb):
                ie, idc = LearnInput(), LearnInput()
                mask, le, ld = [], [], []
                for s in samples[bi * batch:(bi + 1) * batch]:
                    ie.add(s.sv_enc)
                    idc.add(s.sv_dec)
                    le.append(s.value)
                    ld.extend(s.policy)
                    for _ in range(len(s.policy)):
                        mask.append(1.0)
                    for _ in range(64 - len(s.policy)):
                        mask.append(0.0)
                        ld.append(0.0)
                        idc.offset.append(len(idc.index))
                mt = torch.tensor(mask, dtype=torch.float32, device=device).view(batch, -1)
                lte = torch.tensor(le, dtype=torch.float32, device=device).view(batch, -1)
                ltd = torch.tensor(ld, dtype=torch.float32, device=device).view(batch, -1)
                opt.zero_grad()
                oe, od = model(
                    torch.tensor(ie.index, dtype=torch.int32, device=device),
                    torch.tensor(ie.value, dtype=torch.float32, device=device),
                    torch.tensor(ie.offset, dtype=torch.int32, device=device),
                    torch.tensor(idc.index, dtype=torch.int32, device=device),
                    torch.tensor(idc.value, dtype=torch.float32, device=device),
                    torch.tensor(idc.offset, dtype=torch.int32, device=device),
                )
                le = loss_enc(oe, lte)
                # MCTS visit counts are a categorical target distribution. Mask
                # padded actions, then optimize soft cross-entropy over only the
                # legal root actions. The x10 scale matches inference priors.
                logits = (od * 10.0).masked_fill(mt == 0, -1e9)
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                l_d = -(ltd * log_probs * mt).sum(dim=1).mean()
                (le + l_d).backward()
                opt.step()
            print(
                f"[iter {it}] trained on {len(samples)} replay samples "
                f"({len(iteration_samples)} new, {nb} batches)",
                flush=True,
            )
        else:
            print(f"[iter {it}] only {len(samples)} samples (<{batch}) — skipped training", flush=True)

        # Save inference weights + a resumable checkpoint (atomic via temp file).
        torch.save(model.state_dict(), out_path)
        tmp = ckpt_path + ".tmp"
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "iter": it,
            "py_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "replay": replay,
        }, tmp)
        os.replace(tmp, ckpt_path)
        print(f"[iter {it}] saved -> {out_path} (+ checkpoint)", flush=True)

    print("done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--search", type=int, default=6)
    ap.add_argument("--out", default=os.path.join(_ROOT, "agent", "model.pth"))
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    # bigger-model knobs (defaults = the small baseline net)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--n-enc", type=int, default=1)
    ap.add_argument("--n-dec", type=int, default=1)
    ap.add_argument("--replay-capacity", type=int, default=50000)
    ap.add_argument(
        "--league-mix", type=float, default=0.0,
        help="fraction of training games against frozen random/v1/v3 opponents",
    )
    ap.add_argument(
        "--league-opponents",
        default="random,v1,v3",
        help="comma-separated frozen opponents selected from random,v1,v3",
    )
    args = ap.parse_args()
    arch = {"d_model": args.d_model, "num_heads": args.num_heads, "d_ff": args.d_ff,
            "n_enc": args.n_enc, "n_dec": args.n_dec}
    train(args.iterations, args.games, args.search, args.out, SAMPLE_DECK,
          resume=not args.fresh, arch=arch,
          replay_capacity=args.replay_capacity,
          league_mix=max(0.0, min(1.0, args.league_mix)),
          league_opponents=tuple(
              value.strip()
              for value in args.league_opponents.split(",")
              if value.strip()
          ))


if __name__ == "__main__":
    main()
