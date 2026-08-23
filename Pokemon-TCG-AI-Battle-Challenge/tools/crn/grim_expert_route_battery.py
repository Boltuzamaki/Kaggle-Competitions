"""Counterfactual expert-routing screen for the public Grim policy.

This is hypothesis generation: compare one forced expert with the published
router on deterministic upper-meta opponents.  Any winner must later be gated
from visible cards and confirmed on fresh seeds before becoming our residual.
"""
from __future__ import annotations

import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for path in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

from paired_eval import build, mcnemar, play  # noqa: E402
import domain_policy  # noqa: E402
import top_decks  # noqa: E402

GRIM_DIR = os.environ.get("GER_BASE_DIR", os.path.join(
    ROOT, "references", "top_rankers", "grim_control", "extracted"
))
MAIN = os.path.join(GRIM_DIR, "main.py")
DECK = [int(x) for x in open(os.path.join(GRIM_DIR, "deck.csv")) if x.strip()]
PANELS = {
    "upper": {
        "grimmsnarl": top_decks.TD_01,
        "alakazam": top_decks.TD_02,
        "froslass": top_decks.TD_04,
        "festival": top_decks.TD_10,
    },
    "confirm": {
        "dragapult": top_decks.TD_03,
        "froslass_alt": top_decks.TD_09,
        "ogerpon": top_decks.TD_00,
        "crustle": top_decks.TD_07,
    },
}


def load(name):
    spec = ilu.spec_from_file_location(name, MAIN)
    module = ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(args):
    panel, seeds, expert = args
    if GRIM_DIR not in sys.path:
        sys.path.insert(0, GRIM_DIR)
    candidate = load(f"grim_candidate_{os.getpid()}")
    control = load(f"grim_control_{os.getpid()}")
    if expert == "tempo":
        candidate.route_choice = lambda obs, baseline, mirror, tempo: tempo
    elif expert == "mirror":
        candidate.route_choice = lambda obs, baseline, mirror, tempo: mirror
    elif expert == "baseline":
        candidate.route_choice = lambda obs, baseline, mirror, tempo: baseline
    elif expert == "no_human":
        candidate.human_choose=lambda obs, choices: None
    elif expert.startswith("no_"):
        layer = expert[3:] + "_choose"
        if layer not in {"advisor_choose", "residual_choose", "tactical_choose", "development_choose", "robustness_choose"}:
            raise ValueError(expert)
        setattr(candidate, layer, lambda *args, **kwargs: None)
    elif expert.startswith("human_"):
        key=expert[6:]
        if key not in {"model","strategic","mirror","tempo","coalition","baseline_route"}:
            raise ValueError(expert)
        candidate.human_choose=lambda obs, choices, chosen=key: choices.get(chosen)
    elif expert in {
        "gate_arch_mirror","gate_arch_tempo","gate_arch_mirror_main",
        "gate_arch_mirror_search","gate_arch_mirror_attach",
        "gate_arch_mirror_damage","gate_arch_mirror_other",
        "gate_arch_mirror_no_main","gate_arch_mirror_no_search",
        "gate_arch_mirror_no_attach","gate_arch_mirror_no_damage",
        "gate_arch_mirror_no_other",
    }:
        original=candidate.human_choose
        chosen="tempo" if expert=="gate_arch_tempo" else "mirror"
        group=expert.removeprefix("gate_arch_mirror_") if expert.startswith("gate_arch_mirror_") else "all"
        groups={
            "all":None,"main":{0},"search":{5,7},"attach":{21,22},
            "damage":{13,14,15,16},
        }
        route_globals=candidate.route_choice.__globals__
        def gated(obs,choices,original=original,chosen=chosen,route_globals=route_globals,group=group,groups=groups):
            state=route_globals.get("S") or {}
            value=(obs.get("select") or {}).get("context")
            context=int(value if value is not None else -1)
            negate=group.startswith("no_")
            base_group=group[3:] if negate else group
            wanted=groups.get(base_group)
            context_ok=(wanted is None or context in wanted)
            known=set().union(*[x for x in groups.values() if x])
            if base_group=="other":context_ok=context not in known
            if negate:context_ok=not context_ok
            if state.get("profile")=="arch" and state.get("locked") and context_ok:
                return choices.get(chosen)
            return original(obs,choices)
        candidate.human_choose=gated
    elif expert == "exact":
        pass
    elif expert.startswith("gate_psychic_no_"):
        short=expert.removeprefix("gate_psychic_no_")
        layer=short+"_choose"
        if layer not in {"advisor_choose","residual_choose","tactical_choose","development_choose","robustness_choose"}:
            raise ValueError(expert)
        original_layer=getattr(candidate,layer)
        route_globals=candidate.route_choice.__globals__
        def psychic_layer(obs,*args,original=original_layer,route_globals=route_globals,**kwargs):
            state=route_globals.get("S") or {}
            if state.get("profile")=="psychic" and state.get("locked"):
                return None
            return original(obs,*args,**kwargs)
        setattr(candidate,layer,psychic_layer)
    else:
        raise ValueError(expert)

    # Optional promoted-control comparison for refinements of the full Arch
    # gate. This makes A/B differ only in the candidate's context mask.
    if os.environ.get("GER_CONTROL") == "gate_arch_mirror":
        original_control=control.human_choose
        control_globals=control.route_choice.__globals__
        def control_gate(obs,choices,original=original_control,route_globals=control_globals):
            state=route_globals.get("S") or {}
            if state.get("profile")=="arch" and state.get("locked"):
                return choices.get("mirror")
            return original(obs,choices)
        control.human_choose=control_gate

    rows = {}
    if panel == "public":
        opponents = {}
        public_names=tuple(os.environ.get(
            "GER_OPPS","public-archaludon,public-alakazam,meta-grimmsnarl"
        ).split(","))
        for name in public_names:
            opponents[name] = build(name)
    else:
        opponents = {}
        for name, deck in PANELS[panel].items():
            def opponent(obs, frozen=deck):
                return list(frozen) if obs.get("select") is None else domain_policy.domain_agent(obs, frozen)
            opponents[name] = (opponent, deck)

    for opponent_name, (opponent, opponent_deck) in opponents.items():

        cw = bw = co = bo = 0
        for seed in seeds:
            counts = []
            for policy in (candidate, control):
                wins = 0
                for seat in (0, 1):
                    policy._reset()
                    random.seed(seed)
                    if seat == 0:
                        result = play(seed, policy.agent, opponent, DECK, opponent_deck)
                        wins += int(result == 0)
                    else:
                        result = play(seed, opponent, policy.agent, opponent_deck, DECK)
                        wins += int(result == 1)
                counts.append(wins)
            cw += counts[0]; bw += counts[1]
            co += int(counts[0] > counts[1]); bo += int(counts[1] > counts[0])
        rows[opponent_name] = (cw, bw, co, bo)
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    expert = os.environ.get("GER_EXPERT", "tempo")
    panel = os.environ.get("GER_PANEL", "upper")
    seed0 = int(os.environ.get("GER_SEED0", "6480000"))
    seeds = [seed0 + i for i in range(n)]
    jobs = [(panel, seeds[i::workers], expert) for i in range(workers)]
    names = tuple(os.environ.get(
        "GER_OPPS","public-archaludon,public-alakazam,meta-grimmsnarl"
    ).split(",")) if panel == "public" else PANELS[panel]
    aggregate = {name: [0, 0, 0, 0] for name in names}
    with mp.get_context("fork").Pool(workers) as pool:
        for result in pool.imap_unordered(chunk, jobs):
            for name, row in result.items():
                aggregate[name] = [a + b for a, b in zip(aggregate[name], row)]
    pooled = [0, 0, 0, 0]
    for name, row in aggregate.items():
        pooled = [a + b for a, b in zip(pooled, row)]
        cw, bw, co, bo = row
        print(f"{name:14s} {expert} {cw:3d} exact {bw:3d} disc {co:2d}/{bo:2d} p={mcnemar(co,bo):.5f}")
    cw, bw, co, bo = pooled
    print(f"POOLED         {expert} {cw:3d} exact {bw:3d} disc {co:2d}/{bo:2d} p={mcnemar(co,bo):.6f}")


if __name__ == "__main__":
    main()
