"""One-card coordinate screen around the public Grim list.

The policy is held exact.  Each arm makes one legal count change, avoiding the
high-dimensional/deck-policy confound of wholesale deck evolution.
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

GRIM_DIR = os.environ.get("GDB_BASE_DIR", os.path.join(
    ROOT, "references", "top_rankers", "grim_control", "extracted"
))
MAIN = os.path.join(GRIM_DIR, "main.py")
BASE = [int(x) for x in open(os.path.join(GRIM_DIR, "deck.csv")) if x.strip()]
# Card IDs: energy 7, Fros 104, Munki 112, Imp 646, Morg 647, Grim 648,
# Snorunt 860, Candy 1079, Pokegear 1122, Scrapper 1137, Boss 1182,
# Petrel 1219, Lillie 1227, Dawn 1231, Gym 1259.
SWAPS = {
    "candy_scrap": (1079, 1137),
    "scrap_gear": (1137, 1122), "scrap_energy": (1137, 7),
    "scrap_boss": (1137, 1182), "scrap_candy": (1137, 1079),
    "dawn_gear": (1231, 1122), "dawn_energy": (1231, 7),
    "dawn_boss": (1231, 1182), "dawn_candy": (1231, 1079),
    "petrel_energy": (1219, 7), "petrel_gear": (1219, 1122),
    "petrel_boss": (1219, 1182), "petrel_stretcher": (1219, 1097),
    "gym_energy": (1259, 7), "gym_gear": (1259, 1122),
    "gym_boss": (1259, 1182), "gym_fros": (1259, 104),
    "gym_snorunt": (1259, 860), "gym_morgrem": (1259, 647),
    "gym_grim": (1259, 648),
    # Tech coordinates observed in exact >=900-rated Grim replay lists.
    "gear_fan": (1122, 1161), "gear_xero": (1122, 1197),
    "gear_judge": (1122, 1213), "gear_helmet": (1122, 1156),
    "gear_balloon": (1122, 1174), "gear_lumiose": (1122, 1267),
    "candy_fan": (1079, 1161), "candy_xero": (1079, 1197),
    "morgrem_judge": (647, 1213), "stretcher_gear": (1097, 1122),
}
OPPONENTS = tuple(os.environ.get(
    "GDB_OPPS", "public-archaludon,public-alakazam,meta-grimmsnarl"
).split(","))


def mutated(name):
    old, new = SWAPS[name]
    deck = list(BASE); deck.remove(old); deck.append(new); deck.sort()
    return deck


def load(name):
    spec = ilu.spec_from_file_location(name, MAIN)
    module = ilu.module_from_spec(spec); spec.loader.exec_module(module); return module


def chunk(args):
    arm, seeds = args
    if GRIM_DIR not in sys.path: sys.path.insert(0, GRIM_DIR)
    candidate = load(f"grim_deck_candidate_{os.getpid()}")
    control = load(f"grim_deck_control_{os.getpid()}")
    deck = mutated(arm)
    aw = bw = ao = bo = 0
    by_opp = {}
    for opponent_name in OPPONENTS:
        opponent, opponent_deck = build(opponent_name)
        ca = cb = 0
        for seed in seeds:
            counts=[]
            for module, own_deck in ((candidate, deck), (control, BASE)):
                wins=0
                for seat in (0,1):
                    module._reset(); random.seed(seed)
                    if seat==0: wins += int(play(seed,module.agent,opponent,own_deck,opponent_deck)==0)
                    else: wins += int(play(seed,opponent,module.agent,opponent_deck,own_deck)==1)
                counts.append(wins)
            aw += counts[0]; bw += counts[1]; ca += counts[0]; cb += counts[1]
            ao += int(counts[0]>counts[1]); bo += int(counts[1]>counts[0])
        by_opp[opponent_name]=(ca,cb)
    return arm,aw,bw,ao,bo,by_opp


def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 20
    workers=int(sys.argv[2]) if len(sys.argv)>2 else 12
    seed0=int(os.environ.get("GDB_SEED0","6500000")); seeds=[seed0+i for i in range(n)]
    arms=os.environ.get("GDB_ARMS",",").strip(",").split(",") if os.environ.get("GDB_ARMS") else list(SWAPS)
    chunks=int(os.environ.get("GDB_CHUNKS","1"))
    jobs=[(arm,seeds[i::chunks]) for arm in arms for i in range(chunks)]
    agg={arm:[0,0,0,0,{name:[0,0] for name in OPPONENTS}] for arm in arms}
    with mp.get_context("fork").Pool(min(workers,len(jobs))) as pool:
        for arm,aw,bw,ao,bo,by in pool.imap_unordered(chunk,jobs):
            row=agg[arm]
            for i,x in enumerate((aw,bw,ao,bo)): row[i]+=x
            for name,(a,b) in by.items(): row[4][name][0]+=a; row[4][name][1]+=b
    for arm in arms:
        aw,bw,ao,bo,by=agg[arm]
        detail=" ".join(f"{k.split('-')[-1]}:{a}/{b}" for k,(a,b) in by.items())
        print(f"{arm:16s} {aw:3d}/{bw:3d} disc {ao:2d}/{bo:2d} p={mcnemar(ao,bo):.4f} {detail}",flush=True)


if __name__=="__main__": main()
