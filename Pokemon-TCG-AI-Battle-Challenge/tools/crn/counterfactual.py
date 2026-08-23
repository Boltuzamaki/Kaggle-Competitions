"""Find the exact decisions where two agents diverge on IDENTICAL games.

Discussion 731298 proposes analysing LB replays with an LLM to extract strategy
and synthesise rules. The naive form of that has already been measured and it
BACKFIRED: discussion 713608 analysed 40 losing games, derived three sensible
fixes, and shipped a pooled -7.6pp regression (p=0.003). Their diagnosis was
survivorship bias -- reading only losses makes "stay in at 30 HP and attack" look
wrong, because the losing traces hide every game won by grabbing a prize first.

This is the counterfactual version, which is only possible because we have common
random numbers (the same team noted they did not: "if you have CRN working, we'd
love to hear how").

Method: play the same seed with agent A and agent B. Keep only DISCORDANT seeds --
one won, the other lost -- with identical shuffles, draws and coin flips. Then
replay both and record the FIRST decision where they chose differently, with the
board state at that point. That decision is a causal candidate for the outcome
difference, not a story fitted to a loss.

Output is a set of divergence cases for analysis. Any rule derived from them still
has to clear paired CRN at p<0.05 before shipping -- the step 713608 skipped.
"""
from __future__ import annotations

import ctypes
import json
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "agent"))

from paired_eval import build, _lib  # noqa: E402

try:
    from cg.api import all_card_data, OptionType, SelectContext
    _C = {c.cardId: c.name for c in all_card_data()}
except Exception:
    _C = {}


def _name(cid):
    return _C.get(cid, f"#{cid}")


def run_traced(seed, agents, decks, me_seat):
    """Play one game, recording (step, options, chosen) for our seat."""
    cards = (ctypes.c_int * 120)(*(list(decks[0]) + list(decks[1])))
    st = _lib.CrnBattleStart(cards, ctypes.c_uint(seed & 0xFFFFFFFF or 1))
    if st.errorPlayer >= 0:
        return None, []
    ptr = st.battlePtr
    trace = []
    try:
        for step in range(6000):
            sd = _lib.GetBattleData(ptr)
            if not sd.json:
                return None, trace
            obs = json.loads(ctypes.string_at(sd.json).decode("utf-8", "replace"))
            obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode("ascii")
            cur = obs.get("current") or {}
            if cur.get("result", -1) >= 0:
                return cur["result"], trace
            sel = obs.get("select")
            if sel is None:
                return None, trace
            n = len(sel.get("option") or [])
            if n == 0:
                return None, trace
            who = sd.selectPlayer if sd.selectPlayer in (0, 1) else 0
            try:
                ch = agents[who](obs)
            except Exception:
                ch = [0]
            ch = [c for c in (ch or [0]) if isinstance(c, int) and 0 <= c < n] or [0]
            if who == me_seat and n > 1:
                trace.append({"step": step, "ctx": sel.get("context"),
                              "n": n, "pick": ch[0],
                              "options": sel.get("option"), "obs": obs})
            arr = (ctypes.c_int * len(ch))(*ch)
            if _lib.Select(ptr, arr, len(ch)) != 0:
                return 1 - who, trace
        return None, trace
    finally:
        _lib.BattleFinish(ptr)


def describe(opt):
    t = opt.get("type")
    try:
        tn = OptionType(t).name
    except Exception:
        tn = str(t)
    cid = opt.get("cardId") or 0
    bits = [tn]
    if cid:
        bits.append(_name(cid))
    if opt.get("attackId"):
        bits.append(f"atk{opt['attackId']}")
    return " ".join(bits)


def main():
    import arena  # noqa: F401
    a_name, b_name = sys.argv[1], sys.argv[2]
    opp_name = sys.argv[3] if len(sys.argv) > 3 else "public-archaludon"
    nseeds = int(sys.argv[4]) if len(sys.argv) > 4 else 40

    fa, da = build(a_name)
    fb, db = build(b_name)
    fo, do = build(opp_name)
    if list(da) != list(db):
        print("NOTE: decks differ; divergence will conflate policy and deck.")

    cases = []
    stats = Counter()
    for s in [50000 + i for i in range(nseeds)]:
        random.seed(s)
        ra, ta = run_traced(s, (fa, fo), (da, do), 0)
        random.seed(s)
        rb, tb = run_traced(s, (fb, fo), (db, do), 0)
        a_won, b_won = (ra == 0), (rb == 0)
        if a_won == b_won:
            stats["concordant"] += 1
            continue
        stats["discordant"] += 1
        # first decision where the two agents chose differently
        for x, y in zip(ta, tb):
            if x["pick"] != y["pick"]:
                cases.append({"seed": s, "a_won": a_won, "step": x["step"],
                              "ctx": x["ctx"], "n": x["n"],
                              "a_pick": describe(x["options"][x["pick"]]),
                              "b_pick": describe(y["options"][y["pick"]]),
                              "turn": (x["obs"].get("current") or {}).get("turn")})
                break

    print(f"{stats['discordant']} discordant / {stats['discordant']+stats['concordant']} seeds")
    print(f"{len(cases)} first-divergence points found\n")

    by_ctx = Counter()
    for c in cases:
        try:
            cn = SelectContext(c["ctx"]).name
        except Exception:
            cn = str(c["ctx"])
        by_ctx[(cn, c["a_won"])] += 1
    print("divergence context -> outcome for A:")
    for (cn, won), k in by_ctx.most_common(15):
        print(f"  {cn:26s} A {'WON ' if won else 'LOST'}  {k}")

    print("\nsample cases:")
    for c in cases[:12]:
        try:
            cn = SelectContext(c["ctx"]).name
        except Exception:
            cn = str(c["ctx"])
        print(f"  seed {c['seed']} turn {c['turn']} {cn:20s} "
              f"A({'win' if c['a_won'] else 'loss'})={c['a_pick'][:34]:34s} B={c['b_pick'][:34]}")

    out = os.path.join(ROOT, "scratchpad", "scrape_20260804", "counterfactuals.json")
    json.dump(cases, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
