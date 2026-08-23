"""Battery over leaf-evaluator variants, on the verified-deterministic panel.

Same harness as clean_battery (stock-vs-stock `control` arm must come back at or
near 0/0, and it is the noise floor every other arm is judged against), but the
arms vary `_leaf_eval` rather than the deck or the search depth.

Motivation: depth2 and depth4 both came back flat with ~51 discordant games out
of 600 against a 9-game noise floor. The search reaches genuinely different
positions and cannot tell which are better, which points at the evaluator.

Each arm adds ONE term so an effect is attributable:
    hand    card advantage (absent from the fork's evaluator entirely)
    deck    remaining deck size / deck-out pressure
    stage   Abra < Kadabra < Alakazam evolution progress
    bench   board width
    frachp  health as fraction-of-max instead of raw HP totals
    all     every additive term together
"""
from __future__ import annotations

import importlib.util as ilu
import multiprocessing as mp
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, HERE)

from paired_eval import play, build, mcnemar  # noqa: E402
import deep_search  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
DECK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]
# Panel is env-overridable so a confirmation run can use DIFFERENT opponents
# as well as disjoint seeds -- one panel is never a verdict.
PANEL = [n for n in os.environ.get(
    "MB_PANEL", "public-archaludon,meta-grimmsnarl,public-alakazam").split(",") if n]
ARMS = [a for a in os.environ.get(
    "MB_ARMS", "control,opts40,opts60,optsinf").split(",") if a]


def _mk(tag, variant):
    s = ilu.spec_from_file_location("fm_" + tag, FP)
    m = ilu.module_from_spec(s)
    s.loader.exec_module(m)
    m._TEMPLATES = []
    m._TEMPLATE_SIG = []
    b = os.environ.get("CB_BUDGET")
    if b:
        m.TIME_BUDGET_S = float(b)
    # The smoke test showed every discordant seed favouring stock when the
    # margin was LOWERED, i.e. the search's overrides are worse than the
    # heuristic and 500.0 acts as a safety valve. So probe upward too, and
    # include switching the search off entirely as the limiting case.
    # SEARCH_MAX_OPTS gates the search OFF when a decision has too many options.
    # Measured: 11 of 130 MAIN decisions (8.5%) are skipped for having >24 options,
    # and those are the most complex positions in the game. With margin 3000 now
    # filtering out the search's bad overrides, extending coverage to them may pay
    # where it would not have at the stock margin. `cap8` additionally widens the
    # candidate cap, which is a second gate on the same decisions.
    _CAPS = {"control": (24, 8), "opts40": (40, 8), "opts60": (60, 8),
             "optsinf": (9999, 8), "opts60c12": (60, 12)}
    mo, cc = _CAPS.get(variant, (24, 8))
    m.SEARCH_MAX_OPTS = mo
    deep_search.install(m, extra_turns=0, margin=3000.0)

    def w(o, _m=m):
        return list(DECK) if o.get("select") is None else _m.agent(o)
    return w


def chunk(args):
    name, seeds = args
    import arena  # noqa: F401
    cand = _mk(name, name)
    base = _mk("base_" + name, "control")
    aw = bw = a = b = 0
    for opp in PANEL:
        fo, do = build(opp)
        for s in seeds:
            x = y = 0
            for seat in (0, 1):
                random.seed(s)
                x += int(play(s, cand, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(s, fo, cand, do, DECK) == 1)
            for seat in (0, 1):
                random.seed(s)
                y += int(play(s, base, fo, DECK, do) == 0) if seat == 0 else \
                     int(play(s, fo, base, do, DECK) == 1)
            aw += x
            bw += y
            if x > y:
                a += 1
            elif y > x:
                b += 1
    return name, aw, bw, a, b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seed0 = int(os.environ.get("MB_SEED0", "338000"))
    chunks = int(os.environ.get("MB_CHUNKS", "3"))
    seeds = [seed0 + i for i in range(n)]
    jobs = []
    for name in ARMS:
        for i in range(chunks):
            jobs.append((name, seeds[i::chunks]))
    print(f"{len(ARMS)} arms x {n} seeds x {len(PANEL)} opponents; "
          f"{len(jobs)} jobs on {workers} workers", flush=True)

    agg = {k: [0, 0, 0, 0] for k in ARMS}
    with mp.get_context("fork").Pool(workers) as pool:
        for name, aw, bw, a, b in pool.imap_unordered(chunk, jobs):
            g = agg[name]
            g[0] += aw; g[1] += bw; g[2] += a; g[3] += b
            print(f"    [partial] {name}: {g[0]} vs {g[1]}  disc {g[2]}/{g[3]}", flush=True)

    print("\n===== SEARCH OPTION-CAP RESULTS =====", flush=True)
    floor = agg["control"][2] + agg["control"][3]
    for name in ARMS:
        aw, bw, a, b = agg[name]
        p = mcnemar(a, b)
        if name == "control":
            print(f"  {name:8s} {aw:4d} vs {bw:4d}  disc {a:3d}/{b:3d}  "
                  f"(noise floor = {floor} discordant)")
            continue
        tag = "BETTER" if (a > b and p < 0.05) else ("worse" if b > a else "flat")
        print(f"  {name:8s} {aw:4d} vs {bw:4d}  disc {a:3d}/{b:3d}  p={p:.4f}  {tag}")


if __name__ == "__main__":
    main()
