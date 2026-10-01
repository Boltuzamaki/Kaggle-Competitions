"""Kaggriculture local arena - the thing that decides what gets submitted.

Design constraints that come straight from how the ladder scores:

  * The ladder rates **win / loss / tie only**. Bank margin is worthless as an
    objective, so every number here is a match score (W=1, T=0.5, L=0). Bank is
    reported only as a diagnostic.
  * The market is SHARED, so a matchup is not symmetric. Every pairing is played
    in **both seats** on the **same seed**, which also pairs the noise.
  * An agent that ERRORs is an automatic loss on the ladder. Any non-DONE status
    is a hard failure here, not a bad score.
  * Episodes are deterministic given (agents, seed), so results are **cached by
    file content hash**. Re-running after changing one agent only replays that
    agent's games.

Modes:
  gauntlet     one candidate against every opponent            (the promotion test)
  roundrobin   every agent against every other + BT ratings    (calibration)
  head2head    two agents, many seeds                          (A/B)

Usage:
  python arena.py gauntlet main.py --seeds 6
  python arena.py roundrobin --seeds 4
  python arena.py head2head main.py champion.py --seeds 8
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, ".arena_cache.json")
BUILTINS = ("starter", "random", "pass")


def engine_version():
    """Cache results per engine version: 1.32.6 changed the town-demand rules,
    so results from another version describe a different game."""
    try:
        import importlib.metadata as _m
        return _m.version("kaggle-environments")
    except Exception:
        return "unknown"


ENGINE = engine_version()


# --------------------------------------------------------------------------
# identity & cache
# --------------------------------------------------------------------------
def agent_key(a):
    """Content hash for a file agent; the name itself for a builtin."""
    if a in BUILTINS:
        return a
    with open(a, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH))
        except Exception:
            return {}
    return {}


def save_cache(c):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f)
    os.replace(tmp, CACHE_PATH)


# --------------------------------------------------------------------------
# one game
# --------------------------------------------------------------------------
def _play(task):
    a, b, seed, steps = task
    from kaggle_environments import make
    t0 = time.time()
    try:
        env = make("kaggriculture",
                   configuration={"episodeSteps": steps, "seed": seed},
                   debug=False)
        env.run([a, b])
        last = env.steps[-1]
        return {
            "r0": float(last[0].reward or 0.0), "r1": float(last[1].reward or 0.0),
            "s0": str(last[0].status), "s1": str(last[1].status),
            "secs": time.time() - t0, "err": None,
        }
    except Exception as e:
        return {"r0": 0.0, "r1": 0.0, "s0": "EXC", "s1": "EXC",
                "secs": time.time() - t0, "err": repr(e)[:300]}


def run_games(pairings, seeds, steps, procs, cache, refresh=False):
    """pairings: list of (a, b). Plays both seats on each seed. Returns records."""
    tasks, keys, records = [], [], []
    for a, b in pairings:
        ka, kb = agent_key(a), agent_key(b)
        for seed in range(seeds):
            for seat in (0, 1):
                x, y = (a, b) if seat == 0 else (b, a)
                kx, ky = (ka, kb) if seat == 0 else (kb, ka)
                ck = f"{ENGINE}|{kx}|{ky}|{seed}|{steps}"
                rec = {"a": a, "b": b, "seed": seed, "seat": seat, "ck": ck}
                if not refresh and ck in cache:
                    rec["res"] = cache[ck]
                    records.append(rec)
                else:
                    tasks.append((x, y, seed, steps))
                    keys.append(ck)
                    records.append(rec)
    if tasks:
        print(f"  playing {len(tasks)} games ({len(records)-len(tasks)} cached) "
              f"on {procs} procs...", flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=procs) as ex:
            out = list(ex.map(_play, tasks))
        for ck, res in zip(keys, out):
            cache[ck] = res
        save_cache(cache)
        print(f"  done in {time.time()-t0:.0f}s", flush=True)
    else:
        print(f"  all {len(records)} games cached", flush=True)
    for rec in records:
        if "res" not in rec:
            rec["res"] = cache[rec["ck"]]
    return records


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_for(rec, agent):
    """Match score for `agent` in this record: 1 win, 0.5 tie, 0 loss."""
    seat = rec["seat"] if agent == rec["a"] else 1 - rec["seat"]
    res = rec["res"]
    mine = res["r0"] if seat == 0 else res["r1"]
    theirs = res["r1"] if seat == 0 else res["r0"]
    if mine > theirs:
        return 1.0
    if mine < theirs:
        return 0.0
    return 0.5


def bank_for(rec, agent):
    seat = rec["seat"] if agent == rec["a"] else 1 - rec["seat"]
    return rec["res"]["r0"] if seat == 0 else rec["res"]["r1"]


def is_clean(rec):
    r = rec["res"]
    return r["err"] is None and r["s0"] == "DONE" and r["s1"] == "DONE"


def wilson(w, n, z=1.96):
    """Wilson score interval for a proportion - honest at small n."""
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def bradley_terry(pairs, iters=400):
    """MM algorithm. pairs: {(i,j): (wins_i, wins_j)}. Returns Elo-ish ratings."""
    players = sorted({p for pair in pairs for p in pair})
    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    s = [1.0] * n
    for _ in range(iters):
        new = [0.0] * n
        den = [0.0] * n
        for (a, b), (wa, wb) in pairs.items():
            ia, ib = idx[a], idx[b]
            tot = wa + wb
            if tot == 0:
                continue
            new[ia] += wa
            new[ib] += wb
            den[ia] += tot / (s[ia] + s[ib])
            den[ib] += tot / (s[ia] + s[ib])
        for i in range(n):
            if den[i] > 0:
                s[i] = max(new[i] / den[i], 1e-9)
        g = statistics.geometric_mean([x for x in s if x > 0]) or 1.0
        s = [x / g for x in s]
    return {p: 1500 + 400 * math.log10(s[idx[p]]) for p in players}


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
def report_gauntlet(candidate, opponents, records):
    print(f"\n{'='*84}")
    print(f"  GAUNTLET - {candidate}")
    print(f"{'='*84}")
    print(f"  {'opponent':<24}{'score':>8}{'W-L-T':>12}{'95% CI':>16}"
          f"{'mean bank':>13}{'opp bank':>13}")
    print("  " + "-" * 82)
    total_s, total_n, dirty = 0.0, 0, []
    rows = []
    for opp in opponents:
        recs = [r for r in records if r["b"] == opp or r["a"] == opp]
        recs = [r for r in recs if candidate in (r["a"], r["b"])]
        if not recs:
            continue
        w = sum(1 for r in recs if score_for(r, candidate) == 1.0)
        l = sum(1 for r in recs if score_for(r, candidate) == 0.0)
        t = len(recs) - w - l
        s = sum(score_for(r, candidate) for r in recs)
        lo, hi = wilson(s, len(recs))
        mb = statistics.mean([bank_for(r, candidate) for r in recs])
        ob = statistics.mean([bank_for(r, opp) for r in recs])
        bad = [r for r in recs if not is_clean(r)]
        dirty += bad
        flag = "  !! ERRORS" if bad else ""
        print(f"  {opp:<24}{s/len(recs):>8.2f}{f'{w}-{l}-{t}':>12}"
              f"{f'[{lo:.2f},{hi:.2f}]':>16}{mb:>13,.0f}{ob:>13,.0f}{flag}")
        rows.append((opp, s / len(recs), w, l, t, mb, ob))
        total_s += s
        total_n += len(recs)
    overall = total_s / max(1, total_n)
    lo, hi = wilson(total_s, total_n)
    print("  " + "-" * 82)
    print(f"  {'OVERALL':<24}{overall:>8.2f}{'':>12}{f'[{lo:.2f},{hi:.2f}]':>16}"
          f"   over {total_n} games")
    if dirty:
        print(f"\n  !! {len(dirty)} NON-CLEAN GAMES (an ERROR is a ladder loss):")
        for r in dirty[:5]:
            print(f"     {r['a']} vs {r['b']} seed={r['seed']} seat={r['seat']} "
                  f"{r['res']['s0']}/{r['res']['s1']} {r['res']['err'] or ''}")
    return {"overall": overall, "n": total_n, "ci": [lo, hi],
            "clean": not dirty, "rows": rows}


def report_roundrobin(agents, records):
    pairs = {}
    for r in records:
        a, b = r["a"], r["b"]
        key = (a, b) if a < b else (b, a)
        sa = score_for(r, key[0])
        wa, wb = pairs.get(key, (0.0, 0.0))
        pairs[key] = (wa + sa, wb + (1.0 - sa))
    bt = bradley_terry(pairs)
    tot = {a: [0.0, 0] for a in agents}
    for r in records:
        for who in (r["a"], r["b"]):
            tot[who][0] += score_for(r, who)
            tot[who][1] += 1
    print(f"\n{'='*74}")
    print("  ROUND ROBIN - Bradley-Terry ratings")
    print(f"{'='*74}")
    print(f"  {'#':>3}  {'agent':<26}{'BT':>9}{'score':>9}{'games':>8}{'mean bank':>14}")
    print("  " + "-" * 70)
    order = sorted(agents, key=lambda a: -bt.get(a, 0))
    for i, a in enumerate(order, 1):
        s, n = tot[a]
        banks = [bank_for(r, a) for r in records if a in (r["a"], r["b"])]
        mb = statistics.mean(banks) if banks else 0
        print(f"  {i:>3}  {a:<26}{bt.get(a,0):>9.0f}{s/max(1,n):>9.2f}"
              f"{n:>8}{mb:>14,.0f}")
    return {a: bt.get(a, 0) for a in agents}


# --------------------------------------------------------------------------
def gauntlet_agents(extra_builtins=True):
    d = os.path.join(HERE, "gauntlet")
    out = []
    if os.path.isdir(d):
        out = [os.path.join("gauntlet", f) for f in sorted(os.listdir(d))
               if f.endswith(".py")]
    if extra_builtins:
        out.append("starter")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gauntlet", "roundrobin", "head2head"])
    ap.add_argument("agents", nargs="*")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=720)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--refresh", action="store_true", help="ignore cached results")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cache = load_cache()

    if args.mode == "gauntlet":
        cand = args.agents[0] if args.agents else "main.py"
        opps = args.agents[1:] or gauntlet_agents()
        recs = run_games([(cand, o) for o in opps], args.seeds, args.steps,
                         args.procs, cache, args.refresh)
        out = report_gauntlet(cand, opps, recs)
    elif args.mode == "head2head":
        a, b = args.agents[0], args.agents[1]
        recs = run_games([(a, b)], args.seeds, args.steps, args.procs, cache,
                         args.refresh)
        out = report_gauntlet(a, [b], recs)
    else:
        agents = args.agents or (["main.py"] + gauntlet_agents())
        recs = run_games(list(itertools.combinations(agents, 2)), args.seeds,
                         args.steps, args.procs, cache, args.refresh)
        out = report_roundrobin(agents, recs)

    if args.json:
        json.dump(out, open(args.json, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
