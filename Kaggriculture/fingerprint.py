"""Extract the STRATEGIC DECISIONS from an agent's play, not its actions.

Copying a 719-turn tape gives you its moves. Reading the decisions out of it
gives you the reasoning, which transfers to an agent that can still adapt.

For each day this records what a human strategist would want to know:
  capital, herd composition, crop mix, land, labour hired, and what was sold.

Usage:
  python fingerprint.py gauntlet/tetsutani.py main.py --seed 3
"""
import argparse
import collections
import json
import sys

from kaggle_environments import make


def fingerprint(agent, opponent, seed, seat=0, steps=720):
    agents = [agent, opponent] if seat == 0 else [opponent, agent]
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed},
               debug=False)
    env.run(agents)

    days = {}
    hires = collections.Counter()
    sold = collections.Counter()
    bought = collections.Counter()
    ops = collections.Counter()
    sell_day = collections.defaultdict(collections.Counter)

    for st in env.steps:
        obs = st[0].observation
        day, hour = obs["day"], obs["hour"]
        a = st[seat].action
        if isinstance(a, dict):
            f = a.get("farmer") or ["PASS"]
            if isinstance(f, list) and f:
                ops[f[0]] += 1
            for h in (a.get("hands") or []):
                if isinstance(h, list) and h:
                    ops[h[0]] += 1
            for m in (a.get("market") or []):
                if not (isinstance(m, list) and m):
                    continue
                if m[0] == "HIRE":
                    hires[day] += 1
                elif m[0] == "SELL" and len(m) >= 3:
                    sold[m[1]] += int(m[2])
                    sell_day[day][m[1]] += int(m[2])
                elif m[0] in ("BUY_ANIMAL", "BUY_SEED", "BUY_PRODUCT") and len(m) >= 3:
                    bought[m[1]] += int(m[2])
                elif m[0] == "BUY_LAND":
                    bought["LAND"] += 1
        if hour != 0 or day in days:
            continue
        farm = obs["farms"][seat]
        c = collections.Counter()
        free = weeds = 0
        for row in farm["tiles"]:
            for t in row:
                if t is None:
                    free += 1
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "WEED":
                        weeds += 1
                    elif k == "PLANT":
                        c[t["crop"]] += 1
                    elif "animal" in t:
                        c[t["animal"]] += 1
                    else:
                        c[k] += 1
        days[day] = dict(money=farm["money"], quads=len(farm["unlocked_quadrants"]),
                         free=free, weeds=weeds, **{k: c[k] for k in
                         ("COW", "SHEEP", "GOOSE", "MELON", "STRAWBERRY", "TOMATO",
                          "WHEAT", "CARROT", "COOP", "PASTURE")})
    final = env.steps[-1]
    return dict(days=days, hires=hires, sold=sold, bought=bought, ops=ops,
                sell_day=sell_day,
                final=float(final[seat].reward or 0),
                opp_final=float(final[1 - seat].reward or 0))


def show(name, fp):
    print(f"\n{'='*96}")
    print(f"  {name}   final bank ${fp['final']:,.0f}  (opponent ${fp['opp_final']:,.0f})")
    print(f"{'='*96}")
    print(f"{'day':>4}{'money':>10}{'hire':>5}{'cow':>4}{'shp':>4}{'gse':>4}"
          f"{'mel':>4}{'str':>4}{'tom':>4}{'wht':>4}{'car':>4}"
          f"{'past':>5}{'coop':>5}{'free':>5}{'wd':>4}{'q':>3}   top sales that day")
    for d in sorted(fp["days"]):
        r = fp["days"][d]
        s = fp["sell_day"].get(d, {})
        top = " ".join(f"{k[:3]}:{v}" for k, v in
                       sorted(s.items(), key=lambda kv: -kv[1])[:4])
        print(f"{d:>4}{r['money']:>10,.0f}{fp['hires'][d]:>5}"
              f"{r['COW']:>4}{r['SHEEP']:>4}{r['GOOSE']:>4}"
              f"{r['MELON']:>4}{r['STRAWBERRY']:>4}{r['TOMATO']:>4}"
              f"{r['WHEAT']:>4}{r['CARROT']:>4}"
              f"{r['PASTURE']:>5}{r['COOP']:>5}{r['free']:>5}{r['weeds']:>4}"
              f"{r['quads']:>3}   {top}")
    tot = sum(fp["ops"].values())
    move = sum(fp["ops"][k] for k in ("NORTH", "SOUTH", "EAST", "WEST"))
    print(f"\n  actions {tot}   move {100*move/max(1,tot):.0f}%   "
          f"PASS {100*fp['ops']['PASS']/max(1,tot):.0f}%   "
          f"hires {sum(fp['hires'].values())}")
    print("  sold   : " + ", ".join(f"{k} {v}" for k, v in fp["sold"].most_common()))
    print("  bought : " + ", ".join(f"{k} {v}" for k, v in bought_items(fp)))


def bought_items(fp):
    return fp["bought"].most_common()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agents", nargs="+")
    ap.add_argument("--opponent", default="starter")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--seat", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}
    for a in args.agents:
        fp = fingerprint(a, args.opponent, args.seed, args.seat)
        show(a, fp)
        out[a] = {k: (dict(v) if isinstance(v, collections.Counter) else v)
                  for k, v in fp.items() if k != "sell_day"}
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
