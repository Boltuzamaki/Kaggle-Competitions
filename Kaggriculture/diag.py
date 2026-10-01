"""Run one episode and dump a daily fingerprint of our agent, to find leaks."""
import sys
import collections
from kaggle_environments import make

AGENT = sys.argv[1] if len(sys.argv) > 1 else "main.py"
OPP = sys.argv[2] if len(sys.argv) > 2 else "starter"
SEAT = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0

agents = [AGENT, OPP] if SEAT == 0 else [OPP, AGENT]
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": SEED}, debug=True)
env.run(agents)

ops = collections.Counter()
mops = collections.Counter()
sold = collections.Counter()
bought = collections.Counter()
hires = 0

for st in env.steps:
    a = st[SEAT].action
    if not isinstance(a, dict):
        continue
    f = a.get("farmer") or ["PASS"]
    ops[f[0]] += 1
    for h in (a.get("hands") or []):
        if isinstance(h, list) and h:
            ops[h[0]] += 1
    for m in (a.get("market") or []):
        if isinstance(m, list) and m:
            mops[m[0]] += 1
            if m[0] == "SELL":
                sold[m[1]] += int(m[2])
            elif m[0] in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL"):
                bought[m[1]] += int(m[2])
            elif m[0] == "HIRE":
                hires += 1

print("=" * 78)
print(f"{AGENT} (seat {SEAT}) vs {OPP}   seed={SEED}")
print("=" * 78)

hdr = f"{'day':>4}{'money':>10}{'hands':>6}{'cow':>4}{'shp':>4}{'gse':>4}" \
      f"{'mel':>4}{'str':>4}{'wht':>4}{'car':>4}{'weed':>5}{'free':>5}{'quads':>6}" \
      f"{'shed':>6}"
print(hdr)
prev_day = -1
for st in env.steps:
    obs = st[0].observation
    day = obs["day"]
    hour = obs["hour"]
    if day == prev_day or hour != 0:
        continue
    prev_day = day
    farm = obs["farms"][SEAT]
    priv = st[SEAT].observation["private"]
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
    shed = sum(v for v in priv["shed"].values() if v > 0)
    print(f"{day:>4}{farm['money']:>10,.0f}{len(farm['hands']):>6}"
          f"{c['COW']:>4}{c['SHEEP']:>4}{c['GOOSE']:>4}"
          f"{c['MELON']:>4}{c['STRAWBERRY']:>4}{c['WHEAT']:>4}{c['CARROT']:>4}"
          f"{weeds:>5}{free:>5}{len(farm['unlocked_quadrants']):>6}{shed:>6}")

last = env.steps[-1]
print(f"\nFINAL  us={last[SEAT].reward:,.0f}   opp={last[1-SEAT].reward:,.0f}"
      f"   status={last[SEAT].status}/{last[1-SEAT].status}")

print("\nUNIT ACTIONS (total across farmer+hands):")
tot = sum(ops.values())
for k, v in ops.most_common():
    print(f"   {k:<20}{v:>7}  {100.0*v/tot:>5.1f}%")
print(f"   {'TOTAL':<20}{tot:>7}")

print("\nMARKET ORDERS:")
for k, v in mops.most_common():
    print(f"   {k:<20}{v:>7}")
print(f"   hires issued: {hires}")

print("\nSOLD:")
for k, v in sold.most_common():
    print(f"   {k:<14}{v:>7}")
print("\nBOUGHT:")
for k, v in bought.most_common():
    print(f"   {k:<14}{v:>7}")

fin = env.steps[-1][0].observation["market"]
print("\nFINAL MARKET (inv / price):")
for k in fin["prices"]:
    print(f"   {k:<14}{fin['inventory'][k]:>8}  ${fin['prices'][k]}")
