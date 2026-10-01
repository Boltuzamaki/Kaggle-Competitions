"""Generate the Kaggle notebook from README + main.py so they never drift."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT = open(os.path.join(HERE, "main.py")).read()

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s}

cells = []

cells.append(md("""# Kaggriculture - Capacity Farmer 🌾

**Sizing a farm to market depth, derived from the engine source.**

This notebook does four things:

1. Reads the **engine** (`kaggriculture.py`) rather than the rules text, and computes what each product is actually worth.
2. Shows the one number that decides strategy: **how many units of each product the market can absorb before it pays $1.**
3. Derives the herd from that number - and lands on the public meta's "8 cows + 6 sheep" from first principles.
4. Ships a complete, self-contained agent (`main.py`) you can submit from this notebook.

> ⚠️ **This is a simulation competition.** There is no `submission.csv`. The scorer wants `main.py` with an `agent(obs)` function. Ranking is a **skill rating from win/loss/tie only** - your coin margin does not move it.

---"""))

cells.append(md("""## 1. Setup"""))
cells.append(code("""import sys, subprocess, math, json
try:
    import kaggle_environments
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle-environments"], check=True)
    import kaggle_environments

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS, ANIMALS, PRODUCTS, MARKET_PARAMS, MARKET_I0, SHOPS,
    TOWN_CENTER_PRODUCTS, TOWN_CENTER_DEMAND_SCHEDULE, market_price, _fib,
)
import importlib.metadata as _md
print("kaggle-environments", _md.version("kaggle-environments"))"""))

cells.append(md("""## 2. The price curve

```
price(inv) = base ± amp · f(|inv − I0|)        I0 = 10,000,  floor $1
amp        = target · base / f(T)
f          ∈ {linear, sq, sqrt, log}           chosen INDEPENDENTLY per side
```

The asymmetry is the whole game. `WHEAT` panics on scarcity but shrugs off a glut;
`MELON` barely reacts to scarcity but collapses when oversupplied."""))

cells.append(code("""import pandas as pd
rows = []
for item, p in MARKET_PARAMS.items():
    rows.append(dict(item=item, base=p["base"], T=p["T"],
                     below=f'{p["below_func"]} {p["below_target"]}',
                     above=f'{p["above_func"]} {p["above_target"]}',
                     p_scarce=market_price(item, MARKET_I0 - p["T"]),
                     p_glut=market_price(item, MARKET_I0 + p["T"]),
                     p_glut2=market_price(item, MARKET_I0 + 2 * p["T"])))
pd.DataFrame(rows).set_index("item")"""))

cells.append(md("""## 3. Glut resilience - what dumping N units actually pays

Sell one unit at a time, quoting at the pre-sell inventory, and remember the engine
rule that **a unit sold at the $1 floor does not increase market supply**."""))

cells.append(code("""def revenue_curve(item, n, inv=MARKET_I0):
    out, total = [], 0
    for _ in range(n):
        p = market_price(item, inv)
        total += p
        out.append(total)
        if p > 1:
            inv += 1
    return out

def units_to_floor(item):
    inv, n = MARKET_I0, 0
    while market_price(item, inv) > 1 and n < 5000:
        inv += 1; n += 1
    return n if n < 5000 else float("inf")

rows = []
for item in PRODUCTS:
    c = revenue_curve(item, 400)
    rows.append(dict(item=item, base=MARKET_PARAMS[item]["base"],
                     rev_50=c[49], rev_100=c[99], rev_200=c[199], rev_400=c[399],
                     units_to_floor=units_to_floor(item)))
glut = pd.DataFrame(rows).set_index("item")
glut"""))

cells.append(md("""**Read `rev_100` against `rev_200`.** Selling 200 milk earns about **$100 more** than
selling 100. Wool and strawberry are the same story. Past the floor, extra production
is worth nothing at all - so the correct herd size is a *market* question, not a
farming one.

Meanwhile `WHEAT` and `EGG` never reach the floor: both ride `log` glut curves with
target `0.20`. They are the safe overflow sink."""))

cells.append(code("""import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Categorical hues in fixed slot order (never cycled, never re-ordered per chart).
SERIES = [("MELON", "#2a78d6"), ("MILK", "#eb6834"), ("WOOL", "#1baf7a"),
          ("EGG", "#eda100"), ("WHEAT", "#e87ba4")]

fig, ax = plt.subplots(figsize=(10, 5.6))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
N = 320
for name, colour in SERIES:
    y = revenue_curve(name, N)
    ax.plot(range(1, N + 1), y, color=colour, linewidth=2, label=name, zorder=3)
    # direct label at the line end (relief rule: these hues need visible labels)
    ax.annotate(f" {name}", (N, y[-1]), color="#0b0b0b", fontsize=9,
                va="center", ha="left", fontweight="medium")

ax.set_xlim(0, N * 1.22)
ax.set_xlabel("units sold into a fresh market", color="#52514e")
ax.set_ylabel("cumulative revenue ($)", color="#52514e")
ax.set_title("Revenue saturates - and where it saturates decides the farm",
             color="#0b0b0b", fontsize=13, fontweight="semibold", loc="left", pad=14)
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.grid(axis="y", color="#e6e5e1", linewidth=1, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#d6d5d0")
ax.tick_params(colors="#52514e")
ax.legend(frameon=False, loc="upper left", labelcolor="#52514e")
plt.tight_layout(); plt.show()"""))

cells.append(md("""Melon is the steepest earner in the game *and* it cliffs at 158 units. That is
exactly why the "melon IPO" works once and punishes the second dumper.

## 4. The town is the demand side

The town centre and its shops consume product **for free**, pulling inventory below
`I0` and pushing prices up. This is the pie both players compete for - and it is
**back-loaded**, with over half arriving in the last third of the season."""))

cells.append(code("""TURNS_PER_DAY, DAYS = 24, 30
def town_drain():
    n_shops = len(SHOPS)
    per_item = {p: [0.0] * DAYS for p in PRODUCTS}
    for day in range(DAYS):
        n_unlocked = min(n_shops, day // 3)           # one shop every 3 days
        center_mult = next(m for th, m in TOWN_CENTER_DEMAND_SCHEDULE if day >= th)
        for p in PRODUCTS:
            d = 0.0
            for shop, items in SHOPS.items():
                if p in items:
                    d += (n_unlocked / n_shops) * (2 if len(items) == 1 else 1) * (TURNS_PER_DAY / 4)
            if p in TOWN_CENTER_PRODUCTS:
                d += center_mult * (TURNS_PER_DAY / 12)
            per_item[p][day] = d
    return per_item

drain = town_drain()
rows = [dict(item=p, d0_9=round(sum(drain[p][:10])), d10_19=round(sum(drain[p][10:20])),
             d20_29=round(sum(drain[p][20:])), season=round(sum(drain[p])),
             price_if_matched=market_price(p, MARKET_I0 - int(sum(drain[p]))))
        for p in PRODUCTS]
town = pd.DataFrame(rows).set_index("item").sort_values("season", ascending=False)
town"""))

cells.append(code("""fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
d = town.sort_values("season")
bars = ax.barh(d.index, d["season"], color="#2a78d6", height=0.62, zorder=3)
for rect, v in zip(bars, d["season"]):
    ax.annotate(f"  {int(v)}", (rect.get_width(), rect.get_y() + rect.get_height() / 2),
                va="center", ha="left", color="#52514e", fontsize=9)
ax.set_xlim(0, d["season"].max() * 1.16)
ax.set_title("Free town demand over the season (units)", color="#0b0b0b",
             fontsize=13, fontweight="semibold", loc="left", pad=14)
ax.set_xlabel("units consumed by town centre + shops", color="#52514e")
ax.grid(axis="x", color="#e6e5e1", linewidth=1, zorder=0); ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#d6d5d0")
ax.tick_params(colors="#52514e", length=0)
plt.tight_layout(); plt.show()"""))

cells.append(md("""**Fertilizer is the outlier: the town never buys it.** Its 493 units before the
floor are a fixed, uncontested pot - and every surviving animal drops one per day
*whether or not it was fed*. One action per animal per day for ~$60-100 is the best
$/action in the game.

## 5. The sizing rule

```
your capacity ≈ (town drain + floor headroom) / 2 players
```"""))

cells.append(code("""for product, animal, per in (("MILK", "COW", 33), ("WOOL", "SHEEP", 32), ("EGG", "GOOSE", 50)):
    dr = sum(drain[product]); hd = units_to_floor(product)
    share = (dr + hd) / 2
    n = share / per
    hd_s = "inf" if hd == float("inf") else f"{hd:.0f}"
    print(f"{product:<11} drain {dr:>6.0f} + headroom {hd_s:>5} -> your share "
          f"{share:>8.0f} / {per} per {animal.lower()} = {n:>5.1f} {animal.lower()}s")"""))

cells.append(md("""That reproduces the public meta's **8 cows + 6 sheep** without copying anyone - it
falls out of market depth. Eggs are the exception: with no floor, geese are limited
by *tiles and feed*, not by the market.

## 6. Labour is nearly free

Hire cost is `fib(n)` and resets every day."""))

cells.append(code("""cum = 0
rows = []
for n in range(14):
    cum += _fib(n)
    rows.append(dict(hands=n + 1, marginal=_fib(n), cumulative=cum,
                     extra_actions=(n + 1) * 24, cost_per_action=round(cum / ((n + 1) * 24), 3)))
pd.DataFrame(rows).set_index("hands")"""))

cells.append(md("""Ten hands cost **$143 for 240 extra unit-actions** - about 60 cents an action.

**Tiles and market depth are the binding constraints, never actions.** Under-hiring
is a death spiral: no labour → no service → no produce → no cash → fewer hands.

---

## 7. Five engine facts that cost me real money to learn

| Trap | What it looks like |
|---|---|
| The **first hire of each day spawns on (5,4), which is LOCKED** until you buy NE - and *every* tile op silently no-ops on a locked tile | hundreds of wasted `PICKUP`s; animals starve and escape |
| **`SELL` only sees the shed**; `HARVEST` fills the *unit's* inventory | produce never reaches the market |
| **`PLACE` needs the animal in the unit's inventory** - buying puts it in the shed | you buy 30 animals and place none |
| **The 10-orders-per-turn cap silently drops extras** - sells queued first eat the hour-0 budget, so `HIRE` never lands | you run 7 units instead of 12 |
| **Shed caps at 100 and overflow is DESTROYED** at the nightly drop | a steady invisible leak from mid-season |

Fixing just the locked-shed-tile bug moved a local game from **$12k to $30k**.
Moving `HIRE` ahead of `SELL` moved it from **$54k to $61k**.

---

## 8. The agent

Priority-ordered market layer, dollar-priced job layer, globally-greedy router.

Two design notes worth stealing:

- **Sells are ordered by price impact** - `qty × (price_now − price_after)`, i.e. how
  much revenue is lost by going second. Market orders resolve index-by-index across
  both players, so slot 0 is quoted first. Steep glut curves go first; flat staples
  (wheat, egg) can wait. *"Sell the expensive thing first" is the intuitive rule and
  it is wrong* - the expensive thing is often the one whose price barely moves.
- **The router is globally greedy**, scoring every (unit, job) pair and taking the
  best pair first. That stops a unit already standing on a tile from being outbid for
  that tile's own jobs by a unit four steps away. An animal tile has four jobs a day
  (`FEED`/`CARE`/`COLLECT_FERTILIZER`/`HARVEST`); doing them on one visit instead of
  four is the difference between 1 and 4 useful actions per round trip."""))

# %%writefile keeps the agent readable in the notebook and avoids nesting the
# agent's own triple-quoted docstring inside a wrapper string.
cells.append(code("%%writefile main.py\n" + AGENT))
cells.append(code("""import os
print("main.py written:", os.path.getsize("main.py"), "bytes")
import ast; ast.parse(open("main.py").read()); print("parses OK")"""))

cells.append(md("""## 9. Validate

Both agents must finish `DONE`. The market is shared, so seats are **not**
symmetric - always test both."""))

cells.append(code("""from kaggle_environments import make
import time

for opp in ("starter", "random"):
    for seat in (0, 1):
        agents = ["main.py", opp] if seat == 0 else [opp, "main.py"]
        t = time.time()
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 7}, debug=True)
        env.run(agents)
        last = env.steps[-1]
        me, them = last[seat].reward, last[1 - seat].reward
        print(f"vs {opp:<8} seat {seat}: us={me:>10,.0f}  them={them:>10,.0f}  "
              f"{last[0].status}/{last[1].status}  ({time.time()-t:.1f}s)")"""))

cells.append(code("""# self-play: the melon-war check - does the plan survive an opponent doing the same thing?
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3}, debug=True)
env.run(["main.py", "main.py"])
last = env.steps[-1]
print(f"self-play: {last[0].reward:,.0f} vs {last[1].reward:,.0f}  "
      f"{last[0].status}/{last[1].status}")"""))

cells.append(md("""## 10. Package for submission"""))

cells.append(code("""import tarfile, os
with tarfile.open("submission.tar.gz", "w:gz") as tar:
    tar.add("main.py", arcname="main.py")
with tarfile.open("submission.tar.gz") as tar:
    print("archive_members:", tar.getnames())
print("size:", os.path.getsize("submission.tar.gz"), "bytes")"""))

cells.append(md("""**To score this notebook:** run all cells, then click **Submit to competition** and
choose `submission.tar.gz` (or `main.py` - either works).

From the CLI:

```bash
kaggle competitions submit kaggriculture -f main.py -m "Capacity Farmer"
kaggle competitions submissions kaggriculture
```

---

## 11. Where this can go next

- **Opponent-aware herd sizing.** Both farms are public. Count the opponent's cows and
  shift the mix toward whatever they are *not* saturating - eggs and fertilizer never
  floor, so they are the natural hedge in a milk war.
- **Sell timing.** Town demand is back-loaded; holding premium goods for a drained
  late market beats dumping on the day of harvest.
- **Travel.** Roughly half of all unit-actions are still movement. Tighter beat
  assignment (a unit owns a cluster of tiles for the day) is the largest remaining
  win, and it is worth more than any crop-table tweak.

If this was useful, upvote the notebooks that mapped the meta first - they are
credited in the repo README.

*Built by reading the engine, then instrumenting real games until the leaks showed up.*"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(HERE, "notebook", "kaggriculture-capacity-farmer.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "cells:", len(cells))
