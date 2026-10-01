"""Kaggriculture economic EDA.

Everything here is derived from the engine source
(kaggle_environments/envs/kaggriculture/kaggriculture.py), not from the README,
so the numbers match what actually happens in an episode.

Outputs:
  analysis/out/*.png   charts
  analysis/out/*.csv   tables
"""
import math
import os
import csv

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    CROPS, ANIMALS, PRODUCTS, MARKET_PARAMS, MARKET_I0, SHOPS,
    TOWN_CENTER_PRODUCTS, TOWN_CENTER_DEMAND_SCHEDULE,
    market_price, _shape, _fib, LAND_PRICES,
)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

TURNS_PER_DAY = 24
DAYS = 30
SHOP_UNLOCK_INTERVAL = 3
SHOP_SELL_INTERVAL = 4
CENTER_SELL_INTERVAL = 12


# --------------------------------------------------------------------------
# 1. Price curves: what does the market pay for the Nth unit I dump?
# --------------------------------------------------------------------------
def revenue_curve(item, n_units, start_inv=MARKET_I0):
    """Sell n_units one at a time; return (unit_prices, total_revenue).

    Mirrors _commit_unit: price is quoted at pre-sell inventory, and a unit sold
    at the $1 floor does NOT increase market supply.
    """
    inv = start_inv
    prices = []
    total = 0
    for _ in range(n_units):
        p = market_price(item, inv)
        prices.append(p)
        total += p
        if p > 1:
            inv += 1
    return prices, total


def price_at_offset(item, offset):
    return market_price(item, MARKET_I0 + offset)


# --------------------------------------------------------------------------
# 2. Town demand: how much does the town drain per product over the season?
# --------------------------------------------------------------------------
def town_drain_schedule():
    """Replay the engine's town consumption for a full 720-step season.

    Shops unlock at end_of_day when (day+1) % 3 == 0, chosen randomly from the
    remaining pool -> by day 24 all 8 are active. We assume the average case of
    the random order by taking the expectation over which shops are unlocked
    (all orders give the same *total* once every shop is in, and the expected
    per-product drain at time t is the mean over remaining shops).
    """
    # expected number of unlocked shops at each day
    per_day = {p: [0.0] * DAYS for p in PRODUCTS}
    n_shops = len(SHOPS)
    for day in range(DAYS):
        # shops unlocked so far: one every 3 days, first at end of day 2 (day+1=3)
        n_unlocked = min(n_shops, (day) // SHOP_UNLOCK_INTERVAL)
        # expected demand contribution: each shop equally likely to be among the
        # n_unlocked, so expected multiplier per product = n_unlocked/n_shops * total
        ticks_shop = TURNS_PER_DAY / SHOP_SELL_INTERVAL          # 6 per day
        ticks_center = TURNS_PER_DAY / CENTER_SELL_INTERVAL      # 2 per day
        center_mult = next(m for th, m in TOWN_CENTER_DEMAND_SCHEDULE if day >= th)
        for p in PRODUCTS:
            demand = 0.0
            for shop, items in SHOPS.items():
                if p in items:
                    mult = 2 if len(items) == 1 else 1
                    demand += (n_unlocked / n_shops) * mult * ticks_shop
            if p in TOWN_CENTER_PRODUCTS:
                demand += center_mult * ticks_center
            per_day[p][day] = demand
    return per_day


# --------------------------------------------------------------------------
# 3. Production economics per tile
# --------------------------------------------------------------------------
def crop_profile(crop):
    """Units produced and tile-days occupied for one optimally-watered planting."""
    c = CROPS[crop]
    if not c["ongoing"]:
        # yield starts at 1, +1 per watered day in [ceil(max_yield_day/2), max_yield_day]
        window_start = (c["max_yield_day"] + 1) // 2
        watered_days = c["max_yield_day"] - window_start + 1
        units = min(c["max_yield"], 1 + watered_days)
        units_fert = min(c["max_yield"], 1 + 2 * watered_days)
        # harvest at max_yield_day (earliest day at full yield)
        occupancy = c["max_yield_day"]
        # actions: plant + water each day of life + harvest
        actions = 1 + c["max_yield_day"] + 1
        return dict(units=units, units_fert=units_fert, days=occupancy, actions=actions)
    # ongoing: max_yield scheduled productions at `interval` spacing
    n = c["max_yield"]
    last_day = c["first_yield_day"] + (n - 1) * c["interval"]
    units = n                      # base 1 each
    units_fert = 2 * n             # doubled if fertilized AND watered
    actions = 1 + last_day + n     # plant + waters + harvests
    return dict(units=units, units_fert=units_fert, days=last_day, actions=actions)


def animal_profile(animal, days_alive):
    """Steady-state units + fertilizer for one animal kept fed+cared."""
    a = ANIMALS[animal]
    prod_days = max(0, days_alive - a["first_yield_day"] + 1)
    n_prod = math.ceil(prod_days / a["interval"]) if prod_days > 0 else 0
    # fed+cared every day -> care bonus = interval (banked once per fed+cared day)
    units_per_prod = 1 + a["interval"]
    units = n_prod * units_per_prod
    fertilizer = days_alive                     # 1/day for every surviving animal
    wheat_feed = days_alive                     # 1 wheat/day to keep fed+cared
    # actions/day: FEED + CARE + COLLECT_FERTILIZER, plus harvests
    actions = days_alive * 3 + n_prod
    return dict(units=units, fertilizer=fertilizer, wheat=wheat_feed,
                actions=actions, n_prod=n_prod)


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------
def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    print("=" * 78)
    print("1. GLUT RESILIENCE - revenue from dumping N units into a fresh market")
    print("=" * 78)
    print(f"{'item':<12}{'base':>6}{'1u':>7}{'50u':>9}{'100u':>9}{'200u':>10}"
          f"{'400u':>10}{'floor@':>9}")
    rows = []
    for item in PRODUCTS:
        base = MARKET_PARAMS[item]["base"]
        line = [item, base]
        for n in (1, 50, 100, 200, 400):
            _, tot = revenue_curve(item, n)
            line.append(tot)
        # units until price hits the floor
        inv = MARKET_I0
        n = 0
        while market_price(item, inv) > 1 and n < 100000:
            inv += 1
            n += 1
        line.append(n)
        rows.append(line)
        print(f"{item:<12}{base:>6}{line[2]:>7}{line[3]:>9}{line[4]:>9}"
              f"{line[5]:>10}{line[6]:>10}{line[7]:>9}")
    write_csv("glut_resilience.csv",
              ["item", "base", "rev_1", "rev_50", "rev_100", "rev_200", "rev_400",
               "units_to_floor"], rows)

    print()
    print("=" * 78)
    print("2. SCARCITY UPSIDE - price when the town has drained N units")
    print("=" * 78)
    print(f"{'item':<12}{'base':>6}{'-100':>8}{'-200':>8}{'-400':>8}{'-800':>8}")
    rows = []
    for item in PRODUCTS:
        line = [item, MARKET_PARAMS[item]["base"]]
        for n in (100, 200, 400, 800):
            line.append(price_at_offset(item, -n))
        rows.append(line)
        print(f"{item:<12}{line[1]:>6}{line[2]:>8}{line[3]:>8}{line[4]:>8}{line[5]:>8}")
    write_csv("scarcity_upside.csv",
              ["item", "base", "p_-100", "p_-200", "p_-400", "p_-800"], rows)

    print()
    print("=" * 78)
    print("3. TOWN DEMAND - free drain per product over the 30-day season")
    print("=" * 78)
    drain = town_drain_schedule()
    print(f"{'item':<12}{'d0-9':>9}{'d10-19':>9}{'d20-29':>9}{'TOTAL':>9}"
          f"{'price@drain':>13}")
    rows = []
    for item in PRODUCTS:
        a = sum(drain[item][0:10])
        b = sum(drain[item][10:20])
        c = sum(drain[item][20:30])
        t = a + b + c
        p = price_at_offset(item, -int(t))
        rows.append([item, round(a, 1), round(b, 1), round(c, 1), round(t, 1), p])
        print(f"{item:<12}{a:>9.0f}{b:>9.0f}{c:>9.0f}{t:>9.0f}{p:>13}")
    write_csv("town_demand.csv",
              ["item", "days_0_9", "days_10_19", "days_20_29", "total", "price_if_undersupplied"],
              rows)
    print("\n  -> Total free demand is the size of the pie both players fight over.")
    print("     Selling AT the drain rate holds base price; selling past it crashes you.")

    print()
    print("=" * 78)
    print("4. CROP ECONOMICS - value per tile-day at BASE price (no glut)")
    print("=" * 78)
    print(f"{'crop':<12}{'seed':>6}{'units':>7}{'+fert':>7}{'days':>6}{'acts':>6}"
          f"{'$/tile-day':>12}{'$/action':>10}")
    rows = []
    for crop in CROPS:
        pr = crop_profile(crop)
        base = MARKET_PARAMS[crop]["base"]
        gross = pr["units"] * base
        net = gross - CROPS[crop]["seed"]
        per_tile_day = net / pr["days"]
        per_action = net / pr["actions"]
        rows.append([crop, CROPS[crop]["seed"], pr["units"], pr["units_fert"],
                     pr["days"], pr["actions"], round(per_tile_day, 1), round(per_action, 1)])
        print(f"{crop:<12}{CROPS[crop]['seed']:>6}{pr['units']:>7}{pr['units_fert']:>7}"
              f"{pr['days']:>6}{pr['actions']:>6}{per_tile_day:>12.1f}{per_action:>10.1f}")
    write_csv("crop_economics.csv",
              ["crop", "seed", "units", "units_fert", "tile_days", "actions",
               "per_tile_day", "per_action"], rows)

    print()
    print("=" * 78)
    print("5. ANIMAL ECONOMICS - one animal placed on day 2, fed+cared to day 29")
    print("=" * 78)
    days_alive = 28
    print(f"{'animal':<8}{'cost':>6}{'product':>9}{'units':>7}{'fert':>6}{'wheat':>7}"
          f"{'acts':>6}{'gross@base':>11}{'$/tile-day':>12}{'$/action':>10}")
    rows = []
    for an in ANIMALS:
        pr = animal_profile(an, days_alive)
        prod = ANIMALS[an]["product"]
        base = MARKET_PARAMS[prod]["base"]
        fert_base = MARKET_PARAMS["FERTILIZER"]["base"]
        gross = pr["units"] * base + pr["fertilizer"] * fert_base
        net = gross - ANIMALS[an]["cost"] - pr["wheat"] * MARKET_PARAMS["WHEAT"]["base"]
        rows.append([an, ANIMALS[an]["cost"], prod, pr["units"], pr["fertilizer"],
                     pr["wheat"], pr["actions"], gross, round(net / days_alive, 1),
                     round(net / pr["actions"], 1)])
        print(f"{an:<8}{ANIMALS[an]['cost']:>6}{prod:>9}{pr['units']:>7}"
              f"{pr['fertilizer']:>6}{pr['wheat']:>7}{pr['actions']:>6}{gross:>11}"
              f"{net/days_alive:>12.1f}{net/pr['actions']:>10.1f}")
    write_csv("animal_economics.csv",
              ["animal", "cost", "product", "units", "fertilizer", "wheat_fed",
               "actions", "gross_at_base", "per_tile_day", "per_action"], rows)

    print()
    print("=" * 78)
    print("6. LABOUR - cumulative cost of hiring N hands in one day")
    print("=" * 78)
    cum = 0
    rows = []
    print(f"{'nth hire':>9}{'cost':>7}{'cumulative':>12}{'actions/day':>13}{'$/action':>10}")
    for n in range(0, 20):
        c = _fib(n)
        cum += c
        acts = (n + 1) * TURNS_PER_DAY
        rows.append([n + 1, c, cum, acts, round(cum / acts, 3)])
        if n < 16:
            print(f"{n+1:>9}{c:>7}{cum:>12}{acts:>13}{cum/acts:>10.3f}")
    write_csv("hire_costs.csv",
              ["n_hands", "marginal_cost", "cumulative_cost", "unit_actions_per_day",
               "cost_per_action"], rows)
    print("\n  -> 10 hands for $143/day is ~$0.58 per extra action. Labour is nearly free;")
    print("     LAND and MARKET DEPTH are the real constraints.")

    print()
    print("=" * 78)
    print("7. LAND - payback on each quadrant")
    print("=" * 78)
    for i, price in enumerate(LAND_PRICES):
        quad = ["NE", "SW", "SE"][i]
        print(f"  {quad}: ${price:>5}  -> 25 tiles = ${price/25:>6.0f}/tile")
    print("  A cow tile nets roughly $200-400/day at healthy prices, so land pays back")
    print("  in days IF the market can still absorb the extra output.")


if __name__ == "__main__":
    main()
