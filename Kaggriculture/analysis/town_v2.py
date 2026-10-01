"""Town demand under engine 1.32.6 vs the 1.32.4 rules we tuned against.

Three changes matter:

  1. townCenterSellInterval 12 -> 24 (once per day, not twice) AND the
     day-10 / day-20 multipliers (1x -> 2x -> 4x) are GONE. Flat 1 per day.
  2. Shops are drawn WITH REPLACEMENT, capped at MAX_SHOP_INSTANCES = 8.
     Duplicates concentrate demand on a random subset of products; variety is
     no longer guaranteed.
  3. Shed ops (PICKUP / DROP / PLACE) now resolve BEFORE the LOCKED guard, so
     a hand spawned on a locked shed-access tile can work immediately.

(1) is the big one: it cuts town demand by roughly 3x, which shrinks every
product's sellable capacity and makes gluts far more punishing.
"""
import os
import sys

V326 = "/tmp/v326/lib/python3.11/site-packages"
if os.path.isdir(V326):
    sys.path.insert(0, V326)

from kaggle_environments.envs.kaggriculture.kaggriculture import (  # noqa: E402
    PRODUCTS, SHOPS, TOWN_CENTER_PRODUCTS, market_price, MARKET_I0,
)

TURNS_PER_DAY = 24
DAYS = 30
SHOP_UNLOCK_INTERVAL = 3
SHOP_SELL_INTERVAL = 4
MAX_SHOP_INSTANCES = 8


def expected_shop_mult(product):
    """Expected consumption per shop instance per tick, drawn uniformly."""
    tot = 0.0
    for items in SHOPS.values():
        if product in items:
            tot += 2 if len(items) == 1 else 1
    return tot / len(SHOPS)


def demand(center_interval, center_schedule, with_replacement):
    """Season demand per product. center_schedule=None -> flat 1."""
    out = {p: 0.0 for p in PRODUCTS}
    n_shop_kinds = len(SHOPS)
    for day in range(DAYS):
        n_inst = min(MAX_SHOP_INSTANCES if with_replacement else n_shop_kinds,
                     day // SHOP_UNLOCK_INTERVAL)
        ticks_shop = TURNS_PER_DAY / SHOP_SELL_INTERVAL
        ticks_center = TURNS_PER_DAY / center_interval
        if center_schedule:
            cmult = next(m for th, m in center_schedule if day >= th)
        else:
            cmult = 1
        for p in PRODUCTS:
            out[p] += n_inst * expected_shop_mult(p) * ticks_shop
            if p in TOWN_CENTER_PRODUCTS:
                out[p] += cmult * ticks_center
    return out


def units_to_floor(item):
    inv, n = MARKET_I0, 0
    while n < 5000 and market_price(item, inv) > 1:
        inv += 1
        n += 1
    return n if n < 5000 else float("inf")


PER_ANIMAL = {"MILK": ("cows", 33), "WOOL": ("sheep", 32), "EGG": ("geese", 50)}


def main():
    old = demand(12, [(20, 4), (10, 2), (0, 1)], with_replacement=False)
    new = demand(24, None, with_replacement=True)

    print("=" * 82)
    print("SEASON TOWN DEMAND - engine 1.32.4 (what we tuned on) vs 1.32.6 (live)")
    print("=" * 82)
    print(f"{'item':<12}{'OLD':>9}{'NEW':>9}{'change':>10}{'floor':>8}"
          f"{'old cap/plyr':>14}{'new cap/plyr':>14}")
    for p in PRODUCTS:
        f = units_to_floor(p)
        fs = "inf" if f == float("inf") else f"{f:.0f}"
        oc = (old[p] + (0 if f == float("inf") else f)) / 2
        nc = (new[p] + (0 if f == float("inf") else f)) / 2
        chg = (new[p] / old[p] - 1) * 100 if old[p] else 0.0
        ocs = "inf" if f == float("inf") else f"{oc:,.0f}"
        ncs = "inf" if f == float("inf") else f"{nc:,.0f}"
        print(f"{p:<12}{old[p]:>9.0f}{new[p]:>9.0f}{chg:>9.0f}%{fs:>8}"
              f"{ocs:>14}{ncs:>14}")

    print()
    print("=" * 82)
    print("HERD SIZING - capacity / units per animal")
    print("=" * 82)
    for product, (label, per) in PER_ANIMAL.items():
        f = units_to_floor(product)
        if f == float("inf"):
            print(f"  {product:<6} unbounded (log glut curve) -> limited by tiles/feed, not market")
            continue
        oc = (old[product] + f) / 2
        nc = (new[product] + f) / 2
        print(f"  {product:<6} OLD {oc/per:>5.1f} {label:<6} ->  NEW {nc/per:>5.1f} {label}")

    print()
    print("  Shops now draw WITH REPLACEMENT (cap 8 instances), so the per-product")
    print("  split above is only an EXPECTATION. In any single game demand can pile")
    print("  onto a few products and leave others at zero -- which is why the strong")
    print("  public agents read `town.unlocked_shops` live and count duplicates.")


if __name__ == "__main__":
    main()
