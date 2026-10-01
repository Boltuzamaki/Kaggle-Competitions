"""Where does the bank actually come from, and what did bad sell timing cost?

Runs one episode and, for each turn, records the market orders each player
issued next to the price the market was quoting at the time.  Revenue is
attributed from the bank delta, so it counts only orders that actually
executed (the engine caps orders per turn and silently drops the rest).

The counterfactual column is a ceiling, not a plan: it prices the same units at
the best quote that product saw over the rest of the season.  It exists to size
the prize on sell timing before anyone writes a scheduler.

Two warnings, both learned the hard way:

  * The `units` column is DERIVED (revenue / quoted price), not measured. When a
    product is pinned at the $1 floor it reports absurd unit counts. Trust the
    revenue column; treat units as a smell, not a number.
  * Revenue is attributed to the item SOLD, which credits the wrong thing for
    anything whose value is indirect. Wool clears about $1 a unit here, which
    reads as "sheep do not pay" -- but deleting sheep costs 65% of the bank,
    because animals feed the fertilizer chain that doubles crop yields. This
    split cannot see that.
"""
import collections, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def audit(a, b, seed=0, steps=720):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": steps})
    env.run([a, b])
    st = env.steps
    best_future = {}
    prices_by_step = [s[0]["observation"]["market"]["prices"] for s in st]
    items = list(prices_by_step[0])
    running = {i: 0 for i in items}
    for i in range(len(st) - 1, -1, -1):
        for it in items:
            running[it] = max(running[it], prices_by_step[i][it])
        best_future[i] = dict(running)

    out = []
    for p in (0, 1):
        rev = collections.Counter()
        units = collections.Counter()
        ceiling = collections.Counter()
        spend = collections.Counter()
        for i in range(len(st) - 1):
            act = st[i][p].get("action") or {}
            orders = [o for o in (act.get("market") or []) if o]
            if not orders:
                continue
            sells = [o for o in orders if o[0] == "SELL"]
            if not sells:
                continue
            d = (st[i + 1][0]["observation"]["farms"][p]["money"]
                 - st[i][0]["observation"]["farms"][p]["money"])
            px = prices_by_step[i]
            want = sum(px.get(o[1], 0) * int(o[2]) for o in sells if len(o) > 2)
            for o in sells:
                if len(o) < 3:
                    continue
                item, q = o[1], int(o[2])
                share = (px.get(item, 0) * q / want) if want else 0
                got = max(0.0, d) * share
                rev[item] += got
                # units that plausibly moved, priced at the quote we accepted
                n = got / px[item] if px.get(item) else 0
                units[item] += n
                ceiling[item] += n * best_future[i][item]
            spend["gross_out"] += min(0.0, d)
        out.append((rev, units, ceiling))
    final = st[-1][0]["observation"]["farms"]
    return out, [f["money"] for f in final]


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    b = sys.argv[2] if len(sys.argv) > 2 else "main.py"
    (res, banks) = audit(a, b)
    for p, (rev, units, ceil) in enumerate(res):
        print(f"\n=== player {p} ({[a, b][p]})  final bank ${banks[p]:,.0f}")
        print(f"  {'item':<12}{'revenue':>12}{'units':>9}{'best-quote ceiling':>21}")
        for it, v in rev.most_common():
            print(f"  {it:<12}{v:>12,.0f}{units[it]:>9,.0f}{ceil[it]:>21,.0f}")
        print(f"  {'TOTAL':<12}{sum(rev.values()):>12,.0f}{sum(units.values()):>9,.0f}"
              f"{sum(ceil.values()):>21,.0f}")
