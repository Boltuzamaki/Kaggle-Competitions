"""Kaggriculture agent - "Capacity Farmer" v3.

Design thesis (derived from the engine source, not the README):

  1. Labour is nearly free.  Hire cost is fib(n): 10 hands/day costs $143 for
     240 extra unit-actions (~$0.60/action).  The binding constraints are
     TILES and MARKET DEPTH, never actions -- so hire hard and spend the
     actions on service, not on walking.

  2. Every product has a season capacity: the town drains a fixed amount for
     free, and the glut curve allows a little headroom past that before the $1
     floor.  Town drain / floor headroom:
         MILK 437/76   WOOL 338/59   STRAWBERRY 536/62   MELON 140/158
         EGG  338/inf  WHEAT 635/inf FERTILIZER 0/493
     Split with one opponent, that caps milk at ~8 cows and wool at ~6 sheep.
     Selling past the cap is worthless: 200 milk earns $100 more than 100.

  3. EGG and WHEAT ride `log` glut curves with target 0.20, so they never
     reach the floor.  They are the safe overflow sink when a milk war has
     crashed milk -- but each goose also costs 3 service actions and 1 wheat
     a day, so the sleeve stays small.

  4. FERTILIZER is free money.  Every surviving animal yields 1/day whether or
     not it was fed, the town never drains it, and 493 units fit before the
     floor.  One action per animal per day at ~$60-100 is the best $/action in
     the game.

  5. Animals produce their base unit even when unfed and only escape after two
     consecutive unfed days -- but the banked CARE bonus is forfeited unless
     they are fed on the production day, and that bonus is worth more than the
     wheat.  So: feed and care daily, always.

  6. Melon is the capital event, not a plan: 6 units/tile at $250 base with
     only 140 units of town drain and a 158-unit cliff.  A ~14-tile sleeve
     harvested around day 10-12 is what actually funds the land and the herd.
     Sweeps peak sharply there -- 6 tiles and 22 tiles are both much worse.

Tuning notes that contradict intuition (all measured, both seats, many seeds):
  * Strawberry is a trap.  Zero tiles beats 8 and 14 everywhere: it floors
    after 62 units and holds a tile for 16 days.
  * Grow LESS wheat than feels right (4 tiles).  Buying feed is cheaper than
    paying the water actions to grow it -- but never SELL feed wheat: it buys
    at $35-55 and sells back at ~$20.

Architecture:
    plan      -> targets for herd / land / crops by day
    jobs      -> every legal unit-action, priced in dollars
    router    -> global greedy (unit, job) assignment, missions persist
    market    -> HIRE first, then SELLs ordered by price impact, then
                 feed / melon seed / land / animals / filler
"""

import math

# --------------------------------------------------------------------------
# Engine constants (mirrored so the agent is one self-contained file)
# --------------------------------------------------------------------------
CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

MARKET_I0 = 10000
PRICE_FLOOR = 1

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "T": 450, "below_func": "log",    "below_target": 0.20, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "T": 332, "below_func": "linear", "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

SHOPS = {
    "BAKERY":         ("EGG", "WHEAT"),
    "PIZZA_SHOP":     ("MILK", "TOMATO", "WHEAT"),
    "BRUNCH_SPOT":    ("EGG", "WHEAT", "STRAWBERRY"),
    "YARN_STORE":     ("WOOL",),
    "ICE_CREAM_SHOP": ("STRAWBERRY", "MILK", "WHEAT"),
    "PET_CAFE":       ("CARROT",),
    "SMOOTHIE_SHOP":  ("STRAWBERRY", "MILK"),
    "FARMERS_MARKET": ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY"),
}

# Expected consumption per shop INSTANCE if instances are drawn uniformly.
# Engine 1.32.6 draws shops WITH REPLACEMENT (cap 8 instances), so the realised
# mix is random per game: YARN_STORE landing twice doubles wool demand while
# other products get nothing. A frozen route cannot see this -- the strong
# public agents pre-build separate routes and branch on shop order. We compute it.
_EXPECTED_MULT = {}
for _p in PRODUCTS:
    _tot = 0.0
    for _items in SHOPS.values():
        if _p in _items:
            _tot += 2.0 if len(_items) == 1 else 1.0
    _EXPECTED_MULT[_p] = _tot / len(SHOPS)


LAND_PRICES = [1000, 2000, 4000]
TURNS_PER_DAY = 24
SEASON_DAYS = 30
SHED_CAP = 100
SHED_TILES = [(4, 4), (5, 4), (4, 5), (5, 5)]
_SHED_SET = set(SHED_TILES)


def _shape(func, x):
    x = max(0.0, x)
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return math.sqrt(x)
    if func == "log":
        return math.log(1.0 + x)
    return x


def market_price(item, inventory):
    p = MARKET_PARAMS[item]
    base, T = p["base"], p["T"]
    if inventory < MARKET_I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T)
        price = base + amp * _shape(f, MARKET_I0 - inventory)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T)
        price = base - amp * _shape(f, inventory - MARKET_I0)
    return max(PRICE_FLOOR, int(round(price)))


def sell_revenue(item, inv, n):
    total = 0
    for _ in range(n):
        p = market_price(item, inv)
        total += p
        if p > 1:
            inv += 1
    return total


# --------------------------------------------------------------------------
# Strategy parameters
# --------------------------------------------------------------------------
P = {
    "target_cows": 8,
    "target_sheep": 4,
    # Eggs never hit the $1 floor (`log` glut curve, target 0.20), so geese are
    # the only animal the market cannot saturate -- but each one costs 3 service
    # actions and 1 wheat every day, and sweeps put that below what the same
    # tile earns as melon. A small goose sleeve is the hedge for a milk war,
    # not the main engine.
    "target_geese": 0,
    "melon_tiles": 8,
    "straw_tiles": 14,
    "wheat_tiles": 4,
    "wheat_cap": 26,
    "shed_pressure": 45,
    "flush_hour": 20,
    "filler_start_day": 9,
    "seed_cash_floor": 1100,
    "seed_cash_floor_late": 500,
    "hands_d0": 6,
    "hands_early": 8,
    "hands_mid": 10,
    "hands_late": 11,
    "min_sell_price": {
        "MILK": 55, "WOOL": 60, "STRAWBERRY": 40, "MELON": 55,
        "TOMATO": 18, "CARROT": 10, "EGG": 10, "WHEAT": 6, "FERTILIZER": 15,
    },
    "wheat_buffer_days": 4,
    "land_empty_gate": 16,
    "land_buffers": [900, 2500, 7000],
    "land_last_day": [24, 22, 18],
    # Ladder evidence: we lose by tiny margins (-563, -5231) against agents
    # banking 66-115k. Bank is what converts those, so allow the demand signal
    # to scale targets DOWN as well as up -- it frees tiles and cash from
    # products this game's shops are not buying.
    "demand_ratio_min": 0.5,
    "demand_ratio_max": 2.0,
    "liquidate_day": 28,
}


# Optional tuning hook: a JSON dict in KAGGRICULTURE_P overrides any entry in P.
# Absent in competition runs, so this is a no-op there; used by tune.py locally.
try:  # pragma: no cover
    import os as _os
    import json as _json
    if _os.environ.get("KAGGRICULTURE_P"):
        P.update(_json.loads(_os.environ["KAGGRICULTURE_P"]))
except Exception:
    pass


def _shed_ops_ignore_locked():
    """True on engines where shed ops resolve before the LOCKED tile guard."""
    try:
        import importlib.metadata as _m
        v = tuple(int(x) for x in _m.version("kaggle-environments").split(".")[:3])
        return v >= (1, 32, 5)
    except Exception:
        return False


_SHED_OPS_IGNORE_LOCKED = _shed_ops_ignore_locked()


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _shed_dist(p):
    return min(_dist(p, s) for s in SHED_TILES)


def _step_toward(src, dst):
    sx, sy = src
    dx, dy = dst
    if sx != dx:
        return "EAST" if sx < dx else "WEST"
    if sy != dy:
        return "SOUTH" if sy < dy else "NORTH"
    return None


class Brain:
    """Per-player persistent state (missions survive across turns)."""

    def __init__(self):
        self.missions = {}      # unit index -> mission dict
        self.day = -1
        self.tile_role = {}     # (x,y) -> "ANIMAL" | "CROP"
        # Shed-access tiles that are actually usable. Three of the four sit in
        # locked quadrants at the start, and EVERY tile op (PICKUP/DROP/PLACE)
        # silently no-ops on a locked tile -- while _spawn_hand still sends the
        # first hire of each day to (5,4). Units must walk to an unlocked one.
        self.shed_tiles = [(4, 4)]
        self.shed_set = {(4, 4)}
        self._buying_power = 0
        self.shops = []

    # ------------------------------------------------------------------
    def act(self, obs):
        player = obs.get("player", 0)
        farms = obs.get("farms", [])
        if not farms or player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        priv = obs.get("private", {}) or {}
        shed = dict(priv.get("shed", {}) or {})
        seeds = dict(priv.get("seeds", {}) or {})
        invs = list(priv.get("inventories", []) or [{}])
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        money = float(farm["money"])
        tiles = farm["tiles"]
        n = len(tiles)
        market = obs.get("market", {}) or {}
        minv = dict(market.get("inventory", {}) or {})
        unlocked = list(farm.get("unlocked_quadrants", ["NW"]))

        # Engine 1.32.6 resolves DROP / PICKUP / PLACE BEFORE the LOCKED guard
        # (the shed is always owned; the tile is only a standing position), so
        # all four shed-access tiles work even while their quadrant is locked.
        # On 1.32.4 they silently no-opped and units had to walk to an unlocked
        # one -- keep that fallback if an older engine is ever used.
        if _SHED_OPS_IGNORE_LOCKED:
            self.shed_tiles = list(SHED_TILES)
        else:
            self.shed_tiles = [p for p in SHED_TILES
                               if tiles[p[1]][p[0]] != "LOCKED"] or [(4, 4)]
        self.shed_set = set(self.shed_tiles)

        if day != self.day:
            self.day = day
            self.missions = {}          # units respawn at the shed each day

        units = [list(farm["farmer"])] + [list(h) for h in farm.get("hands", [])]

        town = obs.get("town", {}) or {}
        self.shops = list(town.get("unlocked_shops", []) or [])
        counts = self.census(tiles, n)
        self.assign_roles(tiles, n, counts, day)

        carried_wheat = sum(int(iv.get("WHEAT", 0) or 0) for iv in invs)
        mkt = self.build_market(farm, shed, seeds, minv, day, hour, money,
                                unlocked, tiles, n, counts, carried_wheat)

        jobs = self.build_jobs(tiles, n, shed, seeds, day, hour, minv, counts)
        ops = self.route(units, jobs, invs, shed, seeds, day, hour, counts, tiles)

        return {"farmer": ops[0], "hands": ops[1:], "market": mkt}

    # ==================================================================
    # Census & layout
    # ==================================================================
    def census(self, tiles, n):
        c = {k: 0 for k in list(CROPS) + list(ANIMALS) +
             ["COOP_total", "PASTURE_total", "COOP_free", "PASTURE_free",
              "WEED", "EMPTY"]}
        c["empties"] = []
        c["free_struct"] = {"COOP": [], "PASTURE": []}
        for y in range(n):
            for x in range(n):
                t = tiles[y][x]
                if t is None:
                    c["EMPTY"] += 1
                    c["empties"].append((x, y))
                elif isinstance(t, dict):
                    k = t.get("kind")
                    if k == "PLANT":
                        c[t["crop"]] += 1
                    elif k == "WEED":
                        c["WEED"] += 1
                    elif k in ("COOP", "PASTURE"):
                        c[k + "_total"] += 1
                        if "animal" in t:
                            c[t["animal"]] += 1
                        else:
                            c[k + "_free"] += 1
                            c["free_struct"][k].append((x, y))
        c["empties"].sort(key=self.sd)
        c["animals"] = c["COW"] + c["SHEEP"] + c["GOOSE"]
        return c

    def demand_ratio(self, item):
        """Observed town demand for `item` vs the neutral expectation.

        1.0 = this game's shops demand what an average draw would. >1 = demand
        concentrated here (YARN_STORE twice -> wool ~2x). Blended toward neutral
        while few shops have unlocked, so early turns are not driven by noise.
        """
        shops = self.shops
        n = len(shops)
        if n < 2:
            return 1.0
        observed = 0.0
        for name in shops:
            items = SHOPS.get(name)
            if items and item in items:
                observed += 2.0 if len(items) == 1 else 1.0
        expected = n * _EXPECTED_MULT.get(item, 0.0)
        if expected <= 0:
            return 1.0
        raw = observed / expected
        w = min(1.0, n / 5.0)
        r = 1.0 + w * (raw - 1.0)
        return max(P["demand_ratio_min"], min(P["demand_ratio_max"], r))

    def sd(self, p):
        """Distance to the nearest USABLE (unlocked) shed-access tile."""
        return min(_dist(p, s) for s in self.shed_tiles)

    def assign_roles(self, tiles, n, counts, day):
        """Tiles nearest the shed are reserved for animals: they need
        FEED+CARE+COLLECT+HARVEST every single day, so their travel cost is
        paid ~4x per day, while a melon is watered once."""
        want_struct = P["target_cows"] + P["target_sheep"] + P["target_geese"]
        near = counts["empties"][:max(0, want_struct - counts["COOP_total"]
                                      - counts["PASTURE_total"])]
        self.tile_role = {p: "ANIMAL" for p in near}

    # ==================================================================
    # Jobs
    # ==================================================================
    def build_jobs(self, tiles, n, shed, seeds, day, hour, minv, counts):
        jobs = []
        liq = day >= P["liquidate_day"]
        days_left = SEASON_DAYS - day

        def price(item):
            return market_price(item, minv.get(item, MARKET_I0))

        for y in range(n):
            for x in range(n):
                t = tiles[y][x]
                if t == "LOCKED" or t is None:
                    continue
                if not isinstance(t, dict):
                    continue
                pos = (x, y)
                kind = t.get("kind")

                # ------------------------------------------------ weeds
                if kind == "WEED":
                    jobs.append(dict(pos=pos, op=["DIG"], value=25.0, kind="dig"))
                    continue

                # ------------------------------------------------ plants
                if kind == "PLANT":
                    crop = t["crop"]
                    cd = CROPS[crop]
                    age = day - t["planted_day"]
                    yu = int(t.get("yield_units", 0))
                    pcrop = price(crop)

                    if yu > 0 and age >= cd["first_yield_day"]:
                        if not cd["ongoing"]:
                            ripe = yu >= cd["max_yield"] or age >= cd["max_yield_day"]
                            if ripe or liq or days_left <= 1:
                                jobs.append(dict(pos=pos, op=["HARVEST"],
                                                 value=yu * pcrop, kind="harvest"))
                        else:
                            m = 1.8 if yu >= cd["max_yield"] - 1 else 1.0
                            jobs.append(dict(pos=pos, op=["HARVEST"],
                                             value=yu * pcrop * m, kind="harvest"))

                    if not t.get("watered_today") and not liq:
                        cu = int(t.get("consecutive_unwatered", 0))
                        must = cu >= 1
                        gain = 0.0
                        if not cd["ongoing"]:
                            ws = (cd["max_yield_day"] + 1) // 2
                            if ws <= age <= cd["max_yield_day"] and yu < cd["max_yield"]:
                                gain = (2 if t.get("fertilized_until_day", -1) >= day
                                        else 1) * pcrop
                        elif t.get("fertilized_until_day", -1) >= day:
                            gain = pcrop * 0.6
                        surv = (self.plant_future(t, cd, pcrop, day) if must else 0.0)
                        v = gain + surv
                        if v > 1.0:
                            jobs.append(dict(pos=pos, op=["WATER"], value=v,
                                             kind="water", urgent=must))

                    if (cd["ongoing"] and not liq
                            and t.get("fertilized_until_day", -1) < day
                            and yu < cd["max_yield"]
                            and days_left > 2):
                        gain = pcrop - price("FERTILIZER") * 0.7
                        if gain > 5:
                            jobs.append(dict(pos=pos, op=["FERTILIZE"], value=gain,
                                             kind="fertilize", need="FERTILIZER"))
                    continue

                # ----------------------------------------------- animals
                if "animal" in t:
                    an = t["animal"]
                    ad = ANIMALS[an]
                    prod = ad["product"]
                    pprod = price(prod)
                    yu = int(t.get("yield_units", 0))

                    if yu > 0:
                        m = 2.0 if yu >= ad["max_held"] - 1 else 1.0
                        jobs.append(dict(pos=pos, op=["HARVEST"],
                                         value=yu * pprod * m, kind="harvest"))

                    if t.get("fertilizer_available"):
                        jobs.append(dict(pos=pos, op=["COLLECT_FERTILIZER"],
                                         value=price("FERTILIZER") * 0.9,
                                         kind="fert"))

                    if not t.get("fed_today") and not liq:
                        cu = int(t.get("consecutive_unfed", 0))
                        surv = self.animal_future(ad, pprod, day)
                        # feeding on a production day releases the banked CARE bonus
                        bonus = t.get("pending_care_bonus", 0) * pprod
                        v = (surv if cu >= 1 else surv * 0.20) + bonus
                        jobs.append(dict(pos=pos, op=["FEED"], value=max(v, 40.0),
                                         kind="feed", need="WHEAT", urgent=cu >= 1))

                    if (not t.get("cared_today") and not liq and yu < ad["max_held"]
                            and days_left > 1):
                        jobs.append(dict(pos=pos, op=["CARE"],
                                         value=pprod / ad["interval"] * 0.9,
                                         kind="care"))
                    continue

                # -------------------------------------- empty structures
                if kind in ("COOP", "PASTURE"):
                    for an, ad in ANIMALS.items():
                        if ad["structure"] == kind and shed.get(an, 0) > 0:
                            if days_left > ad["first_yield_day"]:
                                jobs.append(dict(pos=pos, op=["PLACE", an],
                                                 value=1200.0, kind="place",
                                                 need=an))
                            break

        jobs.extend(self.empty_tile_jobs(counts, seeds, day, minv, days_left))
        return jobs

    def plant_future(self, t, cd, pcrop, day):
        age = day - t["planted_day"]
        if cd["ongoing"]:
            done = max(0, (age - cd["first_yield_day"]) // max(1, cd["interval"]) + 1)
            left = max(0, cd["max_yield"] - done)
            return left * pcrop * 0.9 + int(t.get("yield_units", 0)) * pcrop
        ws = (cd["max_yield_day"] + 1) // 2
        rem = max(0, cd["max_yield_day"] - max(age, ws - 1))
        pot = min(cd["max_yield"], int(t.get("yield_units", 0)) + rem)
        return pot * pcrop * 0.9

    def animal_future(self, ad, pprod, day):
        days_left = max(0, SEASON_DAYS - 1 - day)
        n_prod = days_left // ad["interval"]
        return n_prod * (1 + ad["interval"]) * pprod * 0.4 + days_left * 40

    def empty_tile_jobs(self, counts, seeds, day, minv, days_left):
        jobs = []
        empties = counts["empties"]
        if not empties:
            return jobs

        # --- structures on the near tiles
        need_p = self.want_pasture(counts, days_left)
        need_c = self.want_coop(counts, days_left)
        i = 0
        for _ in range(need_p):
            if i >= len(empties):
                break
            jobs.append(dict(pos=empties[i], op=["BUILD_PASTURE"], value=700.0,
                             kind="build"))
            i += 1
        for _ in range(need_c):
            if i >= len(empties):
                break
            jobs.append(dict(pos=empties[i], op=["BUILD_COOP"], value=550.0,
                             kind="build"))
            i += 1

        # --- crops on the rest, nearest-first (travel is the real cost)
        rest = empties[i:]
        wants = self.crop_wants(counts, day, days_left)
        ri = 0
        for crop in ("MELON", "STRAWBERRY", "WHEAT", "CARROT"):
            want = wants.get(crop, 0)
            have_seed = seeds.get(crop, 0)
            k = min(want, have_seed, max(0, len(rest) - ri))
            p = market_price(crop, minv.get(crop, MARKET_I0))
            val = min(CROPS[crop]["max_yield"] * p, 1500) * 0.30
            for _ in range(k):
                jobs.append(dict(pos=rest[ri], op=["PLANT", crop], value=val,
                                 kind="plant"))
                ri += 1
        return jobs

    def want_pasture(self, counts, days_left):
        """Only build ahead of animals we can actually pay for.

        An empty pasture is a dead tile and a wasted action; paving the farm
        with structures we can never stock is how the opening bankrupts itself.
        """
        if days_left <= ANIMALS["SHEEP"]["first_yield_day"] + 1:
            return 0
        cap = P["target_cows"] + P["target_sheep"]
        if days_left <= ANIMALS["COW"]["first_yield_day"] + 1:
            cap = counts["COW"] + counts["SHEEP"]
        affordable = counts["COW"] + counts["SHEEP"] + self._buying_power // 400
        want = min(cap, affordable + 1)
        return max(0, min(want - counts["PASTURE_total"], 3))

    def want_coop(self, counts, days_left):
        if days_left <= ANIMALS["GOOSE"]["first_yield_day"] + 1:
            return 0
        affordable = counts["GOOSE"] + self._buying_power // 300
        want = min(P["target_geese"], affordable + 1)
        return max(0, min(want - counts["COOP_total"], 3))

    def crop_wants(self, counts, day, days_left):
        w = {}
        w["MELON"] = (max(0, P["melon_tiles"] - counts["MELON"])
                      if days_left > CROPS["MELON"]["max_yield_day"] else 0)
        # Before the melon money lands, every tile and every coin belongs to the
        # opening herd. Filler crops here just starve the cows.
        if day < P["filler_start_day"]:
            w["STRAWBERRY"] = w["WHEAT"] = w["CARROT"] = 0
            return w
        straw_t = int(round(P["straw_tiles"] * self.demand_ratio("STRAWBERRY")))
        w["STRAWBERRY"] = (max(0, straw_t - counts["STRAWBERRY"])
                           if days_left > CROPS["STRAWBERRY"]["first_yield_day"] + 2 else 0)
        # wheat feeds the herd and is glut-proof
        w["WHEAT"] = (max(0, P["wheat_tiles"] - counts["WHEAT"])
                      if days_left > CROPS["WHEAT"]["max_yield_day"] else 0)
        w["CARROT"] = (max(0, int(round(8 * self.demand_ratio("CARROT")))
                              - counts["CARROT"])
                       if 3 < days_left <= 10 else 0)
        return w

    # ==================================================================
    # Router - persistent missions
    # ==================================================================
    def route(self, units, jobs, invs, shed, seeds, day, hour, counts, tiles):
        n_units = len(units)
        ops = [["PASS"] for _ in range(n_units)]
        turns_left = TURNS_PER_DAY - hour

        # index jobs by key so a mission can be re-validated each turn
        jmap = {}
        for j in jobs:
            jmap[(j["pos"], tuple(j["op"]))] = j

        claimed = set()
        # keep still-valid missions
        for i in list(self.missions.keys()):
            if i >= n_units:
                del self.missions[i]
                continue
            key = self.missions[i]
            if key not in jmap:
                del self.missions[i]
            else:
                claimed.add(key)

        shed_wheat = shed.get("WHEAT", 0)
        shed_fert = shed.get("FERTILIZER", 0)
        shed_animals = {a: shed.get(a, 0) for a in ANIMALS}

        # Global greedy assignment. Scoring each (unit, job) pair and taking the
        # best pair first -- instead of letting unit 0 pick first -- keeps a unit
        # that is already standing on a tile from being outbid for that tile's
        # own jobs by a unit four steps away. An animal tile has four jobs a day
        # (FEED / CARE / COLLECT_FERTILIZER / HARVEST); servicing them on one
        # visit instead of four is the difference between 1 and 4 useful actions
        # per round trip.
        need_unit = [i for i in range(n_units) if i not in self.missions]
        if need_unit:
            pairs = []
            for i in need_unit:
                pos = tuple(units[i])
                inv = invs[i] if i < len(invs) else {}
                for key, j in jmap.items():
                    sc = self.score(pos, inv, j, turns_left)
                    if sc > 0.0:
                        pairs.append((sc, i, key))
            pairs.sort(key=lambda t: -t[0])
            busy = set()
            for sc, i, key in pairs:
                if i in busy or key in claimed:
                    continue
                self.missions[i] = key
                claimed.add(key)
                busy.add(i)

        for i in range(n_units):
            pos = tuple(units[i])
            inv = invs[i] if i < len(invs) else {}
            carrying = sum(v for v in inv.values() if v > 0)

            key = self.missions.get(i)
            if key is None:
                ops[i] = self.idle(pos, carrying, hour)
                continue

            job = jmap[key]
            need = job.get("need")

            # ---------------- fetch a required item from the shed first
            if need and inv.get(need, 0) <= 0:
                if pos in self.shed_set:
                    avail = (shed_wheat if need == "WHEAT" else
                             shed_fert if need == "FERTILIZER" else
                             shed_animals.get(need, 0))
                    if avail <= 0:
                        del self.missions[i]
                        claimed.discard(key)
                        ops[i] = self.idle(pos, carrying, hour)
                        continue
                    if need == "WHEAT":
                        take = min(avail, max(1, self.feed_load(jobs)))
                        shed_wheat -= take
                    elif need == "FERTILIZER":
                        take = min(avail, 2)
                        shed_fert -= take
                    else:
                        take = 1
                        shed_animals[need] -= 1
                    ops[i] = ["PICKUP", need, take]
                else:
                    tgt = min(self.shed_tiles, key=lambda s: _dist(pos, s))
                    mv = _step_toward(pos, tgt)
                    ops[i] = [mv] if mv else ["PASS"]
                continue

            # ---------------- travel or execute
            if pos == job["pos"]:
                ops[i] = list(job["op"])
                del self.missions[i]
                claimed.discard(key)
                # a unit that just harvested and is full should head home
            else:
                mv = _step_toward(pos, job["pos"])
                ops[i] = [mv] if mv else ["PASS"]

        return ops

    def feed_load(self, jobs):
        return max(1, min(8, sum(1 for j in jobs if j["kind"] == "feed")))

    def score(self, pos, inv, j, turns_left):
        """Dollars per turn if this unit takes this job."""
        need = j.get("need")
        travel = _dist(pos, j["pos"])
        if need and inv.get(need, 0) <= 0:
            travel = self.sd(pos) + 1 + self.sd(j["pos"])   # detour via the shed
        if travel + 1 > turns_left:
            return 0.0
        s = j["value"] / (1.0 + travel)
        if j.get("urgent"):
            s *= 4.0
        return s

    def idle(self, pos, carrying, hour):
        if pos in self.shed_set:
            if carrying > 0:
                return ["DROP"]
            return ["PASS"]
        if carrying > 0 or hour >= TURNS_PER_DAY - 3:
            tgt = min(self.shed_tiles, key=lambda s: _dist(pos, s))
            mv = _step_toward(pos, tgt)
            return [mv] if mv else ["PASS"]
        tgt = min(self.shed_tiles, key=lambda s: _dist(pos, s))
        mv = _step_toward(pos, tgt)
        return [mv] if mv else ["PASS"]

    # ==================================================================
    # Market
    # ==================================================================
    def build_market(self, farm, shed, seeds, minv, day, hour, money, unlocked,
                     tiles, n, counts, carried_wheat=0):
        orders = []
        days_left = SEASON_DAYS - day
        liq = day >= P["liquidate_day"]
        shed_total = sum(v for v in shed.values() if v > 0)

        # ------------------------------------------------------- 0. HIRE
        # maxMarketOrdersPerTurn is 10 and extras are SILENTLY DROPPED. If sells
        # are queued first they eat the whole budget at hour 0 and the crew never
        # gets hired -- which caps every other thing the farm can do that day.
        # Selling can happen on any of the other 23 turns; hiring cannot.
        if hour == 0 and not liq:
            for _ in range(farm.get("hires_today", 0), self.hand_target(day, money)):
                orders.append(["HIRE"])

        # ------------------------------------------------------- 1. SELL
        sells = []
        for item in PRODUCTS:
            have = shed.get(item, 0)
            if have <= 0:
                continue
            if not liq:
                if item == "WHEAT":
                    # We BUY wheat for feed at $35-55 and it sells back at ~$20,
                    # so round-tripping it is a straight loss. Only home-grown
                    # surplus beyond the feed reserve may go, and only when the
                    # shed is about to destroy something worth more.
                    if shed_total <= P["shed_pressure"]:
                        continue
                    have -= self.wheat_reserve(counts, days_left) + counts["animals"]
                elif item == "FERTILIZER":
                    have -= self.fert_reserve(counts, days_left)
                if have <= 0:
                    continue
            inv0 = minv.get(item, MARKET_I0)
            unit0 = market_price(item, inv0)
            flush = liq or hour >= P["flush_hour"] or shed_total > P["shed_pressure"]
            floor = 0 if flush else P["min_sell_price"].get(item, 0)
            qty, probe = 0, inv0
            while qty < have:
                p = market_price(item, probe)
                if p < floor:
                    break
                qty += 1
                if p > 1:
                    probe += 1
            if flush and qty < have:
                qty = have          # overflow is destroyed at end of day
            if qty <= 0:
                continue
            rev = sell_revenue(item, inv0, qty)
            after = market_price(item, inv0 + qty)
            impact = qty * (unit0 - after)   # revenue lost by being quoted second
            sells.append((impact, rev, item, qty))
        sells.sort(key=lambda s: -s[0])
        for _, _, item, qty in sells:
            orders.append(["SELL", item, qty])
        cash = money + sum(s[1] for s in sells)

        if liq:
            return orders[:10]

        # ------------------------------------------------------ 3. FEED
        # Feed is senior to every other claim on cash. An animal that misses two
        # end-of-day refreshes escapes permanently, and each cow is worth ~$7k
        # of remaining production against ~$40 of wheat.
        room = max(0, SHED_CAP - 6 - shed_total)
        need = self.wheat_reserve(counts, days_left) - shed.get("WHEAT", 0) - carried_wheat
        wp = market_price("WHEAT", minv.get("WHEAT", MARKET_I0))
        if need > 0 and room > 0:
            hungry = (counts["animals"] > 0
                      and shed.get("WHEAT", 0) + carried_wheat < counts["animals"])
            budget = cash if hungry else cash * 0.45
            qty = max(0, min(need, int(budget // max(1, wp)), room, 30))
            if qty > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", qty])
                cash -= qty * wp
                shed_total += qty

        # Everything below spends only what is left after the herd is safe.
        cash -= self.feed_cash_reserve(counts, days_left, wp)
        # What the job layer may assume when deciding to build structures.
        self._buying_power = max(0, int(cash))

        # -------------------------------------- 4. MELON IPO (days 0-2 only)
        # 10 melon tiles -> ~60 units -> ~$14k landing on day 10-12, which is
        # what actually pays for the land and the full herd. Seed cost is $80
        # and needs no wheat, so it outranks a third cow in the opening.
        ordered = {c: 0 for c in CROPS}      # seeds ordered THIS turn; obs["seeds"]
        # will not reflect them until the next observation, so track them here
        # or we re-order the same seeds every turn and drain the bank.
        if day <= 2:
            want = (P["melon_tiles"] - counts["MELON"] - seeds.get("MELON", 0))
            qty = max(0, min(want, counts["EMPTY"], int(cash // CROPS["MELON"]["seed"])))
            if qty > 0:
                orders.append(["BUY_SEED", "MELON", qty])
                cash -= qty * CROPS["MELON"]["seed"]
                ordered["MELON"] += qty

        # ------------------------------------------------------ 5. LAND
        if self.buy_land(day, cash, unlocked, counts):
            orders.append(["BUY_LAND"])
            cash -= LAND_PRICES[len(unlocked) - 1]

        # --------------------------------------------------- 6. ANIMALS
        for an in ("COW", "SHEEP", "GOOSE"):
            ad = ANIMALS[an]
            if days_left <= ad["first_yield_day"] + 1:
                continue
            want = self.animal_want(an, counts, day, days_left)
            free = len(counts["free_struct"][ad["structure"]])
            pending = sum(shed.get(a, 0) for a in ANIMALS
                          if ANIMALS[a]["structure"] == ad["structure"])
            buy = min(want, max(0, free - pending), int(cash // ad["cost"]))
            buy = min(buy, max(0, SHED_CAP - 6 - shed_total))
            if buy > 0:
                orders.append(["BUY_ANIMAL", an, buy])
                cash -= buy * ad["cost"]
                shed_total += buy

        # ----------------------------------------------------- 7. SEEDS
        # Discretionary crops are the LAST claim on cash. A strawberry seed is
        # $100 and pays on day 12+; a cow is $400 and pays $7k. Letting seed
        # buying outbid the opening herd is what bankrupts day 1.
        floor = P["seed_cash_floor"] if day < 10 else P["seed_cash_floor_late"]
        wants = self.crop_wants(counts, day, days_left)
        empties = counts["EMPTY"] - sum(ordered.values())
        for crop in ("MELON", "STRAWBERRY", "WHEAT", "CARROT"):
            want = wants.get(crop, 0) - seeds.get(crop, 0) - ordered[crop]
            if want <= 0 or empties <= 0:
                continue
            c = CROPS[crop]["seed"]
            qty = min(want, empties, int(max(0.0, cash - floor) // c))
            if qty > 0:
                orders.append(["BUY_SEED", crop, qty])
                cash -= qty * c
                empties -= qty
        return orders[:10]

    def feed_cash_reserve(self, counts, days_left, wheat_price):
        """Cash we refuse to spend on anything but wheat."""
        a = counts["animals"]
        if a == 0:
            return 0.0
        return a * min(3, max(1, days_left)) * wheat_price

    def hand_target(self, day, cash):
        # Hiring 11 hands costs $232 for 264 extra actions. Under-hiring is a
        # death spiral: no labour -> no service -> no produce -> no cash.
        # Only a genuinely empty bank should reduce the crew.
        if cash < 40:
            return 4
        if day == 0:
            return P["hands_d0"]
        if day <= 3:
            return P["hands_early"]
        if day <= 25:
            return P["hands_mid"]
        return P["hands_late"]

    def wheat_reserve(self, counts, days_left):
        # Every reserved item costs a shed slot, and the shed caps at 100 with
        # overflow DESTROYED at end of day. Wheat can be bought any turn, so
        # hold a thin buffer rather than a warehouse.
        a = counts["animals"]
        if a == 0:
            return 0
        return int(min(P["wheat_cap"], a * min(P["wheat_buffer_days"], max(1, days_left))))

    def fert_reserve(self, counts, days_left):
        if days_left < 3:
            return 0
        return min(10, counts["STRAWBERRY"] + counts["TOMATO"])

    def animal_want(self, an, counts, day, days_left):
        base = {"COW": P["target_cows"], "SHEEP": P["target_sheep"],
                "GOOSE": P["target_geese"]}[an]
        # Size the herd to the demand THIS game has, not the average one.
        tgt = int(round(base * self.demand_ratio(ANIMALS[an]["product"])))
        return max(0, tgt - counts[an])

    def buy_land(self, day, cash, unlocked, counts):
        k = len(unlocked) - 1
        if k >= 3:
            return False
        cost = LAND_PRICES[k]
        # A quadrant only pays if there is season left to farm it and we can
        # actually service the tiles. The strong public route has its 2nd
        # quadrant by day 8 and its 3rd by day 12 -- far earlier than we did.
        if counts["EMPTY"] > P["land_empty_gate"]:
            return False
        buf = P["land_buffers"][k]
        last = P["land_last_day"][k]
        return day <= last and cash >= cost + buf


# --------------------------------------------------------------------------
_BRAINS = {}


def agent(obs, config=None):
    try:
        pid = obs.get("player", 0)
        b = _BRAINS.get(pid)
        if b is None:
            b = _BRAINS[pid] = Brain()
        return b.act(obs)
    except Exception:
        return {"farmer": ["PASS"], "hands": [], "market": []}
