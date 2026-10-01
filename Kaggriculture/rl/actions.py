"""Fixed action vocabulary and conservative legality masks.

The engine silently ignores invalid actions. Masks are deliberately conservative:
they remove actions that are certainly impossible, while the environment remains
the final authority. Quantities use small buckets plus ALL to make the market head
compact enough for recurrent RL.
"""
from dataclasses import dataclass
import numpy as np

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = ("GOOSE", "COW", "SHEEP")
ITEMS = PRODUCTS + ANIMALS

UNIT_ACTIONS = (
    [("PASS",), ("NORTH",), ("SOUTH",), ("EAST",), ("WEST",), ("DROP",),
     ("WATER",), ("HARVEST",), ("FERTILIZE",), ("FEED",),
     ("COLLECT_FERTILIZER",), ("CARE",), ("DIG",), ("BUILD_COOP",),
     ("BUILD_PASTURE",)]
    + [("PLANT", x) for x in CROPS]
    + [("PICKUP", x) for x in ITEMS]
    + [("PLACE", x) for x in ITEMS]
)

QTY = (1, 2, 4, 8, 16, 32)
MARKET_ACTIONS = (
    [("STOP",), ("HIRE",), ("BUY_LAND",)]
    + [("BUY_SEED", x, q) for x in CROPS for q in QTY]
    + [("BUY_PRODUCT", x, q) for x in ("WHEAT", "FERTILIZER") for q in QTY]
    + [("BUY_ANIMAL", x, q) for x in ANIMALS for q in QTY]
    + [("SELL", x, q) for x in PRODUCTS for q in QTY]
)


def _tile(obs, unit_idx):
    me = obs["farms"][obs["player"]]
    units = [me["farmer"]] + list(me.get("hands", []))
    if unit_idx >= len(units):
        return None, None, None
    x, y = units[unit_idx]
    return x, y, me["tiles"][y][x]


def unit_mask(obs, unit_idx, max_units=12):
    mask = np.zeros(len(UNIT_ACTIONS), dtype=np.bool_)
    x, y, tile = _tile(obs, unit_idx)
    if x is None:
        mask[0] = True
        return mask
    me, private = obs["farms"][obs["player"]], obs["private"]
    invs = private.get("inventories", [])
    inv = invs[unit_idx] if unit_idx < len(invs) else {}
    n = len(me["tiles"])
    allowed = {"PASS"}
    if y > 0: allowed.add("NORTH")
    if y + 1 < n: allowed.add("SOUTH")
    if x + 1 < n: allowed.add("EAST")
    if x > 0: allowed.add("WEST")
    shed_tiles = {(n//2-1,n//2-1),(n//2,n//2-1),(n//2-1,n//2),(n//2,n//2)}
    at_shed = (x, y) in shed_tiles
    if at_shed and sum(inv.values()) > 0: allowed.add("DROP")
    if tile is None:
        allowed.update(("BUILD_COOP", "BUILD_PASTURE"))
        if tile != "LOCKED": allowed.add("PLANT")
    elif isinstance(tile, dict):
        kind = tile.get("kind")
        if kind == "PLANT":
            allowed.update(("WATER", "HARVEST", "FERTILIZE", "DIG"))
        elif kind == "WEED": allowed.add("DIG")
        elif kind in ("COOP", "PASTURE"):
            if "animal" in tile: allowed.update(("FEED", "HARVEST", "COLLECT_FERTILIZER", "CARE"))
            else: allowed.update(("DIG", "PLACE"))
    if at_shed:
        allowed.update(("PICKUP", "PLACE"))
    for i, a in enumerate(UNIT_ACTIONS):
        op = a[0]
        if op not in allowed: continue
        if op == "PLANT" and private.get("seeds", {}).get(a[1], 0) <= 0: continue
        if op == "PICKUP" and private.get("shed", {}).get(a[1], 0) <= 0: continue
        if op == "PLACE" and inv.get(a[1], 0) <= 0: continue
        if op == "FEED" and inv.get("WHEAT", 0) <= 0: continue
        if op == "FERTILIZE" and inv.get("FERTILIZER", 0) <= 0: continue
        mask[i] = True
    mask[0] = True
    return mask


def market_mask(obs, slot=0):
    mask = np.ones(len(MARKET_ACTIONS), dtype=np.bool_)
    mask[0] = True
    me, private = obs["farms"][obs["player"]], obs["private"]
    money, shed = me["money"], private.get("shed", {})
    if slot > 0: mask[1] = False
    if len(me.get("unlocked_quadrants", [])) >= 4: mask[2] = False
    costs = {"WHEAT":10,"CARROT":20,"TOMATO":50,"STRAWBERRY":100,"MELON":80,
             "GOOSE":300,"COW":400,"SHEEP":500}
    prices = obs.get("market", {}).get("prices", {})
    for i, a in enumerate(MARKET_ACTIONS):
        op = a[0]
        if op == "SELL" and shed.get(a[1], 0) < a[2]: mask[i] = False
        elif op == "BUY_SEED" and money < costs[a[1]] * a[2]: mask[i] = False
        elif op == "BUY_ANIMAL" and money < costs[a[1]] * a[2]: mask[i] = False
        elif op == "BUY_PRODUCT" and money < prices.get(a[1], 9999) * a[2]: mask[i] = False
    return mask


def decode_unit(index):
    return list(UNIT_ACTIONS[int(index)])


def decode_market(indices):
    out = []
    for i in indices:
        a = MARKET_ACTIONS[int(i)]
        if a[0] == "STOP": break
        out.append(list(a))
    return out[:10]

