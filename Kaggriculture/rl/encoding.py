"""Deterministic numerical encoding of an observation."""
import numpy as np
from .actions import CROPS, ANIMALS, PRODUCTS, ITEMS, unit_mask, market_mask

TILE_KINDS = ("EMPTY","LOCKED","WEED","WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","GOOSE","COW","SHEEP","COOP","PASTURE")
SHOPS = ("BAKERY","PIZZA_SHOP","BRUNCH_SPOT","YARN_STORE","ICE_CREAM_SHOP","PET_CAFE","SMOOTHIE_SHOP","FARMERS_MARKET")


def _kind(tile):
    if tile is None: return "EMPTY"
    if tile == "LOCKED": return "LOCKED"
    if not isinstance(tile, dict): return "LOCKED"
    if tile.get("kind") == "PLANT": return tile.get("crop", "EMPTY")
    if "animal" in tile: return tile["animal"]
    return tile.get("kind", "EMPTY")


def encode(obs, max_units=12):
    pid = int(obs["player"]); farms = obs["farms"]; me = farms[pid]; opp = farms[1-pid]
    n = len(me["tiles"]); channels = 24
    board = np.zeros((channels, n, n), np.float32)
    for owner, farm in enumerate((me, opp)):
        off = 0 if owner == 0 else 13
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                k = _kind(tile); idx = TILE_KINDS.index(k) if k in TILE_KINDS else 0
                if owner == 0: board[idx, y, x] = 1
                else: board[13 + min(idx, 7), y, x] = 1
                if isinstance(tile, dict):
                    board[21, y, x] = min(1, tile.get("yield_units",0)/6)
                    board[22, y, x] = float(tile.get("watered_today",tile.get("fed_today",False)))
                    board[23, y, x] = min(1, tile.get("pending_care_bonus",0)/8)
    units = [me["farmer"]] + list(me.get("hands", [])); invs = obs["private"].get("inventories", [])
    uf = np.zeros((max_units,20),np.float32); um = []
    for i in range(max_units):
        if i < len(units):
            x,y=units[i]; uf[i,0]=1; uf[i,1]=x/max(1,n-1); uf[i,2]=y/max(1,n-1)
            inv=invs[i] if i<len(invs) else {}
            for j,item in enumerate(ITEMS[:12]): uf[i,3+j]=min(1,inv.get(item,0)/16)
            uf[i,15]=i/max(1,max_units-1); uf[i,16]=obs.get("hour",0)/23
        um.append(unit_mask(obs,i,max_units))
    g=[]
    g += [obs.get("day",0)/29,obs.get("hour",0)/23,me["money"]/100000,opp["money"]/100000,len(units)/max_units]
    for farm in (me,opp): g += [len(farm.get("unlocked_quadrants",[]))/4,farm.get("hires_today",0)/12]
    shed=obs["private"].get("shed",{}); seeds=obs["private"].get("seeds",{})
    g += [min(1,shed.get(x,0)/100) for x in ITEMS]
    g += [min(1,seeds.get(x,0)/50) for x in CROPS]
    prices=obs.get("market",{}).get("prices",{}); inventory=obs.get("market",{}).get("inventory",{})
    g += [prices.get(x,0)/500 for x in PRODUCTS]
    g += [(inventory.get(x,10000)-10000)/1000 for x in PRODUCTS]
    shops=obs.get("town",{}).get("unlocked_shops",[])
    g += [shops.count(x)/8 for x in SHOPS]
    global_vec=np.zeros(96,np.float32); global_vec[:min(96,len(g))]=g[:96]
    return {"board":board,"global":global_vec,"units":uf,
            "unit_mask":np.stack(um),"market_mask":np.stack([market_mask(obs,s) for s in range(10)]),
            "num_units":len(units)}

