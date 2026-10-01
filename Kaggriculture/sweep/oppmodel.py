"""A live opponent-supply model for the chassis `front_run` layer.

`front_run` asks one question every turn: is the opponent about to sell a
glut-prone product?  The layer is bounded by our own already-planned sales, so
all it can ever do is reorder our sell schedule -- never invent a sale -- which
is what makes an approximate answer safe.

v11 answered it with "assume they are running our tape", which only holds for
the share of the ladder running this same lineage.  This answers it from what
the observation actually shows: both farms are public, so the opponent's
harvest-ready yield is visible even though their shed is not.  Under 1.32.7 a
glut drives MELON, WOOL, MILK and STRAWBERRY to the $1 floor roughly 100-200
units above I0, so being one turn ahead of their supply is the whole game.
"""

LAYER = '''

# --------------------------------------------------------------- live opponent model
_ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
_OPP_OBS = {}


def _opp_ready(observation):
    """Harvest-ready units per product on the OPPONENT's board.

    Both farms are public. The shed is not, so this sees what is standing in the
    field and about to be picked, which is the supply that has not hit the market
    yet -- exactly the part worth pre-empting.
    """
    try:
        me = _int(_get(observation, "player", 0))
        farms = list(_get(observation, "farms", []) or [])
        if len(farms) < 2:
            return {}
        tiles = _get(farms[1 - me], "tiles", []) or []
        ready = {}
        for row in tiles:
            for tile in row or []:
                if not isinstance(tile, dict):
                    continue
                units = _int(tile.get("yield_units", 0))
                if units <= 0:
                    continue
                kind = tile.get("kind")
                if kind == "PLANT":
                    item = tile.get("crop")
                elif kind in ("COOP", "PASTURE"):
                    item = _ANIMAL_PRODUCT.get(tile.get("animal"))
                else:
                    item = None
                if item in PRODUCTS:
                    ready[item] = ready.get(item, 0) + units
        return ready
    except Exception:
        return {}


class _LiveOppPlan:
    """Stands in for the opponent's tape, built from their visible board.

    Indexed by step like a tape, but the step is ignored: what matters to
    front_run is only which products are about to arrive, and the answer is
    recomputed from the latest observation every turn.
    """

    def __len__(self):
        return LAST_ACT_STEP + 2

    def __getitem__(self, i):
        obs = _OPP_OBS.get("latest")
        if obs is None:
            return {}
        ready = _opp_ready(obs)
        return {"market": [["SELL", item, qty] for item, qty in ready.items()
                           if item in FRONT_RUN_ITEMS and qty > 0]}


_IMPL.chassis.opponent_plan = _LiveOppPlan()
_OPP_PARENT = agent
del agent


def agent(observation, configuration=None):
    try:
        _OPP_OBS["latest"] = observation
    except Exception:
        pass
    return _OPP_PARENT(observation, configuration)
'''
