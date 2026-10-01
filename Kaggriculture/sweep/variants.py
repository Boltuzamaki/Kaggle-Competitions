"""Build agent source variants from the shop-router base.

The base agent is an open-loop plan: 13 precomputed 719-step route tapes plus
reactive repair layers.  Two things in it are cheap to re-fit and were fit by
someone else against someone else's opponent pool:

  * ``_router``'s shop-pair -> route table (read at step 144, day 6)
  * the terminal route swapped in at step 648 (day 27)

Everything here is pure text substitution on the source, so a variant is a
self-contained ``.py`` the arena can run unchanged.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

ROUTER_TABLE_RE = re.compile(
    r"(state\['route'\]=)\{.*?\}(\.get\(tuple\(shops\[:2\]\),)(\d+)(\))",
    re.S,
)
TERMINAL_RE = re.compile(r"(if step>=648 and not state\.get\('day27'\):\s*\n\s*state\['route'\]=)(\d+)")
SETTINGS_RE = re.compile(r"^_SETTINGS=\{.*?\}$", re.M)


def parse_router(src):
    """-> (table dict {(shopA, shopB): route}, default route, terminal route)."""
    m = ROUTER_TABLE_RE.search(src)
    if not m:
        raise ValueError("router table not found")
    table = eval(m.group(0)[len(m.group(1)):-len(m.group(2) + m.group(3) + m.group(4))])
    t = TERMINAL_RE.search(src)
    return table, int(m.group(3)), int(t.group(2)) if t else None


def n_routes(src):
    return len(re.findall(r"_ROUTES\[int\(_rid\)\]", src)) and (
        len(eval(re.search(r"\{[^{}]*\}\.get\(tuple\(shops\[:2\]\)", src).group(0)[:-len(".get(tuple(shops[:2])")]))
    )


def build(src, table=None, default=None, terminal=None, settings=None):
    out = src
    if table is not None or default is not None:
        cur_table, cur_default, _ = parse_router(src)
        tbl = cur_table if table is None else table
        dft = cur_default if default is None else default
        lit = "{" + ", ".join(f"({k[0]!r}, {k[1]!r}): {v}" for k, v in sorted(tbl.items())) + "}"
        out = ROUTER_TABLE_RE.sub(
            lambda m: m.group(1) + lit + m.group(2) + str(dft) + m.group(4), out, count=1)
    if terminal is not None:
        out = TERMINAL_RE.sub(lambda m: m.group(1) + str(terminal), out, count=1)
    if settings is not None:
        out = SETTINGS_RE.sub("_SETTINGS=" + repr(settings), out, count=1)
    return out


def write_variant(base_path, out_path, **kw):
    src = open(base_path).read()
    new = build(src, **kw)
    with open(out_path, "w") as f:
        f.write(new)
    return out_path


# --------------------------------------------------------------------------- specs
TERMINAL_HARDCODE_RE = re.compile(r"(routes\[)2( if [a-z_+0-9 ]*>=648 else)")


def build_spec(src, spec):
    """Apply a named variant spec.  Keys:

      router_fix  {(shopA, shopB): route}   overrides merged into the table
      default     int                       route for unlisted shop pairs
      terminal    int                       route swapped in at step 648 --
                                            also rewrites the outer layers that
                                            hardcode route 2 for that window
      settings    dict                      replaces the _SETTINGS literal
    """
    out = src
    router_fix = spec.get("router_fix")
    if router_fix or spec.get("default") is not None:
        table, default, _ = parse_router(src)
        table.update(router_fix or {})
        out = build(out, table=table, default=spec.get("default", default))
    if spec.get("terminal") is not None:
        t = int(spec["terminal"])
        out = build(out, terminal=t)
        out = TERMINAL_HARDCODE_RE.sub(lambda m: m.group(1) + str(t) + m.group(2), out)
    if spec.get("settings") is not None:
        out = build(out, settings=spec["settings"])
    if spec.get("opp_lookahead"):
        out = opp_plan_lookahead(out, spec["opp_lookahead"])
    elif spec.get("opponent_plan"):
        out = add_opponent_plan(out)
    if spec.get("front_run_items"):
        out = set_front_run_items(out, spec["front_run_items"])
    if spec.get("own_window"):
        out = set_own_window(out, spec["own_window"])
    if spec.get("drop_animal"):
        out = drop_animal(out, spec["drop_animal"])
    return out


def parse_settings(src):
    m = SETTINGS_RE.search(src)
    return eval(m.group(0)[len("_SETTINGS="):]) if m else None


MAKE_AGENT_RE = re.compile(r"^_IMPL=make_agent\(_ROUTES,router=_router,\*\*_SETTINGS\)$", re.M)

# The front_run layer needs a model of what the opponent will sell next step.
# A large share of the ladder runs this same Apache-2.0 lineage, and both seats
# see the same shop draw, so "the opponent is running my route" is a defensible
# prior -- and the proxy follows our own route switch instead of pinning route 0.
OPP_PLAN_SHIM = '''
class _OppPlanProxy:
    """Stands in for the opponent's tape: our own current route, read lazily."""
    def __len__(self):
        return len(_ROUTES[0])

    def __getitem__(self, i):
        route = 0
        try:
            # one Chassis instance only ever plays one seat, so there is a single
            # per-player state entry and it carries the route the router settled on
            for _st in _IMPL.chassis.players.values():
                if _st.get("route") is not None:
                    route = _st["route"]
        except Exception:
            route = 0
        tape = _ROUTES.get(route) or _ROUTES[0]
        return tape[i] if 0 <= i < len(tape) else {}

_IMPL=make_agent(_ROUTES,router=_router,opponent_plan=_OppPlanProxy(),**_SETTINGS)
'''


def add_opponent_plan(src):
    if not MAKE_AGENT_RE.search(src):
        raise ValueError("make_agent call site not found")
    return MAKE_AGENT_RE.sub(OPP_PLAN_SHIM.strip("\n"), src, count=1)


def append_layer(src, layer):
    """Append a wrapper layer to the agent source.

    The lineage is already a stack of these: each block rebinds ``agent`` after
    stashing the previous one, so a new layer only has to follow the same shape.
    """
    return src.rstrip("\n") + "\n" + layer


FRONT_RUN_ITEMS_RE = re.compile(r'^FRONT_RUN_ITEMS = \([^)]*\)$', re.M)


def set_front_run_items(src, items):
    """Which products the front_run race covers.

    The shipped list is the four that collapse to the $1 floor on a glut. The
    1.32.7 hinge products (CARROT, TOMATO, EGG) behave the opposite way -- their
    price RISES as the town drains inventory -- so racing to sell them early is
    not obviously right, but a rival dumping them still craters the price for
    both sides. Worth measuring rather than assuming.
    """
    lit = "FRONT_RUN_ITEMS = (" + ", ".join(repr(i) for i in items) + ("," if len(items) == 1 else "") + ")"
    out, n = FRONT_RUN_ITEMS_RE.subn(lit, src, count=1)
    if not n:
        raise ValueError("FRONT_RUN_ITEMS not found")
    return out


def opp_plan_lookahead(src, window):
    """Widen the opponent-plan proxy to a K-step window.

    front_run only ever compares against ``plan[step + 1]``, so a proxy that
    answers with the sells planned across steps t..t+K-1 makes the layer fire
    whenever the opponent's supply is imminent rather than exactly next turn.
    It cannot make us sell MORE: the layer still caps the quantity at our own
    next-step planned SELL. It only changes how often the race is entered.
    """
    # The proxy may already be in the source (a champion built from an earlier
    # variant); only install it when it is not.
    out = src if "_OppPlanProxy" in src else add_opponent_plan(src)
    old = """        tape = _ROUTES.get(route) or _ROUTES[0]
        return tape[i] if 0 <= i < len(tape) else {}"""
    new = """        tape = _ROUTES.get(route) or _ROUTES[0]
        merged = {}
        for j in range(i, min(i + %d, len(tape))):
            if j < 0:
                continue
            entry = tape[j]
            if not isinstance(entry, dict):
                continue
            for o in entry.get("market") or []:
                if o and o[0] == "SELL" and len(o) >= 3:
                    merged[o[1]] = merged.get(o[1], 0) + max(0, _int(o[2]))
        return {"market": [["SELL", k, v] for k, v in merged.items() if v > 0]}""" % int(window)
    if old not in out:
        raise ValueError("proxy body not found")
    return out.replace(old, new, 1)


OWN_NEXT_RE = re.compile(
    r"            own_next = sum\(max\(0, _int\(x\[2\]\)\) for x in self\.routes\[route\]\[nxt\]\.get\(\"market\", \[\]\)\n"
    r"                           if len\(x\) >= 3 and x\[0\] == \"SELL\" and x\[1\] == item\)\n")


def set_own_window(src, window):
    """How far ahead of our own tape front_run may pull a sale.

    The layer caps the quantity it moves at ``own_next`` -- what our tape plans
    to SELL on the single next step. That cap, not the opponent model, is what
    binds: widening the opponent's window alone is provably inert. Widening
    OURS lets more stock reach the market before theirs does, at the cost of
    walking our own price down faster, so it is a genuine trade rather than a
    free win.
    """
    body = ('            own_next = sum(max(0, _int(x[2]))\n'
            '                           for _t in range(nxt, min(nxt + %d, len(self.routes[route])))\n'
            '                           for x in (self.routes[route][_t].get("market", []) or [])\n'
            '                           if len(x) >= 3 and x[0] == "SELL" and x[1] == item)\n' % int(window))
    out, n = OWN_NEXT_RE.subn(body, src, count=1)
    if not n:
        raise ValueError("own_next site not found")
    return out


DROP_ANIMAL_LAYER = '''

# ---------------------------------------------------------------- herd trim
# WOOL clears about $1 a unit in a contested game -- the $1 price floor -- because
# the wool market's whole seasonal capacity is T=105 units and both farms dump
# into it. A sheep costs $500 up front plus wheat every day it lives, so the
# question is whether the herd pays for itself at all. Editing the tapes in place
# after the chassis has built them is the cheap way to ask: the choreography is
# untouched, only the purchase order disappears.
_dropped = 0
for _tape in _IMPL.chassis.routes.values():
    for _i, _a in enumerate(_tape):
        if not isinstance(_a, dict):
            continue
        _m = _a.get("market") or []
        _keep = [o for o in _m if not (o and o[0] == "BUY_ANIMAL" and o[1] == %r)]
        if len(_keep) != len(_m):
            _tape[_i] = dict(_a, market=_keep)
            _dropped += len(_m) - len(_keep)
'''


def drop_animal(src, animal):
    return append_layer(src, DROP_ANIMAL_LAYER % animal)
