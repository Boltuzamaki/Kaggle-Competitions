"""Evolve the fork's DECK LIST, holding its policy fixed.

Deck composition has been the single biggest lever in this project (+112 from the
meta-deck swap, while 12 consecutive policy changes were rejected). The fork's 69
weights key on the core Alakazam/Abra/Kadabra line, so changing the counts of the
SUPPORT cards leaves every weight reference intact -- unlike a wholesale deck swap.

Mutations only touch cards outside CORE, and every genome is re-validated to 60
cards and <=4 copies before it is played, so an illegal list can never score.

Promotion gate is the hardened one from beat_stock: a big margin on the tuning
seeds plus a 4x larger DISJOINT confirmation with a real win. That is what
champion B failed after nine fake promotions on 16-game evaluations.
"""
from __future__ import annotations
import json, multiprocessing as mp, os, random, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "agent"))
from paired_eval import play, build, mcnemar  # noqa: E402

FP = os.path.join(ROOT, "agent", "fork", "fork_main.py")
STOCK = [int(x) for x in open(os.path.join(ROOT, "agent", "fork", "deck.csv")) if x.strip()]
# Never touched: the engine the weights are written against.
CORE = {741, 742, 743, 19, 1079, 1086}
FLEX = sorted({c for c in STOCK if c not in CORE})
GENS = int(os.environ.get("DE_GENS", 300))
POP = int(os.environ.get("DE_POP", 8))
NS = int(os.environ.get("DE_SEEDS", 20))
WORKERS = int(os.environ.get("DE_WORKERS", 8))
CKPT = os.path.join(ROOT, "scratchpad", "scrape_20260804", os.environ.get("DE_CKPT", "deck_evo.json"))
# td-td_08 and hybs-other are NOT deterministic run-to-run (verified), so they
# injected pure noise into every paired comparison and are removed.
OPPS = ["public-archaludon", "meta-grimmsnarl", "public-alakazam"]


def legal(counts):
    """Force a genome back to exactly 60 cards, <=4 copies each."""
    c = {k: max(0, min(4, v)) for k, v in counts.items() if v > 0}
    for k in CORE:
        c[k] = dict(Counter(STOCK))[k]
    keys = sorted(c)
    tot = sum(c.values())
    rng = random.Random(sum(keys) + tot)
    while tot != 60 and keys:
        k = rng.choice(keys)
        if k in CORE:
            continue
        if tot > 60 and c[k] > 0:
            c[k] -= 1; tot -= 1
        elif tot < 60 and c[k] < 4:
            c[k] += 1; tot += 1
    return c


def to_deck(counts):
    d = []
    for k in sorted(counts):
        d += [k] * counts[k]
    return d


def mutate(counts, rng):
    c = dict(counts)
    for _ in range(rng.randint(1, 3)):
        k = rng.choice(FLEX)
        c[k] = c.get(k, 0) + rng.choice([-1, 1])
    return legal(c)


def _load():
    import importlib.util as ilu
    s = ilu.spec_from_file_location("fm_w", FP); m = ilu.module_from_spec(s)
    s.loader.exec_module(m); return m


def _duel(args):
    """Paired duel of two DECK LISTS under the identical stock policy."""
    ca, cb, seeds = args
    import arena  # noqa
    m = _load()
    da, db = to_deck(ca), to_deck(cb)

    def bind(deck):
        def w(o):
            return list(deck) if o.get("select") is None else m.agent(o)
        return w
    A, B = bind(da), bind(db)
    aw = bw = 0
    for name in OPPS:
        fo, do = build(name)
        for s in seeds:
            x = y = 0
            for seat in (0, 1):
                random.seed(s)
                x += int(play(s, A, fo, da, do) == 0) if seat == 0 else int(play(s, fo, A, do, da) == 1)
            for seat in (0, 1):
                random.seed(s)
                y += int(play(s, B, fo, db, do) == 0) if seat == 0 else int(play(s, fo, B, do, db) == 1)
            aw += x; bw += y
    return aw, bw


def main():
    base = legal(dict(Counter(STOCK)))
    champ, gen0, wins = dict(base), 0, 0
    if os.path.exists(CKPT):
        try:
            ck = json.load(open(CKPT))
            champ = {int(k): v for k, v in ck["champion"].items()}
            gen0, wins = ck.get("generation", 0), ck.get("promotions", 0)
            print(f"resumed at gen {gen0}", flush=True)
        except Exception:
            pass
    rng = random.Random(int(os.environ.get("DE_RNG", 7)))
    print(f"{len(FLEX)} flex cards, pop {POP}, {NS} seeds x {len(OPPS)} opps "
          f"= {NS*len(OPPS)*4} games/candidate", flush=True)
    t0 = time.time()
    for gen in range(gen0 + 1, gen0 + GENS + 1):
        seeds = [640000 + gen * 89 + i for i in range(NS)]
        cands = [mutate(champ, rng) for _ in range(POP)]
        with mp.get_context("fork").Pool(min(POP, WORKERS)) as pool:
            res = pool.map(_duel, [(c, champ, seeds) for c in cands])
        best, bm = None, 0
        for c, (a, b) in zip(cands, res):
            if a - b > bm:
                best, bm = c, a - b
        st = "hold"
        if best is not None and bm >= max(4, int(0.10 * NS * len(OPPS) * 4)):
            cs = [310000 + gen * 53 + i for i in range(NS * 3)]
            a, b = _duel((best, champ, cs))
            if a >= b + max(4, int(0.05 * NS * 3 * len(OPPS) * 4)):
                champ, wins = best, wins + 1
                st = f"PROMOTE (+{bm} then +{a-b})"
        print(f"  gen {gen:4d} margin {bm:+3d} {st} promotions={wins} "
              f"({(time.time()-t0)/60:.0f}m)", flush=True)
        json.dump({"champion": {str(k): v for k, v in champ.items()},
                   "generation": gen, "promotions": wins,
                   "deck": to_deck(champ)}, open(CKPT, "w"), indent=1)


if __name__ == "__main__":
    main()
