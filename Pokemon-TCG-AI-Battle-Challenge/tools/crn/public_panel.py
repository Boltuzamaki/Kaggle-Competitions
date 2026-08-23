"""Paired CRN evaluation against REAL public agents piloting their own decks.

Why this exists: band_eval's opponents are our own policy piloting mined field
decks, and our policy on a foreign deck scores 26-38%. So band_eval measures
"beats our impersonation of the field". Ogerpon led +8.2 there and sat ~190
ladder points behind. These opponents are the actual published agents, so a win
here means beating a real agent, not a strawman.

Env: PP_SUBJECTS, PP_OPPS, PP_SEEDS, PP_WORKERS.
Each cell runs in its own process -- the engine aborts at C++ level
("buffer full. capacity:7") when too many battles share one interpreter.
"""
from __future__ import annotations
import importlib.util as ilu, multiprocessing as mp, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT, "agent"), os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
from paired_eval import play  # noqa: E402

_ST = os.path.join(ROOT, "scratchpad", "scrape_20260804", "elo_stage")
_TR = os.path.join(ROOT, "references", "top_rankers")

SUBJ = [s for s in os.environ.get("PP_SUBJECTS", "family_v2_base").split(",") if s]
OPPS = [s for s in os.environ.get("PP_OPPS", "baseline_1084,alakazam_5th,archaludon_starmie").split(",") if s]
SEEDS_N = int(os.environ.get("PP_SEEDS", "40"))
SEED0 = int(os.environ.get("PP_SEED0", "1450000"))
WORKERS = int(os.environ.get("PP_WORKERS", "16"))


_REFS = os.path.join(ROOT, "references")


def _dirs(name):
    """Subjects live in elo_stage; real public agents live under references/*.

    A package qualifies only if it has BOTH main.py and deck.csv -- several
    reference dirs carry one without the other and would silently load a
    mismatched deck.
    """
    cand = [os.path.join(_ST, name),
            os.path.join(_TR, name, "output"),
            os.path.join(_TR, name, "extracted"),
            os.path.join(_TR, name)]
    for sub in ("competitor_refresh", "latest_public", "public_sim_repo", "top_rankers"):
        base = os.path.join(_REFS, sub, name)
        cand += [base, os.path.join(base, "extracted"), os.path.join(base, "output")]
    for d in cand:
        if os.path.isfile(os.path.join(d, "main.py")) and os.path.isfile(os.path.join(d, "deck.csv")):
            return d
    raise FileNotFoundError(name)


def _load(pkgdir):
    deck = [int(x) for x in open(os.path.join(pkgdir, "deck.csv")) if x.strip()]
    before = set(sys.modules)
    sys.path.insert(0, pkgdir)
    try:
        s = ilu.spec_from_file_location("pp_" + os.path.basename(os.path.dirname(pkgdir)),
                                        os.path.join(pkgdir, "main.py"))
        m = ilu.module_from_spec(s)
        s.loader.exec_module(m)
    finally:
        try:
            sys.path.remove(pkgdir)
        except ValueError:
            pass
        for k in set(sys.modules) - before:
            sys.modules.pop(k, None)
    for a in ("TIME_BUDGET_S", "TIME_BUDGET", "SEARCH_TIME_BUDGET", "SEARCH_TIME_BUDGET_S"):
        if hasattr(m, a):
            try:
                setattr(m, a, 1e9 if a.endswith("_S") else 2.6)
            except Exception:
                pass
    fn = m.agent
    return (lambda o, _f=fn, _d=deck: list(_d) if o.get("select") is None else _f(o)), deck


def job(args):
    subj, opp, seeds = args
    try:
        me, sdeck = _load(_dirs(subj))
        him, odeck = _load(_dirs(opp))
    except Exception as e:
        return subj, opp, 0, 0, f"load:{type(e).__name__}"
    w = l = 0
    for s in seeds:
        for seat in (0, 1):
            random.seed(s)
            try:
                if seat == 0:
                    r = play(s, me, him, sdeck, odeck); w += int(r == 0); l += int(r == 1)
                else:
                    r = play(s, him, me, odeck, sdeck); w += int(r == 1); l += int(r == 0)
            except Exception:
                pass
    return subj, opp, w, l, ""


def main():
    # One cell per subprocess (driver uses xargs -P). multiprocessing.Queue
    # deadlocked here: the engine library is already loaded in the parent, and
    # forked children inherited it in a state where they never made progress
    # (all workers sat at cpu=00:00:00).
    if len(sys.argv) == 3:
        seeds = [SEED0 + i for i in range(SEEDS_N)]
        s, o, w, l, err = job((sys.argv[1], sys.argv[2], seeds))
        print(f"CELL\t{s}\t{o}\t{w}\t{l}\t{err}", flush=True)
        return
    res = {}
    for line in sys.stdin:
        f = line.rstrip("\n").split("\t")
        if len(f) >= 5 and f[0] == "CELL":
            res.setdefault(f[1], {})[f[2]] = (int(f[3]), int(f[4]), f[5] if len(f) > 5 else "")
    print("\n===== vs REAL PUBLIC AGENTS =====", flush=True)
    for s in SUBJ:
        tw = tl = 0
        print(f"\n  {s}")
        for o in OPPS:
            w, l, err = res.get(s, {}).get(o, (0, 0, "missing"))
            n = w + l
            tw += w; tl += l
            print(f"      vs {o:22s} {w:4d}-{l:<4d} {100.0*w/n if n else 0:5.1f}%  {err}", flush=True)
        n = tw + tl
        print(f"    => overall {tw}-{tl}  {100.0*tw/n if n else 0:.1f}%", flush=True)


if __name__ == "__main__":
    main()
