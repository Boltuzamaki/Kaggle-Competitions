"""Which policy we own plays the Grimmsnarl deck BEST into our two worst cells?

Measured motivation: family_v2's gate stack fires on Garchomp/Archaludon/Crustle,
which is 7.0% of the real field. Meanwhile field_00 (30.6% share) sits at 62.5%
and field_01 (12.8%) at 40.0% -- 43% of the field in our weakest matchups, all of
it falling through to plain grim_base.

A gate is only worth adding if some policy genuinely beats grim_base in that
specific matchup, so this measures the candidates head-on before any gate is
written. Deck is held FIXED (Grimmsnarl) so a difference is attributable to the
policy, not the list. Opponents use the measured-best pilot from field_pilots.json.

Cells run isolated: the engine aborts at C++ level on some policy/deck pairs.
"""
from __future__ import annotations
import importlib.util as ilu, json, multiprocessing as mp, os, random, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT,"agent"), os.path.join(ROOT,"tools"),
          os.path.join(ROOT,"references","top_rankers")):
    if p not in sys.path: sys.path.insert(0,p)
from paired_eval import play  # noqa: E402

_ST = os.path.join(ROOT,"scratchpad","scrape_20260804","elo_stage")
FIELD = json.load(open(os.path.join(ROOT,"agent","field_decks.json")))
PILOTS = json.load(open(os.path.join(ROOT,"agent","field_pilots.json")))
CELLS = [c for c in os.environ.get("CE_CELLS","field_00,field_01,field_07").split(",") if c]
CANDS = [c for c in os.environ.get("CE_CANDS","family_v2_base,prob_v2,fork_m3000,grim_v2").split(",") if c]
SEEDS_N = int(os.environ.get("CE_SEEDS","40")); SEED0 = int(os.environ.get("CE_SEED0","662000"))
WORKERS = int(os.environ.get("CE_WORKERS","16"))
DECK = [int(x) for x in open(os.path.join(_ST,"family_v2_base","deck.csv")) if x.strip()]

def _load(pkgdir, deck):
    before=set(sys.modules); sys.path.insert(0,pkgdir)
    try:
        s=ilu.spec_from_file_location("ce_"+os.path.basename(pkgdir), os.path.join(pkgdir,"main.py"))
        m=ilu.module_from_spec(s); s.loader.exec_module(m)
    finally:
        try: sys.path.remove(pkgdir)
        except ValueError: pass
        for k in set(sys.modules)-before: sys.modules.pop(k,None)
    for a in ("TIME_BUDGET_S","TIME_BUDGET","SEARCH_TIME_BUDGET","SEARCH_TIME_BUDGET_S"):
        if hasattr(m,a):
            try: setattr(m,a,1e9 if a.endswith("_S") else 2.6)
            except Exception: pass
    return lambda o,_f=m.agent,_d=deck: list(_d) if o.get("select") is None else _f(o)

def _pilot(kind, deck):
    if kind=="domain":
        import domain_policy
        return lambda o,_d=deck: list(_d) if o.get("select") is None else domain_policy.domain_agent(o,_d)
    if kind=="ogerpon":
        import ogerpon_policy
        return lambda o,_d=deck: list(_d) if o.get("select") is None else ogerpon_policy.ogerpon_agent(o,_d)
    return _load(os.path.join(_ST, {"prob_v2":"prob_v2","grim":"grim_v2","fork":"fork_m3000"}.get(kind,kind)), deck)

def _cell(a,q):
    try: q.put(job(a))
    except Exception: pass

def job(args):
    cand, okey, seeds = args
    odeck = FIELD[okey]["deck"]
    me = _load(os.path.join(_ST,cand), DECK)          # candidate policy on OUR deck
    opp = _pilot(PILOTS.get(okey,"domain"), odeck)
    w=l=0
    for s in seeds:
        for seat in (0,1):
            random.seed(s)
            if seat==0:
                r=play(s,me,opp,DECK,odeck); w+=int(r==0); l+=int(r==1)
            else:
                r=play(s,opp,me,odeck,DECK); w+=int(r==1); l+=int(r==0)
    return cand, okey, w, l

def main():
    seeds=[SEED0+i for i in range(SEEDS_N)]
    K=max(1,WORKERS//max(len(CANDS)*len(CELLS),1))
    jobs=[(c,k,seeds[i::K]) for c in CANDS for k in CELLS for i in range(K)]
    print(f"cell-expert: {len(CANDS)} policies x {len(CELLS)} cells x {SEEDS_N} seeds x 2 seats",flush=True)
    ctx=mp.get_context("fork"); agg={}; pend,run=list(jobs),[]
    while pend or run:
        while pend and len(run)<WORKERS:
            a=pend.pop(0); q=ctx.Queue()
            pr=ctx.Process(target=_cell,args=(a,q),daemon=True); pr.start()
            run.append((pr,q,a,time.time()))
        time.sleep(0.4); keep=[]
        for pr,q,a,t0 in run:
            try: r=q.get_nowait()
            except Exception: r=None
            if r is not None:
                c,k,w,l=r; t=agg.setdefault((c,k),[0,0]); t[0]+=w; t[1]+=l; pr.join(timeout=1)
            elif not pr.is_alive(): print(f"    {a[0]} vs {a[1]} CRASH (cell missing)",flush=True)
            elif time.time()-t0>float(os.environ.get("CE_TIMEOUT","3000")):
                pr.terminate(); print(f"    {a[0]} vs {a[1]} TIMEOUT",flush=True)
            else: keep.append((pr,q,a,t0))
        run=keep
    print("\n===== POLICY x CELL (our deck fixed, opponent = measured best pilot) =====")
    for k in CELLS:
        base=agg.get(("family_v2_base",k),[0,0]); bn=sum(base) or 1
        print(f"\n  {k}  share {100*FIELD[k]['share']:5.2f}%   incumbent {100*base[0]/bn:5.1f}%")
        for c in CANDS:
            v=agg.get((c,k))
            if not v or sum(v)==0: print(f"      {c:16s} no data"); continue
            n=sum(v); d=100*v[0]/n - 100*base[0]/bn
            print(f"      {c:16s} {v[0]:3d}-{v[1]:<3d} ({100*v[0]/n:5.1f}%)  delta {d:+5.1f}")
    json.dump({f"{a}|{b}":v for (a,b),v in agg.items()},
              open(os.path.join(ROOT,"scratchpad","scrape_20260804","cell_expert.json"),"w"),indent=1)

if __name__=="__main__": main()
