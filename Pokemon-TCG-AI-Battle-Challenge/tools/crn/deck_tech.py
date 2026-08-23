"""Count-only deck tech on OUR list, scored against the two cells that matter.

Motivation is measured: family_v2's gates cover 7.0% of the field while field_00
(30.6% share) sits at 46% and field_01 (12.8%) at 39% -- 43% of the field in our
worst matchups. Policy is exhausted (every alternative we own scores 1-5% piloting
this deck), and the search margin is inert (1.24% override rate). Deck count is
the one lever in this class that has ever produced a confirmed gain here
(-Tool Scrapper +4 Rare Candy, p=0.0427).

Count-only perturbations (-1 card A, +1 card B) are used deliberately: grim_base
has hardcoded per-card logic, so introducing a foreign card silently breaks the
pilot, whereas re-weighting cards it already understands does not.

Legality enforced as the engine actually applies it: 60 cards, max 4 copies of
anything EXCEPT basic energy (card 7 runs 10 copies in the base list). An earlier
search silently produced ZERO arms by wrongly capping basic energy at 4.
"""
from __future__ import annotations
import collections, importlib.util as ilu, itertools, json, multiprocessing as mp
import os, random, sys, time

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT,"agent"), os.path.join(ROOT,"tools"),
          os.path.join(ROOT,"references","top_rankers")):
    if p not in sys.path: sys.path.insert(0,p)
from paired_eval import play  # noqa: E402

_ST=os.path.join(ROOT,"scratchpad","scrape_20260804","elo_stage")
FIELD=json.load(open(os.path.join(ROOT,"agent","field_decks.json")))
PILOTS=json.load(open(os.path.join(ROOT,"agent","field_pilots.json")))
BASE_PKG=os.path.join(_ST,"family_v2_base")
BASE=[int(x) for x in open(os.path.join(BASE_PKG,"deck.csv")) if x.strip()]
CELLS=[c for c in os.environ.get("DT_CELLS","field_00,field_01").split(",") if c]
SEEDS_N=int(os.environ.get("DT_SEEDS","30")); SEED0=int(os.environ.get("DT_SEED0","713000"))
WORKERS=int(os.environ.get("DT_WORKERS","16"))
BASIC_ENERGY={7}

def variants():
    c=collections.Counter(BASE); out={"base":list(BASE)}
    ids=[k for k in c if k not in BASIC_ENERGY]
    for a,b in itertools.permutations(ids,2):
        if c[a]<1: continue
        if c[b]>=4 and b not in BASIC_ENERGY: continue
        d=collections.Counter(c); d[a]-=1; d[b]+=1
        if d[a]<0: continue
        deck=[k for k,n in sorted(d.items()) for _ in range(n)]
        if len(deck)!=60: continue
        if any(n>4 for k,n in d.items() if k not in BASIC_ENERGY): continue
        out[f"m{a}_p{b}"]=deck
    return out

def _load(pkgdir, deck):
    before=set(sys.modules); sys.path.insert(0,pkgdir)
    try:
        s=ilu.spec_from_file_location("dt_"+os.path.basename(pkgdir), os.path.join(pkgdir,"main.py"))
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

def _pilot(kind,deck):
    if kind=="domain":
        import domain_policy
        return lambda o,_d=deck: list(_d) if o.get("select") is None else domain_policy.domain_agent(o,_d)
    if kind=="ogerpon":
        import ogerpon_policy
        return lambda o,_d=deck: list(_d) if o.get("select") is None else ogerpon_policy.ogerpon_agent(o,_d)
    return _load(os.path.join(_ST,{"prob_v2":"prob_v2","grim":"grim_v2","fork":"fork_m3000"}.get(kind,kind)),deck)

def _cell(a,q):
    try: q.put(job(a))
    except Exception: pass

def job(args):
    name,deck,okey,seeds=args
    odeck=FIELD[okey]["deck"]
    me=_load(BASE_PKG,deck); opp=_pilot(PILOTS.get(okey,"domain"),odeck)
    w=l=0
    for s in seeds:
        for seat in (0,1):
            random.seed(s)
            if seat==0:
                r=play(s,me,opp,deck,odeck); w+=int(r==0); l+=int(r==1)
            else:
                r=play(s,opp,me,odeck,deck); w+=int(r==1); l+=int(r==0)
    return name,okey,w,l

def main():
    V=variants(); seeds=[SEED0+i for i in range(SEEDS_N)]
    keys=list(V)
    # DT_ONLY re-tests a named shortlist on FRESH seeds. The 145-arm sweep is a
    # selection contest: a 120-game cell has SD~4.6%, so the best of 145 noise
    # draws lands near +13% by construction. Only replication on disjoint seeds
    # separates a real effect from the winner's curse, and the full panel is used
    # so a gain in field_00 that guts field_02/06 is visible rather than hidden.
    only=[k for k in os.environ.get("DT_ONLY","").split(",") if k]
    if only: keys=[k for k in keys if k in only or k=="base"]
    lim=int(os.environ.get("DT_LIMIT","0"))
    if lim: keys=keys[:lim]
    print(f"deck tech: {len(keys)} legal count-variants x {len(CELLS)} cells x {SEEDS_N} seeds x 2 seats",flush=True)
    jobs=[(k,V[k],c,seeds) for k in keys for c in CELLS]
    ctx=mp.get_context("fork"); agg={}; pend,run=list(jobs),[]
    t_start=time.time()
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
                n,k,w,l=r; agg[(n,k)]=(w,l); pr.join(timeout=1)
            elif not pr.is_alive(): print(f"    {a[0]} vs {a[2]} CRASH (not counted)",flush=True)
            elif time.time()-t0>float(os.environ.get("DT_TIMEOUT","2400")):
                pr.terminate(); print(f"    {a[0]} vs {a[2]} TIMEOUT",flush=True)
            else: keep.append((pr,q,a,t0))
        run=keep
        if len(agg)%20==0 and agg:
            json.dump({f"{a}|{b}":v for (a,b),v in agg.items()},
                      open(os.path.join(ROOT,"scratchpad","scrape_20260804","deck_tech.json"),"w"),indent=1)
    # share-weighted over the tested cells only
    def sc(n):
        num=den=0.0
        for c in CELLS:
            v=agg.get((n,c))
            if not v or sum(v)==0: return None
            num+=FIELD[c]["share"]*v[0]/sum(v); den+=FIELD[c]["share"]
        return 100*num/den if den else None
    b=sc("base")
    print(f"\n===== DECK TECH ({'+'.join(CELLS)}, grim pilot) =====")
    print(f"  base {b:.1f}%" if b is not None else "  base incomplete")
    rows=[(sc(n),n) for n in keys if n!="base" and sc(n) is not None]
    for s,n in sorted(rows,reverse=True)[:25]:
        print(f"    {n:16s} {s:5.1f}%   delta {s-(b or 0):+5.1f}"
              + "".join(f"   {c}:{agg[(n,c)][0]}-{agg[(n,c)][1]}" for c in CELLS),flush=True)
    json.dump({f"{a}|{b_}":v for (a,b_),v in agg.items()},
              open(os.path.join(ROOT,"scratchpad","scrape_20260804","deck_tech.json"),"w"),indent=1)
    print(f"\n  elapsed {time.time()-t_start:.0f}s",flush=True)

if __name__=="__main__": main()
