"""Score our agent against the field WE ACTUALLY FACE (rating band 700-880).

The panel every prior experiment used was mined at >=1100. Measured overlap with
our own band's top-10 is 2/10, and 39% of our band's share consists of decks that
panel never contained. That is a mechanical explanation for why local gains have
not transferred: we were optimising against someone else's metagame.

Opponents are piloted by grim, which pilot_select measured as the best general
pilot we own (it won 4 of 8 field decks, beating each deck's own specialist), so
a foreign list is driven competently rather than at the 1-5% our deck-specific
policies manage. Cells are isolated because the engine aborts at C++ level on some
deck/policy pairs.
"""
from __future__ import annotations
import importlib.util as ilu, json, multiprocessing as mp, os, random, sys, time

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(ROOT,"agent"), os.path.join(ROOT,"tools"),
          os.path.join(ROOT,"references","top_rankers")):
    if p not in sys.path: sys.path.insert(0,p)
from paired_eval import play  # noqa: E402

_ST=os.path.join(ROOT,"scratchpad","scrape_20260804","elo_stage")
BAND=os.environ.get("BE_BAND","ours_700_880")
F=json.load(open(os.path.join(ROOT,"agent","field_decks_by_band.json")))[BAND]
SUBJ=[s for s in os.environ.get("BE_SUBJECTS","family_v2_base").split(",") if s]
CELLS=[k for k in F][:int(os.environ.get("BE_TOPK","10"))]
SEEDS_N=int(os.environ.get("BE_SEEDS","30")); SEED0=int(os.environ.get("BE_SEED0","1450000"))
WORKERS=int(os.environ.get("BE_WORKERS","16"))
OPP_PKG=os.environ.get("BE_OPP","grim_v2")

def _load(pkgdir, deck):
    before=set(sys.modules); sys.path.insert(0,pkgdir)
    try:
        s=ilu.spec_from_file_location("be_"+os.path.basename(pkgdir), os.path.join(pkgdir,"main.py"))
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

def _cell(a,q):
    try: q.put(job(a))
    except Exception: pass

def job(args):
    subj,okey,seeds=args
    sdeck=[int(x) for x in open(os.path.join(_ST,subj,"deck.csv")) if x.strip()]
    odeck=F[okey]["deck"]
    me=_load(os.path.join(_ST,subj),sdeck); opp=_load(os.path.join(_ST,OPP_PKG),odeck)
    w=l=0
    for s in seeds:
        for seat in (0,1):
            random.seed(s)
            if seat==0:
                r=play(s,me,opp,sdeck,odeck); w+=int(r==0); l+=int(r==1)
            else:
                r=play(s,opp,me,odeck,sdeck); w+=int(r==1); l+=int(r==0)
    return subj,okey,w,l

def main():
    seeds=[SEED0+i for i in range(SEEDS_N)]
    jobs=[(s,c,seeds) for s in SUBJ for c in CELLS]
    tot=sum(F[c]["share"] for c in CELLS)
    print(f"band eval [{BAND}]: {len(SUBJ)} subjects x {len(CELLS)} cells "
          f"({100*tot:.0f}% of that band) x {SEEDS_N} seeds x 2 seats",flush=True)
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
                s,k,w,l=r; agg[(s,k)]=(w,l); pr.join(timeout=1)
            elif not pr.is_alive(): print(f"    {a[0]} vs {a[1]} CRASH (excluded, not a pass)",flush=True)
            elif time.time()-t0>float(os.environ.get("BE_TIMEOUT","2400")):
                pr.terminate(); print(f"    {a[0]} vs {a[1]} TIMEOUT",flush=True)
            else: keep.append((pr,q,a,t0))
        run=keep
    print(f"\n===== BAND {BAND} =====")
    for s in SUBJ:
        num=den=0.0; miss=[]
        print(f"\n  {s}")
        for c in CELLS:
            v=agg.get((s,c))
            if not v or sum(v)==0: miss.append(c); continue
            wr=v[0]/sum(v); num+=F[c]["share"]*wr; den+=F[c]["share"]
            print(f"      {c} share {100*F[c]['share']:5.2f}%  {v[0]:3d}-{v[1]:<3d} {100*wr:5.1f}%",flush=True)
        print(f"    => band-weighted {100*num/den if den else 0:5.1f}%"
              + (f"   MISSING {miss}" if miss else ""),flush=True)
    json.dump({f"{a}|{b}":v for (a,b),v in agg.items()},
              open(os.path.join(ROOT,"scratchpad","scrape_20260804","band_eval.json"),"w"),indent=1)

if __name__=="__main__": main()
