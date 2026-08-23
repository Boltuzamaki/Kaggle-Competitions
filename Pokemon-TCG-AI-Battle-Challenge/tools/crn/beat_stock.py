"""Evolve the fork's 69 weights to BEAT THE STOCK FORK, paired on shared seeds.

Everyone who copies the public notebook runs the stock weights. So the objective
that actually gains rank is "beat stock", and the cheapest, most direct way to
measure it is to play stock head-to-head. It is also ~6x faster than tuning
against our own hybrid agents (0.8 s/game vs 5-7 s/game), which is what made the
earlier runs take 24 minutes per generation.

Paired CRN selection + disjoint-seed confirmation before any promotion.
"""
import json, os, random, sys, time
import multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,'agent'))
from paired_eval import play, mcnemar          # noqa
import importlib.util as ilu

FP=os.path.join(ROOT,'agent','fork','fork_main.py')
DECK=[int(x) for x in open(os.path.join(ROOT,'agent','fork','deck.csv')) if x.strip()]

def load():
    s=ilu.spec_from_file_location('fm_'+str(os.getpid())+str(random.random()), FP)
    m=ilu.module_from_spec(s); s.loader.exec_module(m); return m

_CAND=None; _STOCK=None
def _init():
    global _CAND,_STOCK
    _CAND=load(); _STOCK=load()

def _duel(args):
    genome, seeds = args
    global _CAND,_STOCK
    if _CAND is None: _init()
    _CAND.WEIGHTS.clear(); _CAND.WEIGHTS.update(genome)
    a=b=0
    for s in seeds:
        x=y=0
        for seat in (0,1):
            random.seed(s)
            r = play(s,_CAND.agent,_STOCK.agent,DECK,DECK) if seat==0 else \
                play(s,_STOCK.agent,_CAND.agent,DECK,DECK)
            x += int(r==(0 if seat==0 else 1))
            y += int(r==(1 if seat==0 else 0))
        if x>y: a+=1
        elif y>x: b+=1
    return a,b

def mutate(g,rng,rate=None,scale=None):
    rate = float(os.environ.get('BS_RATE','0.30')) if rate is None else rate
    scale = float(os.environ.get('BS_SCALE','0.25')) if scale is None else scale
    c=dict(g)
    for k in c:
        if rng.random()<rate: c[k]=max(1.0,c[k]*(1.0+rng.gauss(0,scale)))
    return c

if __name__=="__main__":
    gens=int(os.environ.get("BS_GENS","400")); pop=int(os.environ.get("BS_POP","14"))
    ns=int(os.environ.get("BS_SEEDS","8")); wk=int(os.environ.get("BS_WORKERS","14"))
    fm=load(); base={k:float(v) for k,v in fm.WEIGHTS.items()}
    ck=os.path.join(ROOT,'scratchpad','scrape_20260804',os.environ.get('BS_CKPT','beat_stock_ckpt.json'))
    champ=dict(base); g0=0; wins=0
    if os.path.exists(ck):
        try:
            d=json.load(open(ck)); champ=d['champion']; g0=d['generation']; wins=d.get('promotions',0)
            print('resumed gen',g0,flush=True)
        except Exception: pass
    print(f'{len(base)} weights | pop {pop} | objective = beat STOCK fork', flush=True)
    rng=random.Random(int(os.environ.get('BS_RNG','31337'))); t0=time.time()
    with mp.get_context("fork").Pool(processes=wk, initializer=_init) as pool:
        for gen in range(g0+1,g0+gens+1):
            seeds=[610000+gen*53+i for i in range(ns)]
            cands=[mutate(champ,rng) for _ in range(pop)]
            res=pool.map(_duel,[(c,seeds) for c in cands])
            best=None; bm=0
            for c,(a,b) in zip(cands,res):
                if a-b>bm: best,bm=c,a-b
            st='hold'
            # Champion B had 9 "promotions" and then LOST on fresh seeds (54-66).
            # Cause: 16-game evaluations over 69 dimensions -- a mutation looks
            # better by chance about half the time. Require a bigger margin AND a
            # 4x larger disjoint confirmation with a real win, not just a>b.
            if best is not None and bm >= max(3, int(0.25*ns)):
                cs=[750000+gen*41+i for i in range(ns*4)]
                a,b=_duel((best,cs))
                if a >= b + max(3, int(0.15*ns*4)):
                    champ=best; wins+=1; st=f'PROMOTE (+{bm} then +{a-b})'
                    json.dump(champ,open(os.path.join(ROOT,'agent','fork',os.environ.get('BS_CAND','alak_w_cand.json')),'w'),indent=1)
            print(f'  gen {gen:4d} margin {bm:+3d} {st} promo={wins} ({(time.time()-t0)/60:.0f}m)',flush=True)
            json.dump({'champion':champ,'generation':gen,'promotions':wins},open(ck,'w'),indent=1)
