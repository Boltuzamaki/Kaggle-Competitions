"""Memetic tuning of the forked baseline's 69 weights against our league."""
import json, os, random, sys, time
import multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,'agent'))
from paired_eval import play, build, mcnemar   # noqa
import fork_policy                             # noqa

DECK = fork_policy.DECK
OPPS = ["hybs-other","hybs-garchomp","td-td_08","meta-grimmsnarl"]

def _agent(genome):
    def f(obs):
        fork_policy.W.clear(); fork_policy.W.update(genome)
        return fork_policy.agent(obs)
    return f

def duel(ga, gb, opps, seeds):
    a=b=0; fa,fb=_agent(ga),_agent(gb)
    for fo,do in opps:
        for s in seeds:
            x=y=0
            for seat in (0,1):
                random.seed(s)
                x += int(play(s,fa,fo,DECK,do)==0) if seat==0 else int(play(s,fo,fa,do,DECK)==1)
            for seat in (0,1):
                random.seed(s)
                y += int(play(s,fb,fo,DECK,do)==0) if seat==0 else int(play(s,fo,fb,do,DECK)==1)
            if x>y: a+=1
            elif y>x: b+=1
    return a,b

def _worker(args):
    cand, champ, names, seeds = args
    import arena  # noqa
    opps=[build(n) for n in names]
    return duel(cand, champ, opps, seeds)

def mutate(g,rng,rate=0.30,scale=0.25):
    c=dict(g)
    for k in c:
        if rng.random()<rate: c[k]=max(1.0, c[k]*(1.0+rng.gauss(0,scale)))
    return c

if __name__=="__main__":
    import arena  # noqa
    gens=int(os.environ.get("FT_GENS","300")); pop=int(os.environ.get("FT_POP","8"))
    nseed=int(os.environ.get("FT_SEEDS","5")); workers=int(os.environ.get("FT_WORKERS","8"))
    names=[n for n in OPPS]
    opps=[build(n) for n in names]
    base={k: float(v) for k,v in fork_policy.W.items()}
    ck=os.path.join(ROOT,'scratchpad','scrape_20260804','fork_tune_ckpt.json')
    champ=dict(base); g0=0
    if os.path.exists(ck):
        try:
            d=json.load(open(ck)); champ=d['champion']; g0=d['generation']
            print('resumed at gen',g0)
        except Exception: pass
    print(f'{len(base)} fork weights | pop {pop} | {len(names)} opponents', flush=True)
    rng=random.Random(7777); t0=time.time(); wins=0
    for gen in range(g0+1,g0+gens+1):
        seeds=[420000+gen*89+i for i in range(nseed)]
        cands=[mutate(champ,rng) for _ in range(pop)]
        with mp.get_context("fork").Pool(processes=min(pop,workers)) as pool:
            res=pool.map(_worker, [(c,champ,names,seeds) for c in cands])
        best=None; bm=0
        for c,(a,b) in zip(cands,res):
            if a-b>bm: best,bm=c,a-b
        st='hold'
        if best is not None and bm>=2:
            cs=[880000+gen*37+i for i in range(nseed)]
            a,b=duel(best,champ,opps,cs)
            if a>b: champ=best; wins+=1; st=f'PROMOTE (+{bm} then +{a-b})'
        print(f'  gen {gen:4d} margin {bm:+3d} {st} promotions={wins} ({(time.time()-t0)/60:.0f}m)', flush=True)
        json.dump({'champion':champ,'generation':gen,'promotions':wins}, open(ck,'w'), indent=1)
    cs=[990000+i for i in range(40)]
    a,b=duel(champ,base,opps,cs); p=mcnemar(a,b)
    print(f'\nfinal: champion {a} / baseline {b}, p={p:.4f}')
    if a>b and p<0.05:
        json.dump(champ, open(os.path.join(ROOT,'agent','fork','alak_w.json'),'w'), indent=1)
        print('wrote agent/fork/alak_w.json -- needs SPRT before shipping')
