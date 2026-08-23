"""SPRT-style sweep of the fork's two search knobs against our league.

It ships with K_OPP=3 and TIME_BUDGET_S=0.8 -- 0.8 seconds of a 600 second
per-game budget. Both are one-line changes with obvious headroom, and neither
requires touching the agent's logic. Paired on shared seeds vs the stock config.
"""
import itertools, json, os, random, sys, time
import multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,'agent'))
from paired_eval import play, build, mcnemar   # noqa

OPPS=["hybs-other","td-td_08","meta-grimmsnarl"]
SEEDS=[730000+i for i in range(int(os.environ.get("FK_SEEDS","12")))]
GRID=[(k,b) for k,b in itertools.product([3,5,8,12],[0.8,2.0,5.0])]

def _run(cfg):
    k,b=cfg
    import fork_policy, arena  # noqa
    fork_policy.set_knobs(k_opp=k, budget=b)
    cand=fork_policy.agent; deck=fork_policy.DECK
    import importlib.util as ilu
    spec=ilu.spec_from_file_location('fm_base', os.path.join(ROOT,'agent','fork','fork_main.py'))
    fmb=ilu.module_from_spec(spec); spec.loader.exec_module(fmb)   # stock config
    a=bn=0; cw=bw=0; t0=time.time()
    for on in OPPS:
        fo,do=build(on)
        for s in SEEDS:
            x=y=0
            for seat in (0,1):
                random.seed(s)
                x += int(play(s,cand,fo,deck,do)==0) if seat==0 else int(play(s,fo,cand,do,deck)==1)
            for seat in (0,1):
                random.seed(s)
                y += int(play(s,fmb.agent,fo,deck,do)==0) if seat==0 else int(play(s,fo,fmb.agent,do,deck)==1)
            cw+=x; bw+=y
            if x>y: a+=1
            elif y>x: bn+=1
    return cfg,a,bn,cw,bw,(time.time()-t0)/(len(OPPS)*len(SEEDS)*4)

if __name__=="__main__":
    print(f"{len(GRID)} knob configs x {len(OPPS)} opponents x {len(SEEDS)} seeds", flush=True)
    with mp.get_context("fork").Pool(processes=int(os.environ.get("FK_WORKERS","6"))) as pool:
        out=pool.map(_run, GRID)
    rows=[{"k_opp":c[0],"budget":c[1],"a":a,"b":b,"cand":cw,"base":bw,"sec_per_game":sp,
           "p":mcnemar(a,b)} for c,a,b,cw,bw,sp in out]
    rows.sort(key=lambda r:-(r["a"]-r["b"]))
    print("\nranked by paired margin (positive = better than stock):\n")
    for r in rows:
        print(f"  K_OPP={r['k_opp']:2d} budget={r['budget']:4.1f}s  discordant {r['a']}/{r['b']}"
              f"  wins {r['cand']} vs {r['base']}  p={r['p']:.3f}  {r['sec_per_game']:.2f}s/game")
    json.dump(rows, open(os.path.join(ROOT,'scratchpad','scrape_20260804','fork_knobs.json'),'w'), indent=1)
