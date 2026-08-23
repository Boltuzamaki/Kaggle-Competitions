"""Parallel sweep over explicit tempo (attack-timing) rules, paired vs baseline.

Every config plays the SAME seeds against the SAME opponents as the untuned
baseline, and we count discordant seeds -- so the comparison shares shuffles and
flips and is far more sensitive than raw win-rate. Configs are evaluated in
parallel across cores.
"""
import itertools, json, os, random, sys, time
import multiprocessing as mp

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0,HERE); sys.path.insert(0,os.path.join(ROOT,'agent'))
from paired_eval import play, build, mcnemar   # noqa
import hybrid_agent, meta_decks                # noqa

DECK = meta_decks.OTHER
OPPS = ["public-archaludon","public-alakazam","td-td_08"]
SEEDS = [310000+i for i in range(int(os.environ.get("TS_SEEDS","14")))]

GRID = [dict(zip(("attack_min_energy","attack_min_frac","setup_turns","attack_penalty"), v))
        for v in itertools.product([0,2,3,4,5],[0.0,0.4,0.7],[0,2,4],[900.0])]

def _run(cfg):
    import tempo_policy, arena           # noqa
    tempo_policy.T.update(cfg)
    def cand(obs):
        return hybrid_agent.hybrid_agent(obs, DECK, policy_module=tempo_policy)
    def base(obs):
        import ogerpon_policy
        return hybrid_agent.hybrid_agent(obs, DECK, policy_module=ogerpon_policy)
    a_only=b_only=cw=bw_=0
    for on in OPPS:
        fo,do=build(on)
        for s in SEEDS:
            ga=gb=0
            for seat in (0,1):
                random.seed(s)
                ga += int(play(s,cand,fo,DECK,do)==0) if seat==0 else int(play(s,fo,cand,do,DECK)==1)
            for seat in (0,1):
                random.seed(s)
                gb += int(play(s,base,fo,DECK,do)==0) if seat==0 else int(play(s,fo,base,do,DECK)==1)
            cw+=ga; bw_+=gb
            if ga>gb: a_only+=1
            elif gb>ga: b_only+=1
    return cfg, a_only, b_only, cw, bw_

if __name__=="__main__":
    t0=time.time()
    print(f"{len(GRID)} tempo configs x {len(OPPS)} opponents x {len(SEEDS)} seeds", flush=True)
    with mp.get_context("fork").Pool(processes=int(os.environ.get("TS_WORKERS","12"))) as pool:
        out=pool.map(_run, GRID)
    rows=[]
    for cfg,a,b,cw,bw_ in out:
        rows.append({"cfg":cfg,"a":a,"b":b,"cand_wins":cw,"base_wins":bw_,"p":mcnemar(a,b)})
    rows.sort(key=lambda r:-(r["a"]-r["b"]))
    print(f"\ndone in {(time.time()-t0)/60:.0f}m -- ranked by paired margin\n")
    for r in rows[:14]:
        c=r["cfg"]
        print(f"  minE={c['attack_min_energy']} frac={c['attack_min_frac']} setup={c['setup_turns']}"
              f"  discordant {r['a']}/{r['b']}  wins {r['cand_wins']} vs {r['base_wins']}  p={r['p']:.3f}")
    json.dump(rows, open(os.path.join(ROOT,'scratchpad','scrape_20260804','tempo_sweep.json'),'w'), indent=1)
