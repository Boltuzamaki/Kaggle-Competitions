"""CRN evaluation of the public former-5th-place Alakazam against m3000."""
from __future__ import annotations
import importlib.util as ilu, multiprocessing as mp, os, random, sys
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
    if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play  # noqa:E402
import deep_search  # noqa:E402
PUB_DIR=os.path.abspath(os.environ.get("PCE_DIR",os.path.join(ROOT,"references","top_rankers","alakazam_5th","output")))
LABEL=os.environ.get("PCE_LABEL","public-alak5")
PUB_MAIN=os.path.join(PUB_DIR,"main.py"); PUB_DECK=[int(x) for x in open(os.path.join(PUB_DIR,"deck.csv")) if x.strip()]
FORK_MAIN=os.path.join(ROOT,"agent","fork","fork_main.py"); FORK_DECK=[int(x) for x in open(os.path.join(ROOT,"agent","fork","deck.csv")) if x.strip()]
def load(path,name):
 s=ilu.spec_from_file_location(name,path);m=ilu.module_from_spec(s);s.loader.exec_module(m);return m
def chunk(seeds):
 os.chdir(PUB_DIR);pub=load(PUB_MAIN,f"alak5_{os.getpid()}");fork=load(FORK_MAIN,f"fork_{os.getpid()}")
 fork.TIME_BUDGET_S=1e9;deep_search.install(fork,extra_turns=0,margin=3000.0)
 def pa(o):return PUB_DECK if o.get("select") is None else pub.agent(o)
 def fa(o):return FORK_DECK if o.get("select") is None else fork.agent(o)
 pw=fw=po=fo=0
 for seed in seeds:
  counts=[]
  for cand,deck in ((pa,PUB_DECK),(fa,FORK_DECK)):
   wins=0
   for seat in (0,1):
    random.seed(seed)
    if seat==0:wins+=int(play(seed,cand,fa,deck,FORK_DECK)==0)
    else:wins+=int(play(seed,fa,cand,FORK_DECK,deck)==1)
   counts.append(wins)
  pw+=counts[0];fw+=counts[1];po+=counts[0]>counts[1];fo+=counts[1]>counts[0]
 return pw,fw,po,fo
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 80;w=int(sys.argv[2]) if len(sys.argv)>2 else 8
 seed0=int(os.environ.get("PA5_SEED0","12000000"));seeds=[seed0+i for i in range(n)]
 total=[0]*4
 with mp.get_context("fork").Pool(w) as pool:
  for row in pool.imap_unordered(chunk,[seeds[i::w] for i in range(w)]):total=[a+b for a,b in zip(total,row)]
 pw,fw,po,fo=total;print(f"{LABEL} {pw}/{2*n} fork-m3000 {fw}/{2*n}");print(f"discordant {po}/{fo} p={mcnemar(po,fo):.6f}")
if __name__=="__main__":main()
