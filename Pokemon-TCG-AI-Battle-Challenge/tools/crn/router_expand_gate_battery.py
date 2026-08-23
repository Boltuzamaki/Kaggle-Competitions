"""Evolutionary one-gene expansion of router v4's terminal-mill allow-list."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
 if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play  # noqa:E402
import domain_policy,top_decks  # noqa:E402
MAIN=os.path.join(ROOT,"references","top_rankers","router_844","extracted","main.py")
DECK=[int(x) for x in open(os.path.join(ROOT,"references","top_rankers","router_844","extracted","deck.csv")) if x.strip()]
ARMS={
 "mega_fros":({848,849,861},top_decks.TD_09),
 "ogerpon":({96},top_decks.TD_00),
 "garchomp":({341,342,379,380,381},top_decks.TD_06),
 "slowking":({162,163},top_decks.TD_11),
 "td12_line":({117,344,345},top_decks.TD_12),
 "td15_line":({402,403,404,708,709,710},top_decks.TD_15),
 "td20_line":({650,651,652,708,709,710},top_decks.TD_20),
 "td21_line":({400,401,414,431,434},top_decks.TD_21),
}
def load(name):
 s=ilu.spec_from_file_location(name,MAIN);m=ilu.module_from_spec(s);s.loader.exec_module(m);return m
def chunk(args):
 arm,seeds=args;ids,opp_deck=ARMS[arm];cand=load(f"expand_{arm}_{os.getpid()}");base=load(f"base_{os.getpid()}")
 original=cand.facing_terminal_mill_favorable_lineage
 cand.facing_terminal_mill_favorable_lineage=lambda opp,original=original,ids=ids: original(opp) or bool(cand.opponent_visible_ids(opp)&ids)
 def ca(o):return DECK if o.get("select") is None else cand.agent(o)
 def ba(o):return DECK if o.get("select") is None else base.agent(o)
 if os.environ.get("REGB_OPP")=="v3":
  import arena,search_agent
  opponent=arena.bind_deck(search_agent.agent,opp_deck,search_agent)
 else:
  def opponent(o):return list(opp_deck) if o.get("select") is None else domain_policy.domain_agent(o,opp_deck)
 cw=bw=co=bo=0
 for seed in seeds:
  counts=[]
  for policy in (ca,ba):
   wins=0
   for seat in (0,1):
    random.seed(seed)
    if seat==0:wins+=int(play(seed,policy,opponent,DECK,opp_deck)==0)
    else:wins+=int(play(seed,opponent,policy,opp_deck,DECK)==1)
   counts.append(wins)
  cw+=counts[0];bw+=counts[1];co+=counts[0]>counts[1];bo+=counts[1]>counts[0]
 return arm,cw,bw,co,bo
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 100;workers=int(sys.argv[2]) if len(sys.argv)>2 else 12
 seed0=int(os.environ.get("REGB_SEED0","12500000"));seeds=[seed0+i for i in range(n)]
 arms=os.environ.get("REGB_ARMS",",").strip(",").split(",") if os.environ.get("REGB_ARMS") else list(ARMS)
 chunks=int(os.environ.get("REGB_CHUNKS","3"));jobs=[(a,seeds[i::chunks]) for a in arms for i in range(chunks)]
 agg={a:[0,0,0,0] for a in arms}
 with mp.get_context("fork").Pool(min(workers,len(jobs))) as pool:
  for arm,*row in pool.imap_unordered(chunk,jobs):agg[arm]=[a+b for a,b in zip(agg[arm],row)]
 for arm,(cw,bw,co,bo) in agg.items():print(f"{arm:10s} expanded {cw}/{2*n} v4 {bw}/{2*n} disc {co}/{bo} p={mcnemar(co,bo):.6f}")
if __name__=="__main__":main()
