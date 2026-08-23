"""Test removing inherited favorable lineages from router v6."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
    if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play
import domain_policy,top_decks
MAIN=os.path.join(ROOT,"agent","router_v6","main.py")
DECK=[int(x) for x in open(os.path.join(ROOT,"agent","router_v6","deck.csv")) if x.strip()]
PANELS={
 "drag":{"td03":top_decks.TD_03,"td05":top_decks.TD_05,"td19":top_decks.TD_19,"td23":top_decks.TD_23},
 "other":{"grim":top_decks.TD_01,"alak":top_decks.TD_02,"fros":top_decks.TD_09,"festival":top_decks.TD_10,
          "slow":top_decks.TD_11,"rocket":top_decks.TD_21},
}
def load(name):
 s=ilu.spec_from_file_location(name,MAIN);m=ilu.module_from_spec(s);s.loader.exec_module(m);return m
def chunk(args):
 panel,seeds=args;c=load(f"abl_{os.getpid()}");b=load(f"v6_{os.getpid()}")
 c.TERMINAL_MILL_FAVORABLE_IDS.difference_update({119,120,121})
 def ca(o):return DECK if o.get("select") is None else c.agent(o)
 def ba(o):return DECK if o.get("select") is None else b.agent(o)
 out={}
 for name,odeck in PANELS[panel].items():
  def opp(o,d=odeck):return list(d) if o.get("select") is None else domain_policy.domain_agent(o,d)
  cw=bw=co=bo=0
  for seed in seeds:
   score=[]
   for policy in (ca,ba):
    w=0
    for seat in (0,1):
     random.seed(seed)
     if seat==0:w+=int(play(seed,policy,opp,DECK,odeck)==0)
     else:w+=int(play(seed,opp,policy,odeck,DECK)==1)
    score.append(w)
   cw+=score[0];bw+=score[1];co+=score[0]>score[1];bo+=score[1]>score[0]
  out[name]=(cw,bw,co,bo)
 return out
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 100;workers=int(sys.argv[2]) if len(sys.argv)>2 else 8
 panel=os.environ.get("RV6B_PANEL","drag");seed0=int(os.environ.get("RV6B_SEED0","23500000"));seeds=[seed0+i for i in range(n)]
 jobs=[(panel,seeds[i::workers]) for i in range(workers)];agg={k:[0,0,0,0] for k in PANELS[panel]}
 with mp.get_context("fork").Pool(workers) as pool:
  for rows in pool.imap_unordered(chunk,jobs):
   for k,row in rows.items():agg[k]=[a+b for a,b in zip(agg[k],row)]
 pooled=[0,0,0,0]
 for k,row in agg.items():
  pooled=[a+b for a,b in zip(pooled,row)];cw,bw,co,bo=row
  print(f"{k:10s} no_drag {cw}/{2*n} v6 {bw}/{2*n} disc {co}/{bo} p={mcnemar(co,bo):.6f}")
 cw,bw,co,bo=pooled;print(f"POOLED no_drag {cw} v6 {bw} disc {co}/{bo} p={mcnemar(co,bo):.6f}")
if __name__=="__main__":main()
