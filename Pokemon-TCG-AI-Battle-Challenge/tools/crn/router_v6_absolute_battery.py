"""Broad CRN comparison: router v6 versus the exact submitted 844 behavior."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
    if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play
import domain_policy,top_decks
LABEL=os.environ.get("RV6A_LABEL","v6")
V6=os.environ.get("RV6A_MAIN",os.path.join(ROOT,"agent","router_v6","main.py"))
BASE=os.environ.get("RV6A_BASE_MAIN",os.path.join(ROOT,"references","top_rankers","router_844","extracted","main.py"))
DECK=[int(x) for x in open(os.path.join(ROOT,"agent","router_v6","deck.csv")) if x.strip()]
_indices=[int(x) for x in os.environ.get("RV6A_DECKS",",").strip(",").split(",")] if os.environ.get("RV6A_DECKS") else list(range(int(os.environ.get("RV6A_NDECKS","12"))))
PANEL={f"td{i:02d}":getattr(top_decks,f"TD_{i:02d}") for i in _indices}
def load(path,name):
 s=ilu.spec_from_file_location(name,path);m=ilu.module_from_spec(s);s.loader.exec_module(m);return m
def chunk(seeds):
 c=load(V6,f"v6abs_{os.getpid()}");b=load(BASE,f"r844_{os.getpid()}")
 if not os.environ.get("RV6A_KEEP_BASE_GATE"):
  b.facing_terminal_mill_favorable_lineage=lambda opponent:False
 def ca(o):return DECK if o.get("select") is None else c.agent(o)
 def ba(o):return DECK if o.get("select") is None else b.agent(o)
 out={}
 for name,odeck in PANEL.items():
  if os.environ.get("RV6A_OPP")=="v3":
   import arena,search_agent
   opp=arena.bind_deck(search_agent.agent,odeck,search_agent)
  else:
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
 n=int(sys.argv[1]) if len(sys.argv)>1 else 60;workers=int(sys.argv[2]) if len(sys.argv)>2 else 12
 seed0=int(os.environ.get("RV6A_SEED0","22500000"));seeds=[seed0+i for i in range(n)]
 jobs=[seeds[i::workers] for i in range(workers)];agg={k:[0,0,0,0] for k in PANEL}
 with mp.get_context("fork").Pool(workers) as pool:
  for rows in pool.imap_unordered(chunk,jobs):
   for k,row in rows.items():agg[k]=[a+b for a,b in zip(agg[k],row)]
 pooled=[0,0,0,0]
 for k,row in agg.items():
  pooled=[a+b for a,b in zip(pooled,row)];cw,bw,co,bo=row
  print(f"{k} {LABEL} {cw}/{2*n} r844 {bw}/{2*n} disc {co}/{bo} p={mcnemar(co,bo):.5f}")
 cw,bw,co,bo=pooled;print(f"POOLED {LABEL} {cw} r844 {bw} disc {co}/{bo} p={mcnemar(co,bo):.8f}")
if __name__=="__main__":main()
