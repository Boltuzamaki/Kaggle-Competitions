"""Paired policy ablations for router v10 against an extracted public opponent."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
 if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play
MAIN=os.path.join(ROOT,"agent","router_v10","main.py");DECK=[int(x) for x in open(os.path.join(ROOT,"agent","router_v10","deck.csv")) if x.strip()]
OPP=os.path.abspath(os.environ["REA_DIR"]);OM=os.path.join(OPP,"main.py");OD=[int(x) for x in open(os.path.join(OPP,"deck.csv")) if x.strip()]
def load(path,name):
 s=ilu.spec_from_file_location(name,path);m=ilu.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def chunk(seeds):
 os.chdir(OPP)
 if OPP not in sys.path:sys.path.insert(0,OPP)
 c=load(MAIN,f"rea_c_{os.getpid()}");b=load(MAIN,f"rea_b_{os.getpid()}");o=load(OM,f"rea_o_{os.getpid()}")
 orig=c.facing_terminal_mill_favorable_lineage
 def ablated(opp):
  if c.opponent_visible_ids(opp)&{57,169,190,666}:return False
  return orig(opp)
 c.facing_terminal_mill_favorable_lineage=ablated
 def ca(x):return DECK if x.get("select") is None else c.agent(x)
 def ba(x):return DECK if x.get("select") is None else b.agent(x)
 def oa(x):return OD if x.get("select") is None else o.agent(x)
 cw=bw=co=bo=err=0
 for seed in seeds:
  score=[]
  for policy in (ca,ba):
   w=0
   try:
    random.seed(seed);w+=play(seed,policy,oa,DECK,OD)==0
    random.seed(seed);w+=play(seed,oa,policy,OD,DECK)==1
   except Exception:err+=1
   score.append(w)
  cw+=score[0];bw+=score[1];co+=score[0]>score[1];bo+=score[1]>score[0]
 return cw,bw,co,bo,err
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 200;w=int(sys.argv[2]) if len(sys.argv)>2 else 8;s0=int(os.environ.get("REA_SEED0","48500000"));seeds=[s0+i for i in range(n)];tot=[0]*5
 with mp.get_context("fork").Pool(w) as pool:
  for row in pool.imap_unordered(chunk,[seeds[i::w] for i in range(w)]):tot=[a+b for a,b in zip(tot,row)]
 cw,bw,co,bo,e=tot;print(f"no_arch {cw}/{2*n} v10 {bw}/{2*n} disc {co}/{bo} p={mcnemar(co,bo):.6f} err={e}")
if __name__=="__main__":main()
