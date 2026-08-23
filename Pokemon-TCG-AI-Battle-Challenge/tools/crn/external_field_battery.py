"""Paired field audit of an extracted public agent against router v8."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
 if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play
import domain_policy,top_decks
PUB=os.path.abspath(os.environ["EFB_DIR"]);LABEL=os.environ.get("EFB_LABEL",os.path.basename(os.path.dirname(PUB)))
PM=os.path.join(PUB,"main.py");PD=[int(x) for x in open(os.path.join(PUB,"deck.csv")) if x.strip()]
RM=os.path.join(ROOT,"agent","router_v8","main.py");RD=[int(x) for x in open(os.path.join(ROOT,"agent","router_v8","deck.csv")) if x.strip()]
PANEL={f"td{i:02d}":getattr(top_decks,f"TD_{i:02d}") for i in range(int(os.environ.get("EFB_NDECKS","13")))}
def load(path,name):
 s=ilu.spec_from_file_location(name,path);m=ilu.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def chunk(seeds):
 os.chdir(PUB)
 if PUB not in sys.path:sys.path.insert(0,PUB)
 p=load(PM,f"efb_pub_{os.getpid()}");r=load(RM,f"efb_rv8_{os.getpid()}")
 def pa(o):return PD if o.get("select") is None else p.agent(o)
 def ra(o):return RD if o.get("select") is None else r.agent(o)
 out={}
 for name,od in PANEL.items():
  def opp(o,d=od):return list(d) if o.get("select") is None else domain_policy.domain_agent(o,d)
  pw=rw=po=ro=err=0
  for seed in seeds:
   score=[]
   for policy,deck in ((pa,PD),(ra,RD)):
    w=0
    try:
     random.seed(seed);w+=play(seed,policy,opp,deck,od)==0
     random.seed(seed);w+=play(seed,opp,policy,od,deck)==1
    except Exception:err+=1
    score.append(w)
   pw+=score[0];rw+=score[1];po+=score[0]>score[1];ro+=score[1]>score[0]
  out[name]=(pw,rw,po,ro,err)
 return out
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 60;w=int(sys.argv[2]) if len(sys.argv)>2 else 8;s0=int(os.environ.get("EFB_SEED0","37500000"));seeds=[s0+i for i in range(n)]
 agg={k:[0]*5 for k in PANEL}
 with mp.get_context("fork").Pool(w) as pool:
  for rows in pool.imap_unordered(chunk,[seeds[i::w] for i in range(w)]):
   for k,row in rows.items():agg[k]=[a+b for a,b in zip(agg[k],row)]
 pooled=[0]*5
 for k,row in agg.items():
  pooled=[a+b for a,b in zip(pooled,row)];pw,rw,po,ro,e=row;print(f"{k} {LABEL} {pw}/{2*n} rv8 {rw}/{2*n} disc {po}/{ro} p={mcnemar(po,ro):.5f} err={e}")
 pw,rw,po,ro,e=pooled;print(f"POOLED {LABEL} {pw} rv8 {rw} disc {po}/{ro} p={mcnemar(po,ro):.8f} err={e}")
if __name__=="__main__":main()
