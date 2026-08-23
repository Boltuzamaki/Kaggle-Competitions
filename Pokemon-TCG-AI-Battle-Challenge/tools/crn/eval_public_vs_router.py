"""Direct both-seat audit of an extracted public agent against router v8."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
 if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play
PUB=os.path.abspath(os.environ["PVR_DIR"]);LABEL=os.environ.get("PVR_LABEL",os.path.basename(os.path.dirname(PUB)))
PM=os.path.join(PUB,"main.py");PD=[int(x) for x in open(os.path.join(PUB,"deck.csv")) if x.strip()]
RNAME=os.environ.get("PVR_ROUTER","router_v8")
RM=os.path.join(ROOT,"agent",RNAME,"main.py");RD=[int(x) for x in open(os.path.join(ROOT,"agent",RNAME,"deck.csv")) if x.strip()]
def load(path,name):
 s=ilu.spec_from_file_location(name,path);m=ilu.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def chunk(seeds):
 os.chdir(PUB)
 if PUB not in sys.path:sys.path.insert(0,PUB)
 p=load(PM,f"pub_{os.getpid()}");r=load(RM,f"router_{os.getpid()}")
 def pa(o):return PD if o.get("select") is None else p.agent(o)
 def ra(o):return RD if o.get("select") is None else r.agent(o)
 pw=rw=po=ro=err=0
 for seed in seeds:
  wins=[0,0]
  try:
   random.seed(seed);z=play(seed,pa,ra,PD,RD);wins[0]+=z==0;wins[1]+=z==1
   random.seed(seed);z=play(seed,ra,pa,RD,PD);wins[0]+=z==1;wins[1]+=z==0
  except Exception:
   err+=1;continue
  pw+=wins[0];rw+=wins[1];po+=wins[0]>wins[1];ro+=wins[1]>wins[0]
 return pw,rw,po,ro,err
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 80;w=int(sys.argv[2]) if len(sys.argv)>2 else 4;s0=int(os.environ.get("PVR_SEED0","35500000"))
 seeds=[s0+i for i in range(n)];total=[0]*5
 with mp.get_context("fork").Pool(w) as pool:
  for row in pool.imap_unordered(chunk,[seeds[i::w] for i in range(w)]):total=[a+b for a,b in zip(total,row)]
 pw,rw,po,ro,err=total;print(f"{LABEL} {pw}/{2*(n-err)} {RNAME} {rw}/{2*(n-err)} seeds {po}/{ro} p={mcnemar(po,ro):.6f} errors={err}")
if __name__=="__main__":main()
