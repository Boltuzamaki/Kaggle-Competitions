"""Ogerpon-only wall-mode screen around router v10."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
 if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play
import domain_policy,top_decks
MAIN=os.environ.get("ROW_MAIN",os.path.join(ROOT,"agent","router_v10","main.py"));DECK=[int(x) for x in open(os.path.join(os.path.dirname(MAIN),"deck.csv")) if x.strip()]
PANELS={"target":{"td00":top_decks.TD_00,"td08":top_decks.TD_08},
"safety":{"td07":top_decks.TD_07,"td15":top_decks.TD_15,"td20":top_decks.TD_20,"td22":top_decks.TD_22}}
PANELS["venusaur"]={"td20":top_decks.TD_20}
PANELS["venus_safety"]={"td00":top_decks.TD_00,"td07":top_decks.TD_07,"td08":top_decks.TD_08,"td15":top_decks.TD_15,"td22":top_decks.TD_22}
PANELS["fros"]={"td04":top_decks.TD_04,"td09":top_decks.TD_09,"td16":top_decks.TD_16}
PANELS["fros_safety"]={"alak":top_decks.TD_02,"grim":top_decks.TD_01,"drag":top_decks.TD_03,"mirror":DECK}
PANELS["slow"]={"td11":top_decks.TD_11}
PANELS["slow_safety"]={"alak":top_decks.TD_02,"grim":top_decks.TD_01,"fros":top_decks.TD_09,"mirror":DECK}
PANELS["grim"]={"td01":top_decks.TD_01,"td14":top_decks.TD_14,"td17":top_decks.TD_17,"td25":top_decks.TD_25}
PANELS["grim_safety"]={"alak":top_decks.TD_02,"drag":top_decks.TD_03,"fros":top_decks.TD_09,"mirror":DECK}
PANEL=PANELS[os.environ.get("ROW_PANEL","target")]
ARMS=os.environ.get("ROW_ARMS","no_wall,force_wall").split(",")
def load(name):
 s=ilu.spec_from_file_location(name,MAIN);m=ilu.module_from_spec(s);s.loader.exec_module(m);return m
def chunk(args):
 arm,seeds=args;c=load(f"row_{arm}_{os.getpid()}");b=load(f"row_b_{os.getpid()}");orig=c.should_wall_mode
 def wall(me,opp,state):
  visible=c.opponent_visible_ids(opp)
  if os.environ.get("ROW_FAMILY")=="venusaur" and visible&{650,651,652}:
   return arm=="force_wall"
  if os.environ.get("ROW_FAMILY")=="fros" and visible&{848,849,861} and not visible&{741,742,743}:
   return arm=="force_wall"
  if os.environ.get("ROW_FAMILY")=="slow" and visible&{162,163}:
   return arm=="force_wall"
  if os.environ.get("ROW_FAMILY")=="grim" and visible&{104,646,647,648}:
   return arm=="force_wall"
  if arm=="strict_wall" and 96 in visible and 1127 in visible:return True
  if os.environ.get("ROW_FAMILY")!="venusaur" and arm!="strict_wall" and visible&{96}:return arm=="force_wall"
  return orig(me,opp,state)
 c.should_wall_mode=wall
 def ca(o):return DECK if o.get("select") is None else c.agent(o)
 def ba(o):return DECK if o.get("select") is None else b.agent(o)
 out={}
 for name,od in PANEL.items():
  def opp(o,d=od):return list(d) if o.get("select") is None else domain_policy.domain_agent(o,d)
  cw=bw=co=bo=0
  for seed in seeds:
   score=[]
   for policy in (ca,ba):
    w=0;random.seed(seed);w+=play(seed,policy,opp,DECK,od)==0;random.seed(seed);w+=play(seed,opp,policy,od,DECK)==1;score.append(w)
   cw+=score[0];bw+=score[1];co+=score[0]>score[1];bo+=score[1]>score[0]
  out[name]=(cw,bw,co,bo)
 return arm,out
def main():
 n=int(sys.argv[1]) if len(sys.argv)>1 else 150;w=int(sys.argv[2]) if len(sys.argv)>2 else 16;s0=int(os.environ.get("ROW_SEED0","49500000"));seeds=[s0+i for i in range(n)];chunks=max(1,w//len(ARMS));jobs=[(a,seeds[i::chunks]) for a in ARMS for i in range(chunks)]
 agg={a:{k:[0]*4 for k in PANEL} for a in ARMS}
 with mp.get_context("fork").Pool(min(w,len(jobs))) as pool:
  for arm,rows in pool.imap_unordered(chunk,jobs):
   for k,row in rows.items():agg[arm][k]=[x+y for x,y in zip(agg[arm][k],row)]
 for arm in ARMS:
  p=[sum(agg[arm][k][i] for k in PANEL) for i in range(4)];print(arm,p,"p",mcnemar(p[2],p[3]),agg[arm])
if __name__=="__main__":main()
