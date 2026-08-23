"""Validate a non-leaking conjunctive TD12 Crustle/Ogerpon profile on router v6."""
from __future__ import annotations
import importlib.util as ilu,multiprocessing as mp,os,random,sys
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(os.path.dirname(HERE))
for p in (HERE,os.path.join(ROOT,"agent"),os.path.join(ROOT,"tools")):
 if p not in sys.path:sys.path.insert(0,p)
from paired_eval import mcnemar,play
import domain_policy,top_decks
MAIN=os.environ.get("T12P_MAIN",os.path.join(ROOT,"agent","router_v6","main.py"));DECK=[int(x) for x in open(os.path.join(os.path.dirname(MAIN),"deck.csv")) if x.strip()]
PANELS={"target":{"td12":top_decks.TD_12},"fros":{"td04":top_decks.TD_04,"td09":top_decks.TD_09},
"drag":{"td03":top_decks.TD_03,"td05":top_decks.TD_05,"td19":top_decks.TD_19,"td23":top_decks.TD_23},
"kang_crust":{"td12":top_decks.TD_12,"td13":top_decks.TD_13,"td18":top_decks.TD_18,"td26":top_decks.TD_26},
"kang_safety":{"td07":top_decks.TD_07,"td15":top_decks.TD_15,"td22":top_decks.TD_22,"mirror":DECK},
"kang_extra":{"slowking":top_decks.TD_11,"venusaur":top_decks.TD_20},
"safety":{"mirror":DECK,"ogerpon":top_decks.TD_00,"garchomp":top_decks.TD_06,"alak":top_decks.TD_02}}
def load(name):
 s=ilu.spec_from_file_location(name,MAIN);m=ilu.module_from_spec(s);s.loader.exec_module(m);return m
def chunk(args):
 panel,seeds=args;c=load(f"td12p_{os.getpid()}");b=load(f"v6_{os.getpid()}");orig=c.facing_terminal_mill_favorable_lineage
 def strict(opp):
  visible=c.opponent_visible_ids(opp)
  if os.environ.get("T12P_PROFILE")=="fros":
   return (bool(visible&{848,849,861}) and not bool(visible&{741,742,743})) or orig(opp)
  if os.environ.get("T12P_PROFILE")=="drag":
   return (bool(visible&{119,120,121}) and not bool(visible&{741,742,743})) or orig(opp)
  if os.environ.get("T12P_PROFILE")=="kang_crust":
   return (756 in visible and bool(visible&{344,345})) or orig(opp)
  if os.environ.get("T12P_PROFILE")=="kang_only":
   return (756 in visible) or orig(opp)
  return (117 in visible and bool(visible&{344,345})) or orig(opp)
 c.facing_terminal_mill_favorable_lineage=strict
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
 n=int(sys.argv[1]) if len(sys.argv)>1 else 150;w=int(sys.argv[2]) if len(sys.argv)>2 else 8;panel=os.environ.get("T12P_PANEL","target")
 s0=int(os.environ.get("T12P_SEED0","25500000"));seeds=[s0+i for i in range(n)];agg={k:[0,0,0,0] for k in PANELS[panel]}
 with mp.get_context("fork").Pool(w) as pool:
  for rows in pool.imap_unordered(chunk,[(panel,seeds[i::w]) for i in range(w)]):
   for k,row in rows.items():agg[k]=[a+b for a,b in zip(agg[k],row)]
 pooled=[0,0,0,0]
 for k,row in agg.items():
  pooled=[a+b for a,b in zip(pooled,row)];cw,bw,co,bo=row;print(f"{k:10s} strict {cw}/{2*n} v6 {bw}/{2*n} disc {co}/{bo} p={mcnemar(co,bo):.6f}")
 cw,bw,co,bo=pooled;print(f"POOLED strict {cw} v6 {bw} disc {co}/{bo} p={mcnemar(co,bo):.6f}")
if __name__=="__main__":main()
