"""Validate a tuned weight file against STOCK on seeds disjoint from tuning."""
import json, os, random, sys, importlib.util as ilu
sys.path.insert(0,'agent'); sys.path.insert(0,'tools/crn')
from paired_eval import play, mcnemar
FP='agent/fork/fork_main.py'
DECK=[int(x) for x in open('agent/fork/deck.csv') if x.strip()]
def load(tag):
    s=ilu.spec_from_file_location('fm_'+tag,FP); m=ilu.module_from_spec(s); s.loader.exec_module(m); return m
wfile=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 60
champ=json.load(open(wfile))
cand=load('c'); stock=load('s')
cand.WEIGHTS.clear(); cand.WEIGHTS.update(champ)
a=b=cw=sw=0
for s in [314000+i for i in range(n)]:
    x=y=0
    for seat in (0,1):
        random.seed(s)
        r = play(s,cand.agent,stock.agent,DECK,DECK) if seat==0 else play(s,stock.agent,cand.agent,DECK,DECK)
        x += int(r==(0 if seat==0 else 1)); y += int(r==(1 if seat==0 else 0))
    cw+=x; sw+=y
    if x>y: a+=1
    elif y>x: b+=1
    if (s-314000+1)%10==0:
        print(f'  {s-314000+1} seeds: {cw}-{sw}, discordant {a}/{b}, p={mcnemar(a,b):.3f}', flush=True)
p=mcnemar(a,b)
print(f'\n{wfile} vs STOCK, {n} disjoint seeds: {cw} wins vs {sw}')
print(f'  discordant {a}/{b}   p={p:.4f}')
print('  VERDICT:', 'BETTER (p<0.05)' if (a>b and p<0.05) else 'NOT SIGNIFICANT')
