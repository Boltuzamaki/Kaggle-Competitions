"""Does the fork's policy play better on a stronger Alakazam list?

Its 69 weights are Alakazam-card-specific, so they should still apply to another
Alakazam build. td_02 is M Sato's list (LB 1170.6) vs the fork's own (converged
742.5 for us). Paired on shared seeds against a fixed opponent panel.
"""
import os, random, sys, importlib.util as ilu
sys.path.insert(0,'agent'); sys.path.insert(0,'tools/crn')
from paired_eval import play, build, mcnemar
import top_decks
FP='agent/fork/fork_main.py'
STOCK=[int(x) for x in open('agent/fork/deck.csv') if x.strip()]
ALT=list(top_decks.TOP_DECKS['td_02'])
def load(t):
    s=ilu.spec_from_file_location('fm_'+t,FP); m=ilu.module_from_spec(s); s.loader.exec_module(m); return m
fa=load('a'); fb=load('b')
def bind(m,deck):
    def w(o):
        if o.get('select') is None: return list(deck)
        return m.agent(o)
    return w
A=bind(fa,ALT); B=bind(fb,STOCK)
import arena  # noqa
OPPS=[build(n) for n in ("hybs-other","td-td_08","meta-grimmsnarl")]
n=int(sys.argv[1]) if len(sys.argv)>1 else 30
a=b=aw=bw=0
for fo,do in OPPS:
    for s in [271000+i for i in range(n)]:
        x=y=0
        for seat in (0,1):
            random.seed(s)
            x += int(play(s,A,fo,ALT,do)==0) if seat==0 else int(play(s,fo,A,do,ALT)==1)
        for seat in (0,1):
            random.seed(s)
            y += int(play(s,B,fo,STOCK,do)==0) if seat==0 else int(play(s,fo,B,do,STOCK)==1)
        aw+=x; bw+=y
        if x>y: a+=1
        elif y>x: b+=1
p=mcnemar(a,b)
print(f'fork policy on td_02 (M Sato LB1170) vs on its OWN deck')
print(f'  wins {aw} vs {bw}   discordant {a}/{b}   p={p:.4f}')
print('  VERDICT:', 'ALT DECK BETTER' if (a>b and p<0.05) else ('alt worse' if b>a else 'no difference'))
