"""Source-grounded normalized box_L * cusp_a LWD operator around level2."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/nonstationary_lwd_operator_boxcusp_v2';OUT.mkdir(parents=True,exist_ok=True)
l=np.load(ROOT/'exp/results/legal_level2_all_oof_v1/oof.npz');s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);g=s['groups'].astype(str);W=np.array(sorted(set(g)));student=.1*s['accepted']+.9*s['replacement'];center=student+l['anchored_difference'];y=s['y'];order=np.random.RandomState(1701).permutation(W);pilot=set(order[:120]);confirm=set(order[120:360]);ops=[('point',0.)]+[(float(L),float(a)) for L in [.5,.75,1,1.5] for a in [2,3,4]];dat=np.linspace(-3,3,5);tilt=np.linspace(-4,4,5);bow=np.linspace(-2,2,3)
def kernel(op):
 if op[0]=='point':return np.array([1.])
 L,a=op;off=np.arange(-8,9.);u=np.linspace(-L/2,L/2,201);k=np.mean((a/2)*np.exp(-a*np.abs(off[:,None]-u[None])),1);return k/k.sum()
def robust_aff(x,y):
 A=np.c_[x,np.ones(len(x))];c=np.linalg.lstsq(A,y,rcond=None)[0]
 for _ in range(3):r=y-A@c;sc=np.median(abs(r-np.median(r)))*1.4826+5;ww=1/np.maximum(1,abs(r)/(2.5*sc));c=np.linalg.lstsq(A*ww[:,None],y*ww,rcond=None)[0]
 return c,sc
def one(w):
 m=g==w;n=m.sum();h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').dropna().sort_values('TVT');ps=int(h.TVT_input.notna().sum());anchor=h.TVT_input.iloc[ps-1];tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();hg=pd.to_numeric(h.GR,errors='coerce').interpolate(limit_direction='both').to_numpy();vis=h.TVT_input.notna().to_numpy();known=h.TVT_input[vis].to_numpy();cal={}
 for op in ops:
  ref=np.convolve(np.interp(known,tt,tg),kernel(op),mode='same');cal[op]=robust_aff(ref,hg[vis])
 base=center[m];u=np.linspace(-1,1,n);cur=[]
 for d,t,bw in [(d,t,bw) for d in dat for t in tilt for bw in bow]:
  cur.append(base+d+t*u+bw*(u*u-1))
 cur=np.asarray(cur);costs={}
 for op in ops:
  cc=[]
  for q in cur:
   path=np.r_[known,anchor+q];sim=np.convolve(np.interp(path,tt,tg),kernel(op),mode='same')[ps:];co,scale=cal[op];sim=co[0]*sim+co[1]
   r=(hg[ps:]-sim)/max(scale,20);cc.append(np.mean(np.log1p((r/3)**2)))
  costs[op]=np.asarray(cc)
 return {'w':w,'m':m,'cur':cur,'cost':costs}
rows=[one(w) for w in order[:360]]
# Legal evidence-only operator choice on pilot; no TVT accessed here.
evid={str(op):float(np.mean([np.min(r['cost'][op]) for r in rows if r['w'] in pilot])) for op in ops};chosen=min(ops,key=lambda op:evid[str(op)])
def report(S,op):
 se0=se=nn=wins=0
 for r in rows:
  if r['w'] not in S:continue
  c=r['cost'][op];ww=np.exp(-(c-c.min())/.35);ww/=ww.sum();p=ww@r['cur'];m=r['m'];e0=np.sum((y[m]-center[m])**2);e=np.sum((y[m]-p)**2);se0+=e0;se+=e;nn+=m.sum();wins+=e<e0
 return {'wells':len(S),'base':float(np.sqrt(se0/nn)),'operator':float(np.sqrt(se/nn)),'well_wins':int(wins)}
out={'chosen':[str(chosen[0]),float(chosen[1])],'legal_evidence':evid,'pilot':report(pilot,chosen),'confirmation':report(confirm,chosen),'point_pilot':report(pilot,('point',0.)),'point_confirmation':report(confirm,('point',0.)),'protocol':'exact normalized box_L*cusp_a evaluated along candidate MD path; operator selected on pilot GR evidence only'};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
