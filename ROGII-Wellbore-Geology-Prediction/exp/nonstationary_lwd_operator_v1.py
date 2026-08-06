"""Inclination-dependent typewell-to-LWD forward response around level2."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d,uniform_filter1d
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/nonstationary_lwd_operator_v1';OUT.mkdir(parents=True,exist_ok=True)
l=np.load(ROOT/'exp/results/legal_level2_all_oof_v1/oof.npz');s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);g=s['groups'].astype(str);W=np.array(sorted(set(g)));student=.1*s['accepted']+.9*s['replacement'];center=student+l['anchored_difference'];y=s['y'];order=np.random.RandomState(1701).permutation(W);pilot=set(order[:120]);confirm=set(order[120:360]);ops=[('point',0),('gauss',.5),('gauss',1),('gauss',2),('box',1),('box',2),('asym',1),('asym',2)];dat=np.linspace(-3,3,5);tilt=np.linspace(-4,4,5);bow=np.linspace(-2,2,3)
def robust_aff(x,y):
 A=np.c_[x,np.ones(len(x))];c=np.linalg.lstsq(A,y,rcond=None)[0]
 for _ in range(3):r=y-A@c;sc=np.median(abs(r-np.median(r)))*1.4826+5;ww=1/np.maximum(1,abs(r)/(2.5*sc));c=np.linalg.lstsq(A*ww[:,None],y*ww,rcond=None)[0]
 return c,sc
def one(w):
 m=g==w;n=m.sum();h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').dropna().sort_values('TVT');ps=int(h.TVT_input.notna().sum());anchor=h.TVT_input.iloc[ps-1];tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();hg=pd.to_numeric(h.GR,errors='coerce').interpolate(limit_direction='both').to_numpy();vis=h.TVT_input.notna().to_numpy();ref=np.interp(h.TVT_input[vis],tt,tg);co,scale=robust_aff(ref,hg[vis]);tg=co[0]*tg+co[1];step=np.median(np.diff(tt));smooth={}
 for typ,b in ops:
  key=(typ,b)
  if typ=='point':smooth[key]=[tg]*5
  else:
   arr=[]
   for sig in [.25,.5,1,2,4]:
    q=gaussian_filter1d(tg,max(.25,sig/step)) if typ in ['gauss','asym'] else uniform_filter1d(tg,max(1,int(2*sig/step+1)))
    if typ=='asym':q=.7*q+.3*np.interp(tt+sig,tt,q)
    arr.append(q)
   smooth[key]=arr
 base=center[m];u=np.linspace(-1,1,n);cur=[]
 for d,t,bw in [(d,t,bw) for d in dat for t in tilt for bw in bow]:
  cur.append(base+d+t*u+bw*(u*u-1))
 cur=np.asarray(cur);costs={}
 for typ,b in ops:
  cc=[]
  for q in cur:
   path=anchor+q;sl=np.abs(np.gradient(path)/(np.gradient(h.MD.iloc[ps:].to_numpy())+1e-6));sig=np.clip(b*(.25+4*sl),.25,4) if b else np.full(n,.25);bins=np.argmin(abs(sig[:,None]-np.array([.25,.5,1,2,4])[None]),1);sim=np.empty(n)
   for k in range(5):mm=bins==k;sim[mm]=np.interp(path[mm],tt,smooth[(typ,b)][k])
   r=(hg[ps:]-sim)/max(scale,20);cc.append(np.mean(np.log1p((r/3)**2)))
  costs[(typ,b)]=np.asarray(cc)
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
out={'chosen':[chosen[0],float(chosen[1])],'legal_evidence':evid,'pilot':report(pilot,chosen),'confirmation':report(confirm,chosen),'point_pilot':report(pilot,('point',0)),'point_confirmation':report(confirm,('point',0)),'protocol':'operator selected on pilot GR evidence only; TVT score after lock; disjoint seed lists'};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
