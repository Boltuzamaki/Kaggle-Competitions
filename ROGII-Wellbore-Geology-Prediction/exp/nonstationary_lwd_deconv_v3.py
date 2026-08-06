"""Visible-evidence-selected conservative deconvolution + box*cusp operator."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.ndimage import convolve1d
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/nonstationary_lwd_operator_deconv_v3';OUT.mkdir(parents=True,exist_ok=True)
l=np.load(ROOT/'exp/results/legal_level2_all_oof_v1/oof.npz');s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);g=s['groups'].astype(str);W=np.array(sorted(set(g)));student=.1*s['accepted']+.9*s['replacement'];center=student+l['anchored_difference'];y=s['y'];order=np.random.RandomState(1701).permutation(W);pilot=order[:120];confirm=order[120:360];ops=[('point',0.)]+[(float(L),float(a)) for L in [.5,.75,1,1.5] for a in [2,3,4]];decs=[('none',0.)]+[('wiener',x) for x in [.1,.3,1.]];comb=[(d,o) for d in decs for o in ops];dat=np.linspace(-3,3,5);tilt=np.linspace(-4,4,5);bow=np.linspace(-2,2,3)
def kernel(op):
 if op[0]=='point':return np.array([1.])
 L,a=op;off=np.arange(-8,9.);u=np.linspace(-L/2,L/2,201);k=np.mean((a/2)*np.exp(-a*np.abs(off[:,None]-u[None])),1);return k/k.sum()
def deconv(x,step,d):
 if d[0]=='none':return x
 f=np.fft.rfftfreq(len(x),d=step);H=np.exp(-.5*(2*np.pi*f*.5)**2);return np.fft.irfft(np.fft.rfft(x)*H/(H*H+d[1]),len(x))
def aff(x,y):
 A=np.c_[x,np.ones(len(x))];c=np.linalg.lstsq(A,y,rcond=None)[0];r=y-A@c;return c,np.median(abs(r-np.median(r)))*1.4826+5
def one(w,choices):
 m=g==w;n=m.sum();h=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');tw=pd.read_csv(ROOT/'data/train'/f'{w}__typewell.csv').dropna().sort_values('TVT');ps=int(h.TVT_input.notna().sum());known=h.TVT_input.iloc[:ps].to_numpy();anchor=known[-1];tt=tw.TVT.to_numpy();tg=tw.GR.to_numpy();hg=pd.to_numeric(h.GR,errors='coerce').interpolate(limit_direction='both').to_numpy();u=np.linspace(-1,1,n);cur=np.asarray([center[m]+d+t*u+b*(u*u-1) for d in dat for t in tilt for b in bow]);cost={}
 for d,op in choices:
  td=deconv(tg,np.median(np.diff(tt)),d);k=kernel(op);vr=convolve1d(np.interp(known,tt,td),k,mode='nearest');co,scale=aff(vr,hg[:ps]);paths=np.c_[np.tile(known,(len(cur),1)),anchor+cur];sim=convolve1d(np.interp(paths,tt,td),k,axis=1,mode='nearest')[:,ps:]*co[0]+co[1];r=(hg[ps:][None]-sim)/max(scale,20);cost[(d,op)]=np.mean(np.log1p((r/3)**2),1)
 return {'w':w,'m':m,'cur':cur,'cost':cost}
pr=[one(w,comb) for w in pilot];evid={str(c):float(np.mean([np.min(r['cost'][c]) for r in pr])) for c in comb};chosen=min(comb,key=lambda c:evid[str(c)]);cr=[one(w,[chosen,(('none',0.),('point',0.))]) for w in confirm]
def report(rows,c):
 se0=se=nn=wins=0
 for r in rows:
  z=r['cost'][c];ww=np.exp(-(z-z.min())/.35);ww/=ww.sum();p=ww@r['cur'];m=r['m'];a=np.sum((y[m]-center[m])**2);b=np.sum((y[m]-p)**2);se0+=a;se+=b;nn+=m.sum();wins+=b<a
 return {'wells':len(rows),'base':float(np.sqrt(se0/nn)),'model':float(np.sqrt(se/nn)),'wins':int(wins)}
point=(('none',0.),('point',0.));out={'chosen':str(chosen),'pilot':report(pr,chosen),'pilot_point':report(pr,point),'confirmation':report(cr,chosen),'confirmation_point':report(cr,point),'best_evidence':evid[str(chosen)],'point_evidence':evid[str(point)],'protocol':'same locked seed lists; deconv/operator selected only by visible-prefix pilot GR evidence'};(OUT/'summary.json').write_text(json.dumps(out,indent=2));out['sha256']=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest();(OUT/'verified.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
