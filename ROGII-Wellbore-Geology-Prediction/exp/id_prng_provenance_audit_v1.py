"""Generic hexadecimal-ID provenance/RNG signal audit with permutation controls."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/id_prng_provenance_v1';OUT.mkdir(parents=True,exist_ok=True)
l=np.load(ROOT/'exp/results/legal_level2_all_oof_v1/oof.npz');s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);assert np.array_equal(l['groups'].astype(str),s['groups'].astype(str));g=l['groups'].astype(str);y=s['y'];base=.1*s['accepted']+.9*s['replacement']+l['anchored_difference'];W=np.array(sorted(set(g)));L=128
def rs(x):return np.interp(np.linspace(0,1,L),np.linspace(0,1,len(x)),x)
C=[];lens=[]
for w in W:m=g==w;C.append(rs(y[m]-base[m]));lens.append(m.sum())
C=np.asarray(C);lens=np.asarray(lens);u=np.linspace(-.5,.5,L);coef=np.asarray([np.polyfit(u,c,1) for c in C])
def idfeat(w):
 n=int(w,16);nib=np.array([int(x,16) for x in w]);bits=np.array([(n>>i)&1 for i in range(32)]);f=list(nib/15)+list(bits);f += [n/2**32,bin(n).count('1')/32]
 for k in [1,2,3,5,8,13,21,34]:a=2*np.pi*((n*k)%2**32)/2**32;f += [np.sin(a),np.cos(a)]
 for st in range(7):f.append(int(w[st:st+2],16)/255)
 for mult,add in [(2654435761,1013904223),(1664525,1013904223),(1103515245,12345),(2246822519,3266489917)]:q=(n*mult+add)&0xffffffff;f += [q/2**32,((q>>16)&65535)/65535]
 return f
X=np.asarray([idfeat(w) for w in W]);perm=np.random.RandomState(1401).permutation(len(W));Xp=X[perm];rows=[];predsave={}
for rep in range(3):
 folds=list(KFold(5,shuffle=True,random_state=1410+rep).split(W))
 for kind,XX in [('id',X),('permuted',Xp)]:
  for model in ['et','ridge']:
   pred=np.zeros_like(C)
   for fo,(tr,va) in enumerate(folds):
    sc=StandardScaler().fit(XX[tr]);a=sc.transform(XX[tr]);q=sc.transform(XX[va]);cp=PCA(16,random_state=1).fit(C[tr]);z=cp.transform(C[tr])
    if model=='et':m=ExtraTreesRegressor(n_estimators=800,min_samples_leaf=15,max_features=.7,n_jobs=-1,random_state=1420+fo)
    else:m=Ridge(alpha=100)
    m.fit(a,z);pred[va]=cp.inverse_transform(m.predict(q))
   rm=float(np.sqrt(np.average(np.mean((C-pred)**2,1),weights=lens)));rows.append({'repeat':rep,'features':kind,'model':model,'rmse':rm});predsave[f'{kind}_{model}_{rep}']=pred
zero=float(np.sqrt(np.average(np.mean(C*C,1),weights=lens)));summary={'wells':len(W),'zero':zero,'rows':rows,'means':{f'{k}_{m}':float(np.mean([r['rmse'] for r in rows if r['features']==k and r['model']==m])) for k in ['id','permuted'] for m in ['et','ridge']},'protocol':'3 repeated shuffled whole-well 5-fold splits; generic ID transform; fixed permutation control'}
np.savez_compressed(OUT/'oof_repeats.npz',wells=W,true=C,length=lens,**predsave);summary['sha256']=hashlib.sha256((OUT/'oof_repeats.npz').read_bytes()).hexdigest();(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
