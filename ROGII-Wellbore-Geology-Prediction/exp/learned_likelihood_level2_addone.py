from pathlib import Path
import hashlib,json,numpy as np
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/learned_path_likelihood_ratio_v1';z=np.load(OUT/'pred_full.npz',allow_pickle=True);W=z['wells'].astype(str);post=z['pred'];v=np.load(ROOT/'exp/results/generative_curve_prior_expert_errors_v6/oof.npz');assert np.array_equal(W,v['wells'].astype(str));diff=post-v['pred'];s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);l=np.load(ROOT/'exp/results/legal_level2_all_oof_v1/oof.npz');g=s['groups'].astype(str);y=s['y'];center=.1*s['accepted']+.9*s['replacement']+l['anchored_difference'];row=np.zeros(len(y),np.float32);fm={W[i]:k for k,(_,va) in enumerate(GroupKFold(5).split(W,groups=W)) for i in va};fid=np.zeros(len(y),np.int8)
for i,w in enumerate(W):m=g==w;n=m.sum();row[m]=np.interp(np.linspace(0,1,n),np.linspace(0,1,diff.shape[1]),diff[i]);fid[m]=fm[w]
rm=lambda p,m:float(np.sqrt(np.mean((y[m]-p[m])**2)));grid=[]
for a in [0,.1,.25,.5,1]:
 p=center+a*row;grid.append({'alpha':a,'rmse':rm(p,np.ones(len(y),bool)),'folds':[rm(p,fid==k) for k in range(5)],'fold_gains':[rm(center,fid==k)-rm(p,fid==k) for k in range(5)]})
np.savez_compressed(OUT/'exact_level2_correction.npz',correction=row,groups=g,fold=fid);out={'baseline':rm(center,np.ones(len(y),bool)),'grid':grid,'best':min(grid,key=lambda x:x['rmse']),'protocol':'frozen posterior-minus-v6 correction added to immutable level2 exact rows'};out['sha256']=hashlib.sha256((OUT/'exact_level2_correction.npz').read_bytes()).hexdigest();(OUT/'level2_addone.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
