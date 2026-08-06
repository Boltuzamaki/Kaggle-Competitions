"""Transfer TVT only at exactly matching trajectory+GR rows, outer-train only."""
from pathlib import Path
import json,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/results/exact_segment_overlap_transfer_v1';OUT.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(ROOT/'exp/results/outer_train_overlap_transfer_v1/matches.csv')
s=np.load(ROOT/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);groups=s['groups'].astype(str);y=s['y'];base=.1*s['accepted']+.9*s['replacement'];pred=base.copy();known_n=np.zeros(len(y),int);matched_rmse=np.full(len(y),np.inf);matched=np.zeros(len(y),bool);reports=[]
def keys(h):return [tuple(x) for x in np.nan_to_num(np.round(h[['X','Y','Z','GR']].to_numpy(float),3),nan=9.96921e36)]
cuts=np.r_[0,np.flatnonzero(groups[1:]!=groups[:-1])+1,len(groups)]
for a,b in zip(cuts[:-1],cuts[1:]):
 w=groups[a];t=m.set_index('well').loc[w,'match'];hq=pd.read_csv(ROOT/'data/train'/f'{w}__horizontal_well.csv');ht=pd.read_csv(ROOT/'data/train'/f'{t}__horizontal_well.csv');kq=keys(hq);kt=keys(ht);mp={q:i for i,q in enumerate(kt)};ps=np.flatnonzero(hq.TVT_input.isna())[0]
 pairs=[(i,mp[q]) for i,q in enumerate(kq) if q in mp];kp=[q for q in pairs if q[0]<ps];ep=[q for q in pairs if q[0]>=ps]
 if kp:
  diff=np.array([hq.TVT_input.iloc[i]-ht.TVT.iloc[j] for i,j in kp]);shift=float(np.median(diff));pr=float(np.sqrt(np.mean((diff-shift)**2)))
 else:shift=0.;pr=np.inf
 loc={i-ps:j for i,j in ep if 0<=i-ps<b-a}
 last=float(hq.TVT_input.iloc[ps-1])
 for i,j in loc.items():pred[a+i]=float(ht.TVT.iloc[j]+shift-last);matched[a+i]=True
 known_n[a:b]=len(kp);matched_rmse[a:b]=pr
 reports.append({'well':w,'match':t,'known_exact_rows':len(kp),'eval_exact_rows':len(ep),'prefix_matched_rmse':pr,'shift':shift})
def rm(p):return float(np.sqrt(np.mean((y-p)**2)))
grid=[]
for nk in [1,10,50,100,400]:
 for lim in [.25,.5,1,2,4,8]:
  use=matched&(known_n>=nk)&(matched_rmse<=lim);q=base.copy();q[use]=pred[use];grid.append({'min_known':nk,'prefix_rmse_limit':lim,'wells':len(np.unique(groups[use])),'rows':int(use.sum()),'rmse':rm(q)})
g=pd.DataFrame(grid).sort_values('rmse');pd.DataFrame(reports).to_csv(OUT/'matches.csv',index=False);g.to_csv(OUT/'grid.csv',index=False);summary={'baseline':rm(base),'best':g.iloc[0].to_dict(),'total_exact_eval_rows':int(matched.sum()),'pairs_with_exact_eval':int(sum(r['eval_exact_rows']>0 for r in reports))};(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
