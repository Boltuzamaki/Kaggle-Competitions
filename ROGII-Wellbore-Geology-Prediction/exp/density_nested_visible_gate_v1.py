"""Nested visible-feature shrinkage gate for strict density-ratio correction."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from sklearn.model_selection import GroupKFold,KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

R=Path(__file__).resolve().parents[1];SRC=R/'exp/results/supervised_density_ratio_gr_full_gkf_v1/oof.npz';OUT=R/'exp/results/density_nested_visible_gate_v1';OUT.mkdir(parents=True,exist_ok=True)
S=np.load(R/'exp/results/pf_student10_full_oof/meta_oof.npz',allow_pickle=True);y=S['y'];g=S['groups'].astype(str);base=.1*S['accepted']+.9*S['replacement']+np.load(R/'exp/results/legal_level2_all_oof_v1/oof.npz',allow_pickle=True)['anchored_difference'];D=np.load(SRC,allow_pickle=True);assert np.array_equal(g,D['groups'].astype(str));corr=D['correction'];err=y-base
V=np.load(R/'exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz');W=V['wells'].astype(str);Q=V['query'];R4=np.load(R/'exp/results/generative_curve_prior_risk_v4/oof.npz',allow_pickle=True);r4={w:a for w,a in zip(R4['wells'].astype(str),R4['alpha'])};cuts=np.r_[0,np.flatnonzero(g[1:]!=g[:-1])+1,len(g)];ixs={g[a]:np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])};F=[];A=[];lens=[];diag=[]
for w,q in zip(W,Q):
 ix=ixs[w];c=corr[ix];t=np.linspace(0,1,len(ix));co=np.polyfit(t,c,3);den=float(c@c)+1e-8;a=float(np.clip((err[ix]@c)/den,0,1));gain=float(err[ix]@err[ix]-(err[ix]-c)@(err[ix]-c));
 curve=[np.mean(c),np.std(c),np.mean(abs(c)),np.max(abs(c)),c[0],c[-1],*co,np.mean(abs(np.diff(c))),np.std(np.diff(c)),r4.get(w,0)]
 F.append(curve);A.append(a);lens.append(len(ix));diag.append({'well':w,'oracle_alpha':a,'gain':gain,'corr_rms':float(np.sqrt(np.mean(c*c)))})
F=np.asarray(F);A=np.asarray(A);lens=np.asarray(lens);folds=list(GroupKFold(5).split(W,groups=W));predw=np.zeros(len(W));choices=[]
def make(train,test,use_q):
 sc=StandardScaler().fit(F[train]);a=sc.transform(F[train]);b=sc.transform(F[test])
 if use_q:
  qs=StandardScaler().fit(Q[train]);qa=qs.transform(Q[train]);qb=qs.transform(Q[test]);pc=PCA(16,whiten=True,random_state=1701).fit(qa);a=np.c_[a,pc.transform(qa)];b=np.c_[b,pc.transform(qb)]
 return a,b
for fo,(tr,va) in enumerate(folds):
 best=None
 for useq in [False,True]:
  for ridge in [10,100,1000]:
   for shrink in [.25,.5,1]:
    se=n=0
    for aa,bb in KFold(4,shuffle=True,random_state=1800+fo).split(tr):
     x,z=make(tr[aa],tr[bb],useq);m=Ridge(ridge).fit(x,A[tr[aa]]);pw=np.clip(shrink*m.predict(z),0,1)
     for i,w0 in zip(tr[bb],pw):ix=ixs[W[i]];se+=float((err[ix]-w0*corr[ix])@(err[ix]-w0*corr[ix]));n+=len(ix)
    score=np.sqrt(se/n)
    if best is None or score<best[0]:best=(score,useq,ridge,shrink)
 x,z=make(tr,va,best[1]);predw[va]=np.clip(best[3]*Ridge(best[2]).fit(x,A[tr]).predict(z),0,1);choices.append({'fold':fo,'inner_rmse':best[0],'use_q':best[1],'ridge':best[2],'shrink':best[3]})
pred=base.copy();
for w,a in zip(W,predw):pred[ixs[w]]+=a*corr[ixs[w]]
fm={w:f for f,(_,v) in enumerate(folds) for w in W[v]};rf=np.array([fm[w] for w in g]);rm=lambda p,m:float(np.sqrt(np.mean((y[m]-p[m])**2)));fr=[{'fold':f,'baseline':rm(base,rf==f),'gated':rm(pred,rf==f),'gain':rm(base,rf==f)-rm(pred,rf==f),'weight_mean':float(np.mean(predw[[fm[w]==f for w in W]]))} for f in range(5)];diag=pd.DataFrame(diag);diag['fold']=[fm[w] for w in W];diag.to_csv(OUT/'well_diagnostics.csv',index=False)
summary={'input_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'baseline':rm(base,np.ones(len(y),bool)),'fixed_density':rm(base+corr,np.ones(len(y),bool)),'gated':rm(pred,np.ones(len(y),bool)),'folds':fr,'fold_wins':sum(x['gain']>0 for x in fr),'choices':choices,'fold_oracle_alpha_mean':diag.groupby('fold').oracle_alpha.mean().to_dict(),'fold_gain_mean':diag.groupby('fold').gain.mean().to_dict(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};np.savez_compressed(OUT/'oof.npz',wells=W,weight=predw,groups=g,prediction=pred);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
