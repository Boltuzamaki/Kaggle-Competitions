"""Strict nested cubic-coefficient predictions as rowwise residual-LGB features."""
from pathlib import Path
import json,joblib,numpy as np,pandas as pd,lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/cubic_aux_features_nested";OUT.mkdir(parents=True,exist_ok=True)
f=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl");tab=np.load(ROOT/"exp/results/well_cubic_coeff_tabular/well_table.npz",allow_pickle=True)
Xw,Yw,ww=tab["X"],tab["Y"],tab["wells"].astype(str)
base=np.asarray(joblib.load(ROOT/"r_v4b/stack_v4_oofs.joblib")["oofs"]["lgb7"],float);y=f.target.to_numpy(float);res=y-base
drop={"well","id","target","last_known_tvt","supertype"};cols=[c for c in f if c not in drop]
Xr=f[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
well_ix={str(w):np.asarray(ix) for w,ix in f.groupby("well",sort=False).indices.items()}
outer=list(GroupKFold(5).split(Xw,groups=ww));plain=np.zeros(len(y));auxp=np.zeros(len(y));fold_rows=[]
def cat(seed):return CatBoostRegressor(loss_function="MultiRMSE",iterations=500,depth=5,learning_rate=.04,
 l2_leaf_reg=12,verbose=False,random_seed=seed,thread_count=12)
def lgbm(seed):return lgb.LGBMRegressor(objective="huber",n_estimators=380,learning_rate=.035,
 num_leaves=28,max_depth=8,min_child_samples=110,max_bin=127,colsample_bytree=.7,
 subsample=.85,subsample_freq=1,reg_alpha=2.,reg_lambda=16.,verbosity=-1,n_jobs=12,random_state=seed)
def rm(p,ix):return float(np.sqrt(np.mean((p[ix]-y[ix])**2)))
for fold,(otr,ova) in enumerate(outer):
 cp_tr=np.zeros((len(otr),4),np.float32);ym=Yw[otr].mean(0);ys=Yw[otr].std(0)+1e-4
 for j,(it,iv) in enumerate(GroupKFold(4).split(Xw[otr],groups=ww[otr])):
  m=cat(1000*fold+j);m.fit(Xw[otr[it]],(Yw[otr[it]]-ym)/ys);cp_tr[iv]=m.predict(Xw[otr[iv]])*ys+ym
 m=cat(9000+fold);m.fit(Xw[otr],(Yw[otr]-ym)/ys);cp_va=m.predict(Xw[ova])*ys+ym
 tri=np.concatenate([well_ix[ww[i]] for i in otr]);vai=np.concatenate([well_ix[ww[i]] for i in ova])
 A_tr=np.zeros((len(tri),6),np.float32);A_va=np.zeros((len(vai),6),np.float32)
 pos=0
 for j,i in enumerate(otr):
  ix=well_ix[ww[i]];n=len(ix);xx=np.linspace(-1,1,n);curve=np.polynomial.legendre.legval(xx,cp_tr[j])
  A_tr[pos:pos+n,:4]=cp_tr[j];A_tr[pos:pos+n,4]=curve;A_tr[pos:pos+n,5]=np.gradient(curve);pos+=n
 pos=0
 for j,i in enumerate(ova):
  ix=well_ix[ww[i]];n=len(ix);xx=np.linspace(-1,1,n);curve=np.polynomial.legendre.legval(xx,cp_va[j])
  A_va[pos:pos+n,:4]=cp_va[j];A_va[pos:pos+n,4]=curve;A_va[pos:pos+n,5]=np.gradient(curve);pos+=n
 # concatenated row order above follows well blocks, not sorted global indices.
 order_tr=np.argsort(tri);tri=tri[order_tr];A_tr=A_tr[order_tr]
 order_va=np.argsort(vai);vai=vai[order_va];A_va=A_va[order_va]
 fit=np.arange(len(tri))[tri%8==fold%8]
 mp=lgbm(20000+fold);mp.fit(Xr[tri[fit]],res[tri[fit]]);plain[vai]=base[vai]+mp.predict(Xr[vai])
 ma=lgbm(30000+fold);ma.fit(np.c_[Xr[tri[fit]],A_tr[fit]],res[tri[fit]])
 auxp[vai]=base[vai]+ma.predict(np.c_[Xr[vai],A_va])
 fold_rows.append({"fold":fold,"base":rm(base,vai),"plain":rm(plain,vai),"aux":rm(auxp,vai),
                   "aux_gain_vs_plain":rm(plain,vai)-rm(auxp,vai)})
 print(fold_rows[-1],flush=True)
allix=np.arange(len(y));summary={"rows":len(y),"wells":len(ww),"features":len(cols),
 "base":rm(base,allix),"plain":rm(plain,allix),"aux":rm(auxp,allix),
 "aux_gain_vs_plain":rm(plain,allix)-rm(auxp,allix),"folds":fold_rows,
 "protocol":"5 outer x 4 inner complete-well GKF stage1 Cat coefficients; stride-8 row residual LGB; all held-out rows score"}
(OUT/"summary.json").write_text(json.dumps(summary,indent=2));np.savez_compressed(OUT/"oof.npz",y=y,base=base,plain=plain,aux=auxp)
print(json.dumps(summary,indent=2))
