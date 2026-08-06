"""Legal duplicate/typewell-master fingerprints and cluster-grouped diagnostics."""
from pathlib import Path
import hashlib,json,contextlib,io,runpy
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"exp/results/duplicate_master_cluster";OUT.mkdir(parents=True,exist_ok=True)
def ahash(a,dec=None):
 a=np.asarray(a,float)
 if dec is not None:a=np.round(a,dec)
 a=np.nan_to_num(a,nan=9.87654321e30,posinf=8.7e30,neginf=-8.7e30)
 return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
rows=[]
for wi,w in enumerate(sorted(p.stem.replace("__horizontal_well","") for p in (ROOT/"data/train").glob("*__horizontal_well.csv"))):
 h=pd.read_csv(ROOT/f"data/train/{w}__horizontal_well.csv");t=pd.read_csv(ROOT/f"data/train/{w}__typewell.csv").sort_values("TVT")
 nv=int(h.TVT_input.notna().sum());p=nv-1
 common=h[["MD","X","Y","Z","GR","TVT_input"]].to_numpy(float);traj=h[["MD","X","Y","Z"]].to_numpy(float)
 pref=common[:nv].copy();pref[:,1:4]-=pref[0,1:4]
 tw=t[["TVT","GR"]].to_numpy(float)
 frame_key="|".join(map(str,(round(float(tw[0,0]),1),round(float(tw[-1,0]),1),
                              round(float(np.nansum(tw[:200,1])),0),len(tw))))
 rows.append({"well":w,"rows":len(h)-nv,"nv":nv,"typewell_exact":ahash(tw,None),
  "master":ahash(tw,6),"master_frame":frame_key,"horizontal_exact":ahash(common,6),
  "horizontal_round":ahash(common,1),"trajectory_shape":ahash(np.c_[traj[:,0]-traj[0,0],traj[:,1:4]-traj[0,1:4]],2),
  "prefix_shape":ahash(pref,2),"x":h.X.iloc[p],"y":h.Y.iloc[p],"z":h.Z.iloc[p],"tvt":h.TVT_input.iloc[p],
  "xe":h.X.iloc[-1],"ye":h.Y.iloc[-1]})
M=pd.DataFrame(rows).set_index("well")
# Near twins: same legal typewell master and nearly colocated heel/anchor.
parent=np.arange(len(M))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):
 a,b=find(i),find(j)
 if a!=b:parent[b]=a
for _,ids in M.reset_index().groupby("master_frame").indices.items():
 ids=np.asarray(ids);Q=M.iloc[ids][["x","y","z","tvt"]].to_numpy(float);tree=cKDTree(Q/np.array([20,20,5,5]))
 for a,b in tree.query_pairs(1.0):union(ids[a],ids[b])
M["near_twin_cluster"]=[find(i) for i in range(len(M))]
M["near_twin_size"]=M.groupby("near_twin_cluster").rows.transform("size").to_numpy()
counts={k:{"unique":int(M[k].nunique()),"duplicate_wells":int(M[k].duplicated(False).sum()),
           "clusters_gt1":int((M.groupby(k).size()>1).sum())}
        for k in ("typewell_exact","master","master_frame","horizontal_exact","horizontal_round","trajectory_shape","prefix_shape","near_twin_cluster")}
with contextlib.redirect_stdout(io.StringIO()):s=runpy.run_path(str(ROOT/"exp/meta_all_honest_oof.py"))
L,y,wells,common=s["legs"],s["y"],s["wells"],s["common"];names=["har_physics","har_lgb","har_xgb","pil_blend_oof_postprocessed","v4_lgb7"]
heel=L["v4_lgb7"]+np.load(ROOT/"exp/results/heel_calibrated_gr_datum/oof.npz")["correction"][common];Z=np.column_stack([*[L[k] for k in names],heel])
def rm(p,mask=None):
 if mask is None:mask=np.ones(len(y),bool)
 return float(np.sqrt(np.mean((p[mask]-y[mask])**2)))
def meta_cv(groups):
 p=np.zeros(len(y));fid=np.zeros(len(y),int)
 for fold,(tr,va) in enumerate(GroupKFold(5).split(Z,groups=groups)):
  m=Ridge(alpha=100,positive=True,fit_intercept=False).fit(Z[tr[::8]],y[tr[::8]]);p[va]=m.predict(Z[va]);fid[va]=fold
 return p,fid
wellp,fw=meta_cv(wells);master_groups=pd.Series(wells).map(M.master_frame.to_dict()).to_numpy();masterp,fm=meta_cv(master_groups)
twin_groups=pd.Series(wells).map(M.near_twin_cluster.to_dict()).to_numpy();twinp,ft=meta_cv(twin_groups)
# Representative-only metrics remove every extra member of exact/near clusters.
domains={}
for kind in ("horizontal_exact","prefix_shape","near_twin_cluster"):
 rep=set(M.reset_index().groupby(kind,sort=False).first().well);mask=np.array([w in rep for w in wells])
 domains[kind]={"representative_wells":len(rep),"rows":int(mask.sum()),"accepted_wellfold":rm(wellp,mask),
                "masterfold":rm(masterp,mask),"near_twinfold":rm(twinp,mask)}
size=pd.Series(wells).map(M.near_twin_size.to_dict()).to_numpy();domains["near_twin_members"]={
 "wells":int(np.unique(wells[size>1]).size),"accepted_rmse":rm(wellp,size>1),"masterfold_rmse":rm(masterp,size>1)}
# Rebuild expert Gram only to score the saved continuous oracle on singleton/twin domains.
fmq=np.load(ROOT/"exp/results/heel_calibrated_gr_datum/full_meta/oof.npz");vf=pd.read_pickle(ROOT/"r_v4b/train_feats.pkl").iloc[common].reset_index(drop=True)
def project(p,deg):
 out=p.copy()
 for _,ii in pd.Series(np.arange(len(y))).groupby(wells,sort=False):
  ix=ii.to_numpy();x=vf.d_md.to_numpy(float)[ix];x=2*(x-x.min())/max(np.ptp(x),1e-6)-1;s0=p[ix]+vf.d_z.to_numpy(float)[ix]
  out[ix]=np.polyval(np.polyfit(x,s0,deg),x)-vf.d_z.to_numpy(float)[ix]
 return out
P=np.column_stack([fmq["add"],fmq["base"],heel,L["v4_lgb7"],L["har_physics"],L["har_lgb"],L["har_xgb"],
 L["pil_blend_oof_postprocessed"],project(L["v4_lgb7"].copy(),2),project(L["v4_lgb7"].copy(),3)])
W=np.load(ROOT/"exp/results/continuous_well_moe/oof_weights.npz")["target"];ordered=pd.Series(wells).drop_duplicates().to_numpy()
assert np.array_equal(ordered.astype(str),np.load(ROOT/"exp/results/continuous_well_moe/oof_weights.npz",allow_pickle=True)["wells"].astype(str))
oracle_pred=np.zeros(len(y))
for i,(_,ii) in enumerate(pd.Series(np.arange(len(y))).groupby(wells,sort=False)):
 ix=ii.to_numpy();oracle_pred[ix]=P[ix]@W[i]
summary={"train_wells":len(M),"fingerprints":counts,"meta":{"ordinary_wellfold":rm(wellp),"master_groupfold":rm(masterp),
 "near_twin_groupfold":rm(twinp)},"domains":domains,"continuous_oracle_lambda_0.5":{"all":rm(oracle_pred),
 "near_twin_members":rm(oracle_pred,size>1),"singletons":rm(oracle_pred,size==1)},
 "v4_note":"Stack-V4 base OOF already groups by 57 typewell supertypes; this audit changes second-level meta folds only.",
 "conclusion":"Cluster counts and grouped degradation quantify leakage; fingerprints use only train/test-common fields."}
M.to_csv(OUT/"well_fingerprints.csv");(OUT/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
