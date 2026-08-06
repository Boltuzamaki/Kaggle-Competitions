"""Local RTX mirror of the strict Kaggle TabICL residual-FPCA protocol."""
from pathlib import Path
import hashlib, json, time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor

ROOT=Path(__file__).resolve().parents[1];IN=ROOT/'exp/kaggle_tabicl_v6_fpca_dataset'
OUT=ROOT/'exp/results/tabicl_v6_fpca_local_v1';OUT.mkdir(parents=True,exist_ok=True)
MODEL=str(ROOT/'exp/public_artifacts/tabicl/tabicl-regressor-v2-20260212.ckpt')
z=np.load(IN/'legal_v6_fpca_cache.npz',allow_pickle=True);prov=json.loads((IN/'provenance.json').read_text())
assert hashlib.sha256((IN/'legal_v6_fpca_cache.npz').read_bytes()).hexdigest()==prov['sha256']
W=z['wells'].astype(str);Q=z['query'].astype(np.float32);BT=z['backtest'].astype(np.float32)
C=z['residual_curve'].astype(np.float32);L=z['lengths'].astype(float)
splits=list(GroupKFold(5).split(Q,groups=W));P=np.zeros_like(C);fid=np.full(len(W),-1,np.int8);fold_rows=[]
for f,(tr,va) in enumerate(splits):
 tq=StandardScaler().fit(Q[tr]);qp=PCA(64,whiten=True,random_state=8600+f).fit(tq.transform(Q[tr]));tb=StandardScaler().fit(BT[tr])
 Xtr=np.c_[qp.transform(tq.transform(Q[tr])),tb.transform(BT[tr])].astype(np.float32)
 Xva=np.c_[qp.transform(tq.transform(Q[va])),tb.transform(BT[va])].astype(np.float32)
 cp=PCA(12,random_state=8700+f).fit(C[tr]);Y=cp.transform(C[tr]);Yp=np.zeros((len(va),12),np.float32);started=time.time()
 for j in range(12):
  m=TabICLRegressor(n_estimators=4,batch_size=4,kv_cache=False,model_path=MODEL,allow_auto_download=False,device='cuda',use_amp=True,random_state=8800+f*20+j,verbose=False)
  m.fit(Xtr,Y[:,j]);Yp[:,j]=m.predict(Xva)
  print(json.dumps({'fold':f,'coefficient':j,'elapsed_min':(time.time()-started)/60}),flush=True)
 P[va]=cp.inverse_transform(Yp);fid[va]=f
 sc=lambda p:float(np.sqrt(np.average(np.mean((C[va]-p)**2,1),weights=L[va])))
 fold_rows.append({'fold':f,'wells':len(va),'zero':sc(np.zeros_like(C[va])),'tabicl':sc(P[va])});print('FOLD',json.dumps(fold_rows[-1]),flush=True)
 np.savez_compressed(OUT/'partial_oof.npz',wells=W,pred=P,true=C,lengths=L,fold=fid)
score=lambda p:float(np.sqrt(np.average(np.mean((C-p)**2,1),weights=L)))
v6=np.load(IN/'v6_correction.npz')['pred'];grid=[{'alpha_tabicl':a,'rmse':score((1-a)*v6+a*P)} for a in (0,.1,.2,.3,.4,.5,.7,1)]
summary={'protocol':'strict outer whole-well GKF; fold-local preprocessing/FPCA/TabICL; immutable cache '+prov['sha256'],'zero':score(np.zeros_like(C)),'v6':score(v6),'tabicl':score(P),'blend_grid':grid,'best_blend':min(grid,key=lambda x:x['rmse']),'folds':fold_rows}
np.savez_compressed(OUT/'oof.npz',wells=W,pred=P,v6=v6,true=C,lengths=L,fold=fid);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
