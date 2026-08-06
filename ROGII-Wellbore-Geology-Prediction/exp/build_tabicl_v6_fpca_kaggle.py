"""Build immutable legal cache and private Kaggle GPU notebook for TabICL FPCA CV."""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "exp/kaggle_tabicl_v6_fpca_dataset"
NB = ROOT / "kernels/tabicl_v6_fpca_cv"
DS.mkdir(parents=True, exist_ok=True); NB.mkdir(parents=True, exist_ok=True)

v = np.load(ROOT / "exp/results/generative_curve_prior_causal_prefix_v1/well_curves.npz")
b = np.load(ROOT / "exp/results/prefix_backtest_moe/backtest_features.npz", allow_pickle=True)
bm = {str(w): i for i, w in enumerate(b["wells"])}
W = v["wells"].astype(str)
BT = b["features"][[bm[w] for w in W]].astype(np.float32)
np.savez_compressed(DS / "legal_v6_fpca_cache.npz", wells=W,
                    query=v["query"].astype(np.float32), backtest=BT,
                    residual_curve=v["true"].astype(np.float32),
                    lengths=v["length"].astype(np.int32))
sha = hashlib.sha256((DS / "legal_v6_fpca_cache.npz").read_bytes()).hexdigest()
(DS / "provenance.json").write_text(json.dumps({
    "sha256": sha, "wells": len(W), "query_features": v["query"].shape[1],
    "causal_backtest_features": BT.shape[1], "curve_points": v["true"].shape[1],
    "labels": "raw-derived Student residual curves; no external prediction labels",
    "protocol": "all PCA/scaling/model fits occur inside outer GroupKFold"
}, indent=2))
(DS / "dataset-metadata.json").write_text(json.dumps({
    "title": "ROGII Legal TabICL V6 FPCA Cache",
    "id": "boltuzamaki/rogii-legal-tabicl-v6-fpca-cache",
    "licenses": [{"name": "CC0-1.0"}], "isPrivate": True
}, indent=2))

code = r'''
from pathlib import Path
import json, hashlib, subprocess, sys, time
import numpy as np
wheel=next(Path('/kaggle/input').rglob('tabicl-2.1.1-py3-none-any.whl'))
subprocess.check_call([sys.executable,"-m","pip","install","-q","--force-reinstall",
 "torch==2.5.1","--index-url","https://download.pytorch.org/whl/cu121"])
subprocess.check_call([sys.executable,"-m","pip","install","-q","--no-deps",str(wheel)])
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor

IN=next(Path('/kaggle/input').rglob('legal_v6_fpca_cache.npz')).parent
MODEL=str(next(Path('/kaggle/input').rglob('tabicl-regressor-v2-20260212.ckpt')))
z=np.load(IN/"legal_v6_fpca_cache.npz",allow_pickle=True)
W=z["wells"].astype(str);Q=z["query"].astype(np.float32);BT=z["backtest"].astype(np.float32)
C=z["residual_curve"].astype(np.float32);L=z["lengths"].astype(float)
assert hashlib.sha256((IN/"legal_v6_fpca_cache.npz").read_bytes()).hexdigest()==json.loads((IN/"provenance.json").read_text())["sha256"]
splits=list(GroupKFold(5).split(Q,groups=W));P=np.zeros_like(C);fid=np.full(len(W),-1,np.int8);fold_rows=[]
for f,(tr,va) in enumerate(splits):
    tq=StandardScaler().fit(Q[tr]);qp=PCA(64,whiten=True,random_state=8600+f).fit(tq.transform(Q[tr]))
    tb=StandardScaler().fit(BT[tr]);Xtr=np.c_[qp.transform(tq.transform(Q[tr])),tb.transform(BT[tr])].astype(np.float32)
    Xva=np.c_[qp.transform(tq.transform(Q[va])),tb.transform(BT[va])].astype(np.float32)
    cp=PCA(12,random_state=8700+f).fit(C[tr]);Y=cp.transform(C[tr]);Yp=np.zeros((len(va),12),np.float32)
    started=time.time()
    for j in range(12):
        m=TabICLRegressor(n_estimators=4,batch_size=4,kv_cache=False,model_path=MODEL,
            allow_auto_download=False,device="cuda",use_amp=True,random_state=8800+f*20+j,verbose=False)
        m.fit(Xtr,Y[:,j]);Yp[:,j]=m.predict(Xva)
        print(json.dumps({"fold":f,"coefficient":j,"elapsed_min":(time.time()-started)/60}),flush=True)
    P[va]=cp.inverse_transform(Yp);fid[va]=f
    def sc(p):return float(np.sqrt(np.average(np.mean((C[va]-p)**2,1),weights=L[va])))
    fold_rows.append({"fold":f,"wells":len(va),"zero":sc(np.zeros_like(C[va])),"tabicl":sc(P[va])})
    print("FOLD",json.dumps(fold_rows[-1]),flush=True)
def score(p):return float(np.sqrt(np.average(np.mean((C-p)**2,1),weights=L)))
v6=np.load('/kaggle/input/rogii-legal-tabicl-v6-fpca-cache/v6_correction.npz')["pred"]
grid=[]
for a in (0,.1,.2,.3,.4,.5,.7,1):grid.append({"alpha_tabicl":a,"rmse":score((1-a)*v6+a*P)})
summary={"protocol":"strict outer whole-well GKF; fold-local query PCA/scalers, residual FPCA and TabICL; no external prediction labels",
 "wells":len(W),"raw_features":int(Q.shape[1]+BT.shape[1]),"model_features":159,"coefficients":12,
 "zero":score(np.zeros_like(C)),"v6":score(v6),"tabicl":score(P),"blend_grid":grid,"best_blend":min(grid,key=lambda x:x['rmse']),"folds":fold_rows}
np.savez_compressed('/kaggle/working/oof.npz',wells=W,pred=P,v6=v6,true=C,lengths=L,fold=fid)
Path('/kaggle/working/summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
'''
# Attach immutable v6 correction separately so its hash/provenance stays explicit.
v6 = np.load(ROOT / "exp/results/generative_curve_prior_expert_errors_v6/oof.npz", allow_pickle=True)
assert np.array_equal(W, v6["wells"].astype(str))
np.savez_compressed(DS / "v6_correction.npz", pred=v6["pred"].astype(np.float32))

nb = {"cells":[{"cell_type":"code","execution_count":None,"metadata":{},
                "outputs":[],"source":code.splitlines(keepends=True)}],
      "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python","version":"3"}},
      "nbformat":4,"nbformat_minor":5}
(NB / "rogii-tabicl-v6-fpca-cv.ipynb").write_text(json.dumps(nb))
(NB / "kernel-metadata.json").write_text(json.dumps({
    "id": "boltuzamaki/rogii-tabicl-v6-fpca-cv", "title": "ROGII TabICL V6 FPCA CV",
    "code_file": "rogii-tabicl-v6-fpca-cv.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["boltuzamaki/rogii-legal-tabicl-v6-fpca-cache", "needless090/rogii-tabicl-mirror"],
    "competition_sources": [], "kernel_sources": []
}, indent=2))
print(json.dumps({"dataset":str(DS),"notebook":str(NB),"sha256":sha},indent=2))
