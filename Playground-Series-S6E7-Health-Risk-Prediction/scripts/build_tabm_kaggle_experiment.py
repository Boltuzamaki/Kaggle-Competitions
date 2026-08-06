"""Build a five-fold TabM GPU experiment using the official tabm package."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"kaggle_kernels"/"rule_tabm_gpu"
SRC=r'''
!pip install -q tabm rtdl_num_embeddings
import gc,json,time
from copy import deepcopy
from pathlib import Path
import numpy as np,pandas as pd,torch,tabm
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
ID="id";TARGET="health_condition";SEED=2027;FOLDS=5
C=np.array(["at-risk","fit","unhealthy"]);M={c:i for i,c in enumerate(C)}
base=Path("/kaggle/input/competitions/playground-series-s6e7")
tr=pd.read_csv(base/"train.csv");te=pd.read_csv(base/"test.csv");y=tr[TARGET].map(M).to_numpy(np.int64)
def fe(d):
 z=d.drop(columns=[ID,TARGET],errors="ignore").copy();s=z.sleep_duration
 z["sleep_lt6"]=np.where(s.isna(),np.nan,(s<6).astype(float));z["sleep_lt7"]=np.where(s.isna(),np.nan,(s<7).astype(float))
 z["key_missing_count"]=z[["sleep_duration","stress_level","physical_activity_level"]].isna().sum(1)
 z["rule_unhealthy"]=((s<6)&z.stress_level.eq("high")).astype(float);z["rule_fit"]=((s>=7)&z.stress_level.eq("low")&z.physical_activity_level.eq("active")).astype(float)
 return z
X=fe(tr);T=fe(te);cats=list(X.select_dtypes("object").columns);nums=[c for c in X if c not in cats]
enc=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1);enc.fit(pd.concat([X[cats],T[cats]]).fillna("NA"))
XC=enc.transform(X[cats].fillna("NA")).astype(np.int64)+1;TC=enc.transform(T[cats].fillna("NA")).astype(np.int64)+1
cards=[int(max(XC[:,j].max(),TC[:,j].max()))+1 for j in range(XC.shape[1])]
med=X[nums].median();XN=X[nums].fillna(med).to_numpy(np.float32);TN=T[nums].fillna(med).to_numpy(np.float32)
dev=torch.device("cuda");oof=np.zeros((len(X),3),np.float32);pred=np.zeros((len(T),3));fs=[];t0=time.time()
def infer(model,xn,xc,bs=8192):
 model.eval();out=[]
 with torch.inference_mode():
  for s in range(0,len(xn),bs):
   logits=model(torch.as_tensor(xn[s:s+bs],device=dev),torch.as_tensor(xc[s:s+bs],device=dev))
   out.append(logits.softmax(-1).mean(1).cpu().numpy())
 return np.vstack(out)
skf=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED)
for f,(a,b) in enumerate(skf.split(XN,y),1):
 scale=StandardScaler().fit(XN[a]);an=scale.transform(XN[a]).astype(np.float32);bn=scale.transform(XN[b]).astype(np.float32);tn=scale.transform(TN).astype(np.float32)
 model=tabm.TabM.make(n_num_features=an.shape[1],cat_cardinalities=cards,d_out=3,num_embeddings=None).to(dev)
 opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=3e-4)
 cw=torch.as_tensor([len(a)/(3*(y[a]==k).sum()) for k in range(3)],device=dev,dtype=torch.float32)
 best=-1;state=None;bad=0;rng=np.random.default_rng(SEED+f)
 for epoch in range(20):
  model.train();order=rng.permutation(len(a))
  for s in range(0,len(a),1024):
   q=order[s:s+1024];xn=torch.as_tensor(an[q],device=dev);xc=torch.as_tensor(XC[a[q]],device=dev);yy=torch.as_tensor(y[a[q]],device=dev)
   logits=model(xn,xc);loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),yy.repeat_interleave(model.backbone.k),weight=cw)
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
  vp=infer(model,bn,XC[b]);score=balanced_accuracy_score(y[b],vp.argmax(1));print(f,epoch,score)
  if score>best:best=score;state=deepcopy(model.state_dict());bad=0
  else:bad+=1
  if bad>=5:break
 model.load_state_dict(state);oof[b]=infer(model,bn,XC[b]);pred+=infer(model,tn,TC)/FOLDS;fs.append(float(balanced_accuracy_score(y[b],oof[b].argmax(1))))
 del model,opt,state;gc.collect();torch.cuda.empty_cache()
overall=float(balanced_accuracy_score(y,oof.argmax(1)));od=pd.DataFrame({ID:tr[ID]});td=pd.DataFrame({ID:te[ID]})
for j,c in enumerate(C):od[c]=oof[:,j];td[c]=pred[:,j]
od.to_csv("oof_preds.csv",index=False);td.to_csv("test_preds.csv",index=False);pd.DataFrame({ID:te[ID],TARGET:C[pred.argmax(1)]}).to_csv("submission.csv",index=False)
Path("training_summary.json").write_text(json.dumps({"experiment":"tabm","fold_scores":fs,"oof_balanced_accuracy":overall,"elapsed_minutes":(time.time()-t0)/60},indent=2));print(overall)
'''
def main():
 OUT.mkdir(parents=True,exist_ok=True);compile("\n".join(x for x in SRC.splitlines() if not x.startswith("!")),"tabm","exec")
 nb={"cells":[{"cell_type":"code","execution_count":None,"id":"tabm-main","metadata":{},"outputs":[],"source":SRC.splitlines(keepends=True)}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}
 (OUT/"rule_tabm_gpu.ipynb").write_text(json.dumps(nb,indent=1))
 meta={"id":"boltuzamaki/health-risk-rule-tabm-gpu","title":"Health Risk Rule TabM GPU","code_file":"rule_tabm_gpu.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_internet":True,"dataset_sources":[],"kernel_sources":[],"competition_sources":["playground-series-s6e7"],"machine_shape":"NvidiaTeslaT4"}
 (OUT/"kernel-metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
if __name__=="__main__":main()
