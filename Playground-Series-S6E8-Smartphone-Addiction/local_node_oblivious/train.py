"""Memory-conscious NODE-style soft oblivious-tree ensemble for local RTX.

Official data only. All target-free preprocessing is fitted per outer-fit fold.
The fixed model and schedule never inspect outer validation labels for selection.
"""
from pathlib import Path
import gc, json, random, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED, FOLDS, EPOCHS = "addicted_label", "id", 6082067, 5, 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def engineer(x):
    z=x.copy(); eps=.5
    for name,a,b in [("social_share","social_media_hours","daily_screen_time_hours"),("gaming_share","gaming_hours","daily_screen_time_hours"),
                     ("weekend_ratio","weekend_screen_time","daily_screen_time_hours"),("open_intensity","app_opens_per_day","daily_screen_time_hours"),
                     ("notification_intensity","notifications_per_day","app_opens_per_day"),("screen_sleep","daily_screen_time_hours","sleep_hours")]:
        if a in z and b in z: z[name]=z[a]/(z[b].abs()+eps)
    if {"daily_screen_time_hours","sleep_hours","work_study_hours"}<=set(z): z["day_residual"]=24-z.daily_screen_time_hours-z.sleep_hours-z.work_study_hours
    z["missing_count"]=x.isna().sum(1).astype("float32")
    for c in x: z[c+"__missing"]=x[c].isna().astype("float32")
    return z


class Encoder:
    def fit(self,x):
        self.num=[c for c in x if pd.api.types.is_numeric_dtype(x[c])]; self.cat=[c for c in x if c not in self.num]
        self.med=x[self.num].median(); self.qt=QuantileTransformer(n_quantiles=min(1000,len(x)),output_distribution="normal",subsample=250000,random_state=SEED).fit(x[self.num].fillna(self.med))
        self.levels={c:list(x[c].astype("string").fillna("__NA__").unique()) for c in self.cat}
        self.names=self.num+[f"{c}={v}" for c in self.cat for v in self.levels[c]]; return self
    def transform(self,x):
        p=[self.qt.transform(x[self.num].fillna(self.med)).astype("float32")]
        for c in self.cat:
            s=x[c].astype("string").fillna("__NA__"); p.append(np.column_stack([(s==v).to_numpy("float32") for v in self.levels[c]]))
        return np.ascontiguousarray(np.concatenate(p,1),dtype="float32")


class ObliviousTreeEnsemble(nn.Module):
    """Trees share a split feature/threshold at each depth, as oblivious trees do."""
    def __init__(self,n_features,trees=64,depth=5):
        super().__init__(); self.trees,self.depth=trees,depth
        self.feature_logits=nn.Parameter(torch.zeros(trees,depth,n_features))
        nn.init.normal_(self.feature_logits,std=.02)
        self.thresholds=nn.Parameter(torch.empty(trees,depth).uniform_(-1,1))
        self.log_temperature=nn.Parameter(torch.full((trees,depth),-1.2))
        self.leaves=nn.Parameter(torch.empty(trees,2**depth).normal_(0,.03))
        self.tree_weight=nn.Parameter(torch.full((trees,),1/trees))
        self.bias=nn.Parameter(torch.zeros(()))
    def forward(self,x,return_selection=False):
        select=torch.softmax(self.feature_logits,dim=-1)                 # T,D,F
        values=torch.einsum("bf,tdf->btd",x,select)
        temp=self.log_temperature.exp().clamp(.08,2.)
        right=torch.sigmoid((values-self.thresholds)/temp)               # B,T,D
        path=torch.ones((x.shape[0],self.trees,1),device=x.device,dtype=x.dtype)
        for d in range(self.depth):
            p=right[:,:,d:d+1]; path=torch.cat([path*(1-p),path*p],dim=2)
        tree=(path*self.leaves.unsqueeze(0)).sum(2)
        out=(tree*self.tree_weight).sum(1)+self.bias
        return (out,select) if return_selection else out


def predict(model,x,batch=8192):
    model.eval(); out=[]
    with torch.no_grad():
        for q in range(0,len(x),batch):
            a=torch.from_numpy(x[q:q+batch]).to(DEVICE)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=DEVICE=="cuda"): out.append(model(a).float().cpu().numpy())
    return np.concatenate(out)


def run_fold(xf,xv,xt,yf,fold,epochs=EPOCHS):
    enc=Encoder().fit(xf); a,b,c=enc.transform(xf),enc.transform(xv),enc.transform(xt)
    torch.manual_seed(SEED+fold); model=ObliviousTreeEnsemble(a.shape[1]).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=2e-5); bs=2048
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); lossf=nn.BCEWithLogitsLoss(); scaler=torch.amp.GradScaler("cuda",enabled=DEVICE=="cuda")
    y=torch.from_numpy(np.asarray(yf,dtype="float32").copy()); gen=torch.Generator().manual_seed(SEED+fold)
    for ep in range(epochs):
        model.train(); losses=[]; perm=torch.randperm(len(a),generator=gen)
        for q in range(0,len(a),bs):
            ix=perm[q:q+bs]; xx,yy=torch.from_numpy(a[ix]).to(DEVICE),y[ix].to(DEVICE); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=DEVICE=="cuda"):
                pred,sel=model(xx,True); entropy=-(sel.clamp_min(1e-8)*sel.clamp_min(1e-8).log()).sum(-1).mean(); loss=lossf(pred,yy)+1e-5*entropy
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),5.); scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
        sched.step(); print(f"fold {fold} epoch {ep+1}/{epochs} loss {np.mean(losses):.6f}",flush=True)
    pv,pt=predict(model,b),predict(model,c)
    with torch.no_grad(): importance=torch.softmax(model.feature_logits,dim=-1).mean((0,1)).cpu().numpy()
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pv,pt,dict(zip(enc.names,map(float,importance)))


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tr,te=pd.read_csv("train.csv"),pd.read_csv("test.csv"); cols=[c for c in te if c!=ID]
    x,xt,y=engineer(tr[cols]),engineer(te[cols]),tr[TARGET].to_numpy("float32")
    oof=np.zeros(len(tr),"float32"); fid=np.zeros(len(tr),"int8"); tests=[]; rows=[]; imps=[]; start=time.time()
    for fold,(fit,val) in enumerate(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(x,y),1):
        pv,pt,imp=run_fold(x.iloc[fit],x.iloc[val],xt,y[fit],fold); oof[val]=pv; fid[val]=fold; tests.append(pt); imps.append(imp)
        rows.append({"fold":fold,"auc":roc_auc_score(y[val],pv)}); print(rows[-1],flush=True)
    out=Path("local_node_oblivious/output"); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({ID:tr[ID],"fold":fid,"y":y,"pred":oof}).to_csv(out/"oof_node.csv",index=False)
    pd.DataFrame({ID:te[ID],TARGET:np.mean(tests,0)}).to_csv(out/"test_node.csv",index=False)
    names=sorted(imps[0]); pd.DataFrame({"feature":names,"selection_weight":[np.mean([q[n] for q in imps]) for n in names]}).sort_values("selection_weight",ascending=False).to_csv(out/"feature_selection.csv",index=False)
    met={"model":"NODE-style differentiable oblivious tree ensemble","official_data_only":True,"folds":FOLDS,"fixed_epochs":EPOCHS,"trees":64,"depth":5,"tuning":"none","fold_metrics":rows,"oof_auc":roc_auc_score(y,oof),"runtime_seconds":time.time()-start,"submission_created":False}
    (out/"metrics_node.json").write_text(json.dumps(met,indent=2)); print(json.dumps(met,indent=2))
if __name__=="__main__": main()
