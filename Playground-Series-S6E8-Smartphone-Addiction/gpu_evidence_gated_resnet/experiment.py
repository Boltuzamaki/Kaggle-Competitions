"""Honest nested-evidence gated residual MLP, official competition data only.

Each outer-fit row's supervised evidence is inner OOF. Outer-validation and test
evidence is fitted on the complete outer-fit partition. Architecture and epochs
are fixed before outer CV; outer validation labels are reporting-only.
"""
from pathlib import Path
import subprocess, sys
if Path('/kaggle').exists():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--index-url',
                           'https://download.pytorch.org/whl/cu121', 'torch==2.5.1'])
import gc, json, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

TARGET, ID, SEED, OUTER, INNER, EPOCHS = 'addicted_label', 'id', 6082026, 5, 5, 22
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
AMP = torch.bfloat16 if DEV == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
SCALER = DEV == 'cuda' and AMP == torch.float16
PAIRS = [('daily_screen_time_hours','sleep_hours'), ('daily_screen_time_hours','notifications_per_day'),
         ('daily_screen_time_hours','app_opens_per_day'), ('daily_screen_time_hours','stress_level'),
         ('daily_screen_time_hours','academic_work_impact'), ('social_media_hours','gaming_hours'),
         ('weekend_screen_time','daily_screen_time_hours')]

def locate():
    for root in (Path('/kaggle/input'), Path('.')):
        if not root.exists(): continue
        for p in root.rglob('train.csv'):
            try: cols = pd.read_csv(p, nrows=1).columns
            except Exception: continue
            if TARGET in cols and (p.parent/'test.csv').exists(): return p, p.parent/'test.csv'
    raise FileNotFoundError

def key(s): return s.astype('string').fillna('__NA__')

def make_keys(x):
    out = {'single__'+c:key(x[c]) for c in x}
    for a,b in PAIRS:
        if a in x and b in x: out[f'pair__{a}__{b}'] = key(x[a])+'\x1f'+key(x[b])
    return pd.DataFrame(out, index=x.index)

def map_evidence(fk, ak, fy, smooth):
    prior = float(np.mean(fy))
    s = pd.DataFrame({'k':np.asarray(fk), 'y':np.asarray(fy)}).groupby('k', sort=False)['y'].agg(['sum','count'])
    a = pd.Series(np.asarray(ak)); count = a.map(s['count']).fillna(0).to_numpy('float64')
    total = a.map(s['sum']).fillna(0).to_numpy('float64')
    rate = (total + smooth*prior)/(count+smooth)
    # Center rates around the fold-local prior; support remains an explicit gate input.
    logit = lambda z: np.log(np.clip(z, 1e-5, 1-1e-5)/np.clip(1-z, 1e-5, 1-1e-5))
    return (logit(rate)-logit(prior)).astype('float32'), np.log1p(count).astype('float32')

def fold_evidence(ktr, kte, y, fit, val, fold):
    fit,val=np.asarray(fit),np.asarray(val); yf=y[fit]; p=ktr.shape[1]
    ef=np.zeros((len(fit),p,2),'float32'); ev=np.zeros((len(val),p,2),'float32'); et=np.zeros((len(kte),p,2),'float32')
    splits=list(StratifiedKFold(INNER,shuffle=True,random_state=SEED+fold*100).split(fit,yf)); cover=np.zeros(len(fit),'int8')
    for j,c in enumerate(ktr):
        smooth=48. if c.startswith('pair__') else 24.
        for ia,ib in splits:
            ef[ib,j,0],ef[ib,j,1]=map_evidence(ktr.iloc[fit[ia]][c],ktr.iloc[fit[ib]][c],yf[ia],smooth)
            if j==0: cover[ib]+=1
        ev[:,j,0],ev[:,j,1]=map_evidence(ktr.iloc[fit][c],ktr.iloc[val][c],yf,smooth)
        et[:,j,0],et[:,j,1]=map_evidence(ktr.iloc[fit][c],kte[c],yf,smooth)
    assert np.all(cover==1)
    return ef,ev,et

def encode_raw(fit,val,test):
    """Fold-local exact embeddings plus fold-local quantiles and missing flags."""
    ids=[np.zeros((len(z),fit.shape[1]),'int64') for z in (fit,val,test)]; cards=[]
    nums=[np.zeros((len(z),fit.shape[1]),'float32') for z in (fit,val,test)]
    miss=[z.isna().to_numpy('float32') for z in (fit,val,test)]
    for j,c in enumerate(fit):
        vals=[v for v in key(fit[c]).unique() if v!='__NA__']; mp={v:i+2 for i,v in enumerate(vals)}; cards.append(len(vals)+2)
        for out,z in zip(ids,(fit,val,test)):
            kz=key(z[c]); q=kz.map(mp).fillna(1).to_numpy('int64'); q[kz.eq('__NA__').to_numpy()]=0; out[:,j]=q
        if pd.api.types.is_numeric_dtype(fit[c]):
            ok=fit[c].notna()
            if ok.any():
                qt=QuantileTransformer(n_quantiles=min(1000,int(ok.sum())),output_distribution='normal',subsample=300000,random_state=SEED+j).fit(fit.loc[ok,[c]])
                for out,z in zip(nums,(fit,val,test)):
                    good=z[c].notna(); out[good.to_numpy(),j]=qt.transform(z.loc[good,[c]]).ravel().astype('float32')
    return ids,nums,miss,cards

class Block(nn.Module):
    def __init__(self,d,drop=.12):
        super().__init__(); self.f=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d*2),nn.GELU(),nn.Dropout(drop),nn.Linear(d*2,d),nn.Dropout(drop))
    def forward(self,x): return x+self.f(x)

class Net(nn.Module):
    """Two towers: raw ResNet and evidence ResNet, fused by reliability gate."""
    def __init__(self,cards,nkeys,d=256):
        super().__init__(); self.emb=nn.ModuleList([nn.Embedding(c,16) for c in cards]); n=len(cards)
        self.raw_in=nn.Linear(n*16+n*2,d); self.raw=nn.Sequential(Block(d),Block(d),Block(d))
        self.ev_token=nn.Sequential(nn.Linear(2,32),nn.GELU(),nn.Linear(32,32)); self.ev_in=nn.Linear(nkeys*32,d)
        self.ev=nn.Sequential(Block(d,.08),Block(d,.08),Block(d,.08))
        self.gate=nn.Sequential(nn.Linear(nkeys+2,d//2),nn.GELU(),nn.Linear(d//2,d),nn.Sigmoid())
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Dropout(.1),nn.Linear(128,1))
    def forward(self,ids,num,miss,e):
        cat=torch.cat([m(ids[:,j]) for j,m in enumerate(self.emb)],1)
        r=self.raw(self.raw_in(torch.cat([cat,num,miss],1)))
        v=self.ev(self.ev_in(self.ev_token(e).flatten(1)))
        support=e[:,:,1]; reliability=torch.cat([support, support.mean(1,keepdim=True), (support>0).float().mean(1,keepdim=True)],1)
        g=self.gate(reliability)
        return self.head(r+g*v).squeeze(-1)

def run_fold(fit,val,x,xt,k,kt,y,fold):
    ids,nums,miss,cards=encode_raw(x.iloc[fit],x.iloc[val],xt); ef,ev,et=fold_evidence(k,kt,y,fit,val,fold)
    ten=lambda a:torch.from_numpy(a); data=list(map(ten,(*ids,*nums,*miss,ef,ev,et)))
    II,IV,IT,NI,NV,NT,MI,MV,MT,EI,EV,ET=data; Y=ten(y[fit])
    torch.manual_seed(SEED+fold); model=Net(cards,k.shape[1]).to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-3,weight_decay=2e-4); bs=2048
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,1.5e-3,total_steps=((len(fit)+bs-1)//bs)*EPOCHS,pct_start=.15)
    scaler=torch.amp.GradScaler('cuda',enabled=SCALER); lossf=nn.BCEWithLogitsLoss(); gen=torch.Generator().manual_seed(SEED+fold)
    for ep in range(EPOCHS):
        model.train(); losses=[]; perm=torch.randperm(len(fit),generator=gen)
        for q in range(0,len(fit),bs):
            sl=perm[q:q+bs]
            # Evidence dropout forces useful raw/evidence complementarity.
            ee=EI[sl].to(DEV); drop=(torch.rand((len(sl),ee.shape[1],1),device=DEV)<.08); ee=ee.masked_fill(drop,0)
            with torch.autocast(device_type=DEV,dtype=AMP,enabled=DEV=='cuda'):
                loss=lossf(model(II[sl].to(DEV),NI[sl].to(DEV),MI[sl].to(DEV),ee),Y[sl].to(DEV))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt); scaler.update(); sched.step(); losses.append(float(loss.detach()))
        print(f'fold {fold} epoch {ep+1}/{EPOCHS} train_loss {np.mean(losses):.6f}',flush=True)
    def pred(a,b,c,e):
        model.eval(); out=[]
        with torch.no_grad():
            for q in range(0,len(a),8192):
                with torch.autocast(device_type=DEV,dtype=AMP,enabled=DEV=='cuda'): out.append(model(a[q:q+8192].to(DEV),b[q:q+8192].to(DEV),c[q:q+8192].to(DEV),e[q:q+8192].to(DEV)).float().cpu())
        return torch.cat(out).numpy()
    pv,pt=pred(IV,NV,MV,EV),pred(IT,NT,MT,ET); del model,data
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect(); return pv,pt

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(torch.__version__,torch.cuda.get_arch_list(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEV)
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(6,0): assert 'sm_60' in torch.cuda.get_arch_list()
    tp,sp=locate(); tr,te=pd.read_csv(tp),pd.read_csv(sp); cols=[c for c in te if c!=ID]; x,xt=tr[cols],te[cols]
    k,kt=make_keys(x),make_keys(xt); y=tr[TARGET].to_numpy('float32'); oof=np.zeros(len(tr)); fid=np.zeros(len(tr),'int8'); tests=[]; rows=[]; t0=time.time()
    for fold,(fit,val) in enumerate(StratifiedKFold(OUTER,shuffle=True,random_state=SEED).split(x,y),1):
        fid[val]=fold; pv,pt=run_fold(fit,val,x,xt,k,kt,y,fold); oof[val]=pv; tests.append(pt); rows.append({'fold':fold,'auc':roc_auc_score(y[val],pv)}); print(rows[-1])
    out=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('artifacts/local_evidence_gated_resnet'); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({ID:tr[ID],'fold':fid,'y':y,'pred':oof}).to_csv(out/'oof_evidence_gated_resnet.csv',index=False)
    pd.DataFrame({ID:te[ID],TARGET:np.mean(tests,axis=0)}).to_csv(out/'test_evidence_gated_resnet.csv',index=False)
    metrics={'model':'nested-evidence support-gated residual MLP','official_data_only':True,'seed':SEED,'outer_folds':OUTER,'inner_folds':INNER,'fixed_epochs':EPOCHS,'hyperparameter_policy':'single prespecified bounded configuration; no outer-fold selection','evidence_contract':'outer-fit inner-OOF; validation/test complete outer-fit mapping','fold_metrics':rows,'oof_auc':roc_auc_score(y,oof),'fold_mean':float(np.mean([r['auc'] for r in rows])),'runtime_seconds':time.time()-t0,'submission_created':False}
    (out/'metrics_evidence_gated_resnet.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
