"""Original fold-safe exact-value DeepFM; official competition data only."""
from pathlib import Path
import subprocess,sys
if Path('/kaggle').exists():
 subprocess.check_call([sys.executable,'-m','pip','install','-q','--index-url','https://download.pytorch.org/whl/cu121','torch==2.5.1'])
import gc,json,math,random,time
import numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
TARGET,ID,SEED,FOLDS,EPOCHS='addicted_label','id',20260803,5,18; DEV='cuda' if torch.cuda.is_available() else 'cpu'; AMP=torch.bfloat16 if DEV=='cuda' and torch.cuda.is_bf16_supported() else torch.float16; SCALE=DEV=='cuda' and AMP==torch.float16
def locate():
 for root in (Path('/kaggle/input'),Path('.')):
  if not root.exists(): continue
  for p in root.rglob('train.csv'):
   try: cols=pd.read_csv(p,nrows=1).columns
   except Exception: continue
   if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
 raise FileNotFoundError
def key(s): return s.astype('string').fillna('__NA__')
def derived(d):
 c=['social_media_hours','gaming_hours','work_study_hours']; s=d[c].sum(axis=1,min_count=3)
 return pd.DataFrame({'other_screen':d.daily_screen_time_hours-s,'component_sum':s,'other_frac':(d.daily_screen_time_hours-s)/(d.daily_screen_time_hours+.1),'component_frac':s/(d.daily_screen_time_hours+.1),'weekend_gap':d.weekend_screen_time-d.daily_screen_time_hours,'weekend_other':d.weekend_screen_time-s,'screen_sleep':d.daily_screen_time_hours/(d.sleep_hours+.1),'notif_open':d.notifications_per_day/(d.app_opens_per_day+1)})
def exact(fit,val,test):
 arr=[np.zeros((len(z),fit.shape[1]),'int64') for z in (fit,val,test)]; off=[]; total=0
 for j,c in enumerate(fit):
  vals=[v for v in key(fit[c]).unique() if v!='__NA__']; mp={v:i+2 for i,v in enumerate(vals)}; off.append(total)
  for out,z in zip(arr,(fit,val,test)):
   kz=key(z[c]); q=kz.map(mp).fillna(1).to_numpy('int64'); q[kz.eq('__NA__').to_numpy()]=0; out[:,j]=q+total
  total+=len(vals)+2
 return arr,np.asarray(off,'int64'),total
def norm(fit,val,test):
 outs=[np.zeros((len(z),fit.shape[1]),'float32') for z in (fit,val,test)]; masks=[z.isna().to_numpy('float32') for z in (fit,val,test)]
 for j,c in enumerate(fit):
  if not pd.api.types.is_numeric_dtype(fit[c]): continue
  ok=fit[c].notna(); qt=QuantileTransformer(n_quantiles=min(1000,int(ok.sum())),output_distribution='normal',subsample=300000,random_state=SEED+j); qt.fit(fit.loc[ok,[c]])
  for o,z in zip(outs,(fit,val,test)):
   good=z[c].notna(); o[good.to_numpy(),j]=qt.transform(z.loc[good,[c]]).ravel().astype('float32')
 return outs,masks
class DeepFM(nn.Module):
 def __init__(self,total,nfield,nder,d=32):
  super().__init__(); self.v=nn.Embedding(total,d); self.first=nn.Embedding(total,1); nn.init.normal_(self.v.weight,std=.03); nn.init.zeros_(self.first.weight); self.num_v=nn.Parameter(torch.randn(nfield,d)*.02); self.num_first=nn.Parameter(torch.zeros(nfield)); self.deep=nn.Sequential(nn.Linear(nfield*d+nder,384),nn.BatchNorm1d(384),nn.SiLU(),nn.Dropout(.15),nn.Linear(384,192),nn.SiLU(),nn.Dropout(.10),nn.Linear(192,1)); self.bias=nn.Parameter(torch.zeros(1))
 def forward(self,ids,num,mask,der):
  fields=self.v(ids)+self.num_v[None,:,:]*num.unsqueeze(-1)*(1-mask).unsqueeze(-1); first=self.first(ids).squeeze(-1).sum(1)+(self.num_first[None,:]*num*(1-mask)).sum(1); fm=.5*((fields.sum(1)**2-fields.square().sum(1)).sum(1)); deep=self.deep(torch.cat([fields.flatten(1),der],1)).squeeze(1); return self.bias+first+fm+deep
def train_fold(fi,vi,raw,testr,der,dtest,y,fold):
 (ia,iv,it),offs,total=exact(raw.iloc[fi],raw.iloc[vi],testr); (na,nv,nt),(ma,mv,mt)=norm(raw.iloc[fi],raw.iloc[vi],testr); (da,dv,dt),_=norm(der.iloc[fi],der.iloc[vi],dtest); T=lambda x:torch.from_numpy(x); IA,IV,IT=map(T,(ia,iv,it)); NA,NV,NT=map(T,(na,nv,nt)); MA,MV,MT=map(T,(ma,mv,mt)); DA,DV,DT=map(T,(da,dv,dt)); Y=T(y[fi]); OFF=T(offs).to(DEV)
 torch.manual_seed(SEED+fold); m=DeepFM(total,raw.shape[1],der.shape[1]).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=1.2e-3,weight_decay=3e-5); bs=4096; sched=torch.optim.lr_scheduler.OneCycleLR(opt,1.2e-3,total_steps=math.ceil(len(fi)/bs)*EPOCHS,pct_start=.15); scaler=torch.cuda.amp.GradScaler(enabled=SCALE); pars=list(m.parameters()); ema=[p.detach().clone() for p in pars]; lossf=nn.BCEWithLogitsLoss()
 def predict(a,n,ms,d):
  m.eval(); out=[]
  with torch.no_grad():
   for q in range(0,len(a),16384):
    with torch.autocast(device_type=DEV,dtype=AMP,enabled=DEV=='cuda'): out.append(m(a[q:q+16384].to(DEV),n[q:q+16384].to(DEV),ms[q:q+16384].to(DEV),d[q:q+16384].to(DEV)).float().cpu())
  return torch.cat(out).numpy()
 best=-1; bestw=None; bad=0; gen=torch.Generator().manual_seed(SEED+fold)
 for ep in range(EPOCHS):
  m.train(); perm=torch.randperm(len(fi),generator=gen)
  for q in range(0,len(fi),bs):
   sl=perm[q:q+bs]; ids=IA[sl].to(DEV); num=NA[sl].to(DEV); ms=MA[sl].to(DEV); drop=torch.rand(ids.shape,device=DEV)<.08; ids=torch.where(drop,OFF.expand_as(ids),ids); ms=torch.maximum(ms,drop.float()); num=torch.where(drop,torch.zeros_like(num),num)
   with torch.autocast(device_type=DEV,dtype=AMP,enabled=DEV=='cuda'): loss=lossf(m(ids,num,ms,DA[sl].to(DEV)),Y[sl].to(DEV))
   opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(pars,2.); scaler.step(opt); scaler.update(); sched.step()
   with torch.no_grad(): torch._foreach_mul_(ema,.999); torch._foreach_add_(ema,[p.detach() for p in pars],alpha=.001)
  if ep>=3:
   bak=[p.detach().clone() for p in pars]
   with torch.no_grad():
    for p,e in zip(pars,ema): p.copy_(e)
   auc=roc_auc_score(y[vi],predict(IV,NV,MV,DV)); print(f'fold={fold} ep={ep} auc={auc:.8f}',flush=True)
   if auc>best: best,bestw,bad=auc,[e.clone() for e in ema],0
   else: bad+=1
   with torch.no_grad():
    for p,z in zip(pars,bak): p.copy_(z)
   if bad>=4: break
 with torch.no_grad():
  for p,e in zip(pars,bestw): p.copy_(e)
 pv=predict(IV,NV,MV,DV); pt=predict(IT,NT,MT,DT); del m; torch.cuda.empty_cache(); gc.collect(); return pv,pt,best
random.seed(SEED); np.random.seed(SEED); tp,sp=locate(); tr=pd.read_csv(tp); te=pd.read_csv(sp); feats=[c for c in te if c!=ID]; raw,test=tr[feats],te[feats]; der,dtest=derived(raw),derived(test); y=tr[TARGET].to_numpy('float32'); outer=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); tests=[]; fid=np.zeros(len(tr),'int8'); rows=[]; t0=time.time()
for fold,(fi,vi) in enumerate(outer.split(raw,y),1): fid[vi]=fold; pv,pt,b=train_fold(fi,vi,raw,test,der,dtest,y,fold); oof[vi]=pv; tests.append(pt); rows.append({'fold':fold,'auc':roc_auc_score(y[vi],pv),'best_ema_auc':b}); print(rows[-1])
out=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('artifacts/local_deepfm_exact'); out.mkdir(parents=True,exist_ok=True)
pd.DataFrame({ID:tr[ID],'fold':fid,'y':y,'pred':oof}).to_csv(out/'oof_exact_deepfm.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:np.mean(tests,0)}).to_csv(out/'test_exact_deepfm.csv',index=False); s=[r['auc'] for r in rows]; metrics={'model':'original fold-safe exact-value DeepFM','official_data_only':True,'seed':SEED,'folds':FOLDS,'fold_metrics':rows,'fold_mean':float(np.mean(s)),'fold_std':float(np.std(s)),'oof_auc':roc_auc_score(y,oof),'runtime_seconds':time.time()-t0,'submission_created':False}; (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
