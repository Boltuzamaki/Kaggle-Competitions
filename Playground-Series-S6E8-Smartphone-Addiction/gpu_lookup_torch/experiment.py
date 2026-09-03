"""Fold-safe source-capacity lookup transformer; official data only."""
from pathlib import Path
import subprocess,sys
# Kaggle P100 needs Pascal kernels; locally the same wheel is installed in the
# project .venv via uv, so runtime mutation is Kaggle-only.
if Path('/kaggle').exists():
 subprocess.check_call([sys.executable,'-m','pip','install','-q','--index-url','https://download.pytorch.org/whl/cu121','torch==2.5.1'])
import gc,json,math,random,time
import numpy as np,pandas as pd,torch,torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
TARGET,ID,SEED,FOLDS,EPOCHS='addicted_label','id',20260803,5,24
DEV='cuda' if torch.cuda.is_available() else 'cpu'; AMP=torch.bfloat16 if DEV=='cuda' and torch.cuda.is_bf16_supported() else torch.float16; SCALER=DEV=='cuda' and AMP==torch.float16
print(torch.__version__,torch.cuda.get_arch_list(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEV)
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)==(6,0): assert 'sm_60' in torch.cuda.get_arch_list()
def locate():
 for root in (Path('/kaggle/input'),Path('.')):
  if not root.exists(): continue
  for p in root.rglob('train.csv'):
   try: cols=pd.read_csv(p,nrows=1).columns
   except Exception: continue
   if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
 raise FileNotFoundError
def key(s): return s.astype('string').fillna('__NA__')
def deriv(d):
 c=['social_media_hours','gaming_hours','work_study_hours']; s=d[c].sum(axis=1,min_count=3)
 return pd.DataFrame({'other_screen':d.daily_screen_time_hours-s,'component_sum':s,'other_frac':(d.daily_screen_time_hours-s)/(d.daily_screen_time_hours+.1),'weekend_gap':d.weekend_screen_time-d.daily_screen_time_hours,'weekend_other':d.weekend_screen_time-s,'component_frac':s/(d.daily_screen_time_hours+.1)})
def lookups(fit,val,test):
 arr=[np.zeros((len(z),fit.shape[1]),'int64') for z in (fit,val,test)]; offsets=[]; total=0
 for j,c in enumerate(fit):
  # 0 within every column is its missing/masking ID; 1 is fold-local OOV.
  vals=[v for v in key(fit[c]).unique() if v!='__NA__']; mp={v:i+2 for i,v in enumerate(vals)}; offsets.append(total)
  for out,z in zip(arr,(fit,val,test)):
   kz=key(z[c]); local=kz.map(mp).fillna(1).to_numpy('int64'); local[kz.eq('__NA__').to_numpy()]=0; out[:,j]=local+total
  total+=len(vals)+2
 return arr,np.asarray(offsets,'int64'),total
def quantiles(fit,val,test):
 outs=[np.zeros((len(z),fit.shape[1]),'float32') for z in (fit,val,test)]; masks=[z.isna().to_numpy('float32') for z in (fit,val,test)]
 for j,c in enumerate(fit):
  # Exact string values are already represented by the lookup table. PLR is
  # continuous-only; categorical PLR inputs stay zero.
  if not pd.api.types.is_numeric_dtype(fit[c]): continue
  ok=fit[c].notna(); q=QuantileTransformer(n_quantiles=min(1000,int(ok.sum())),output_distribution='normal',subsample=300000,random_state=SEED+j); q.fit(fit.loc[ok,[c]])
  for out,z in zip(outs,(fit,val,test)):
   good=z[c].notna(); out[good.to_numpy(),j]=q.transform(z.loc[good,[c]]).ravel().astype('float32')
 return outs,masks
class PLR(nn.Module):
 def __init__(self,n,k,d): super().__init__(); self.f=nn.Parameter(torch.randn(n,k)*.5); self.w=nn.Parameter(torch.randn(n,2*k,d)/math.sqrt(2*k)); self.b=nn.Parameter(torch.zeros(n,d))
 def forward(self,x):
  z=2*math.pi*x.unsqueeze(-1)*self.f.unsqueeze(0); return torch.einsum('bfk,fkd->bfd',torch.cat([z.sin(),z.cos()],-1),self.w)+self.b
class Net(nn.Module):
 def __init__(self,total,nraw,nder,d=128):
  super().__init__(); self.emb=nn.Embedding(total,d); nn.init.normal_(self.emb.weight,std=.02); self.pc=PLR(nraw,24,d); self.pd=PLR(nder,24,d); self.cls=nn.Parameter(torch.zeros(1,1,d)); self.pos=nn.Parameter(torch.randn(1,1+nraw+nder,d)*.02); enc=nn.TransformerEncoderLayer(d,8,d*2,.1,activation='gelu',batch_first=True,norm_first=True); self.tr=nn.TransformerEncoder(enc,4); self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,1))
 def forward(self,i,cn,cm,dn,dm):
  tc=self.emb(i)+self.pc(cn)*(1-cm).unsqueeze(-1); td=self.pd(dn)*(1-dm).unsqueeze(-1); z=torch.cat([self.cls.expand(len(i),-1,-1),tc,td],1)+self.pos; return self.head(self.tr(z)[:,0]).squeeze(-1)
def run_fold(fit,val,rawtr,rawte,dertr,derte,y,fold):
 (ii,iv,it),offs,total=lookups(rawtr.iloc[fit],rawtr.iloc[val],rawte); (ci,cv,ct),(mi,mv,mt)=quantiles(rawtr.iloc[fit],rawtr.iloc[val],rawte); (di,dv,dt),(dmi,dmv,dmt)=quantiles(dertr.iloc[fit],dertr.iloc[val],derte)
 ts=lambda x:torch.from_numpy(x); II,IV,IT=map(ts,(ii,iv,it)); CI,CV,CT=map(ts,(ci,cv,ct)); MI,MV,MT=map(ts,(mi,mv,mt)); DI,DV,DT=map(ts,(di,dv,dt)); DMI,DMV,DMT=map(ts,(dmi,dmv,dmt)); Y=torch.from_numpy(y[fit]); OFF=torch.from_numpy(offs).to(DEV)
 torch.manual_seed(SEED+fold); model=Net(total,rawtr.shape[1],dertr.shape[1]).to(DEV); emb=[p for n,p in model.named_parameters() if n.startswith('emb')]; rest=[p for n,p in model.named_parameters() if not n.startswith('emb')]; opt=torch.optim.AdamW([{'params':rest,'weight_decay':1e-5},{'params':emb,'weight_decay':3e-4}],lr=2e-3); bs=2048; sched=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=math.ceil(len(fit)/bs)*EPOCHS,pct_start=.15); scaler=torch.cuda.amp.GradScaler(enabled=SCALER); lossf=nn.BCEWithLogitsLoss(); params=list(model.parameters()); ema=[p.detach().clone() for p in params]
 def pred(a,b,c,d,e):
  model.eval(); out=[]
  with torch.no_grad():
   for q in range(0,len(a),8192):
    with torch.autocast(device_type=DEV,dtype=AMP,enabled=DEV=='cuda'): out.append(model(a[q:q+8192].to(DEV),b[q:q+8192].to(DEV),c[q:q+8192].to(DEV),d[q:q+8192].to(DEV),e[q:q+8192].to(DEV)).float().cpu())
  return torch.cat(out).numpy()
 best=-1; bestw=None; bad=0; gen=torch.Generator().manual_seed(SEED+fold)
 for ep in range(EPOCHS):
  model.train(); perm=torch.randperm(len(fit),generator=gen)
  for q in range(0,len(fit),bs):
   sl=perm[q:q+bs]; ids=II[sl].to(DEV); cm=MI[sl].to(DEV); drop=torch.rand(ids.shape,device=DEV)<.10; ids=torch.where(drop,OFF.expand_as(ids),ids); cm=torch.maximum(cm,drop.float())
   with torch.autocast(device_type=DEV,dtype=AMP,enabled=DEV=='cuda'): loss=lossf(model(ids,CI[sl].to(DEV),cm,DI[sl].to(DEV),DMI[sl].to(DEV)),Y[sl].to(DEV))
   opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(params,1.); scaler.step(opt); scaler.update(); sched.step()
   with torch.no_grad():
    torch._foreach_mul_(ema,.999); torch._foreach_add_(ema,[p.detach() for p in params],alpha=.001)
  if ep>=5 and (ep%2==1 or ep==EPOCHS-1):
   bak=[p.detach().clone() for p in params]
   with torch.no_grad():
    for p,e in zip(params,ema): p.copy_(e)
   auc=roc_auc_score(y[val],pred(IV,CV,MV,DV,DMV)); print(f'fold {fold} epoch {ep} auc {auc:.8f}',flush=True)
   if auc>best: best,bestw,bad=auc,[e.clone() for e in ema],0
   else: bad+=1
   with torch.no_grad():
    for p,z in zip(params,bak): p.copy_(z)
   if bad>=5: break
 with torch.no_grad():
  for p,e in zip(params,bestw): p.copy_(e)
 pv=pred(IV,CV,MV,DV,DMV); pt=pred(IT,CT,MT,DT,DMT); del model; torch.cuda.empty_cache(); gc.collect(); return pv,pt,best
random.seed(SEED); np.random.seed(SEED); tp,sp=locate(); tr=pd.read_csv(tp); te=pd.read_csv(sp); feats=[c for c in te if c!=ID]; rawtr,rawte=tr[feats],te[feats]; dertr,derte=deriv(rawtr),deriv(rawte); y=tr[TARGET].to_numpy('float32'); outer=StratifiedKFold(FOLDS,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); tests=[]; fid=np.zeros(len(tr),'int8'); rows=[]; t0=time.time()
for fold,(a,b) in enumerate(outer.split(tr,y),1): fid[b]=fold; pv,pt,best=run_fold(a,b,rawtr,rawte,dertr,derte,y,fold); oof[b]=pv; tests.append(pt); rows.append({'fold':fold,'auc':roc_auc_score(y[b],pv),'best_ema_auc':best}); print(rows[-1])
out=Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('artifacts/local_lookup_v2'); out.mkdir(parents=True,exist_ok=True)
pd.DataFrame({ID:tr[ID],'fold':fid,'y':y,'pred':oof}).to_csv(out/'oof_lookup_v2.csv',index=False); pd.DataFrame({ID:te[ID],TARGET:np.mean(tests,axis=0)}).to_csv(out/'test_lookup_v2.csv',index=False); s=[r['auc'] for r in rows]; metrics={'concept_credit':'Tamerlan Omralinov lookup-transformer concept','implementation':'independent fold-safe PyTorch source-capacity adaptation','official_data_only':True,'seed':SEED,'folds':FOLDS,'fold_metrics':rows,'fold_mean':float(np.mean(s)),'fold_std':float(np.std(s)),'oof_auc':roc_auc_score(y,oof),'runtime_seconds':time.time()-t0,'submission_created':False}; (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
