from pathlib import Path
import gc,json,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
TARGET,ID,SEED="addicted_label","id",20260803
def locate():
 for root in (Path('/kaggle/input'),Path('.')):
  for p in root.rglob('train.csv'):
   try: cols=pd.read_csv(p,nrows=2).columns
   except: continue
   if TARGET in cols and (p.parent/'test.csv').exists(): return p,p.parent/'test.csv'
 raise FileNotFoundError('official tables not found')
def vk(s): return s.astype('string').fillna('__NA__')
def rk(s): return pd.to_numeric(s,errors='coerce').round(1).astype('string').fillna('__NA__')
def fk(s): return np.floor(pd.to_numeric(s,errors='coerce')).astype('Int64').astype('string').fillna('__NA__')
def views(df):
 x=df.drop(columns=[ID,TARGET],errors='ignore').copy(); raw=list(x); miss=x.isna(); base=x.copy()
 base['missing_count']=miss.sum(1).astype('int8')
 for c in raw: base[c+'__missing']=miss[c].astype('int8')
 base['leisure_hours']=base.social_media_hours+base.gaming_hours
 base['unaccounted_screen']=base.daily_screen_time_hours-base.leisure_hours-base.work_study_hours
 base['weekend_gap']=base.weekend_screen_time-base.daily_screen_time_hours
 keys={c+'__raw':vk(x[c]) for c in raw}
 for c in x.select_dtypes('number'):
  keys[c+'__r1']=rk(x[c]); keys[c+'__floor']=fk(x[c])
 pairs=[('daily_screen_time_hours','weekend_screen_time'),('social_media_hours','gaming_hours'),('daily_screen_time_hours','social_media_hours'),('stress_level','academic_work_impact')]
 for a,b in pairs:
  for tag,fn in (('raw',vk),('r1',rk),('floor',fk)): keys[f'pair__{a}__{b}__{tag}']=fn(x[a])+'|'+fn(x[b])
 return base.replace([np.inf,-np.inf],np.nan),pd.DataFrame(keys)
def smap(s,y,smooth):
 z=pd.DataFrame({'k':vk(s).to_numpy(),'y':np.asarray(y)}); st=z.groupby('k',observed=True).y.agg(['sum','count']); pr=float(np.mean(y))
 return ((st['sum']+smooth*pr)/(st['count']+smooth)).to_dict(),st['count'].to_dict(),pr
def encode(fb,kb,y,vb,kv,tb,kt,fold):
 a,b,c=fb.reset_index(drop=True).copy(),vb.reset_index(drop=True).copy(),tb.reset_index(drop=True).copy(); keys=[z.reset_index(drop=True) for z in (kb,kv,kt)]; yy=np.asarray(y); inner=StratifiedKFold(4,shuffle=True,random_state=SEED+fold)
 for col in keys[0]:
  smooth=60. if col.startswith('pair__') else 35.; oo=np.full(len(a),yy.mean(),dtype='float32')
  for ii,jj in inner.split(np.zeros(len(a)),yy):
   mp,_,pr=smap(keys[0].iloc[ii][col],yy[ii],smooth); oo[jj]=vk(keys[0].iloc[jj][col]).map(mp).fillna(pr).to_numpy('float32')
  mp,ct,pr=smap(keys[0][col],yy,smooth)
  a[col+'__te']=oo; b[col+'__te']=vk(keys[1][col]).map(mp).fillna(pr).to_numpy('float32'); c[col+'__te']=vk(keys[2][col]).map(mp).fillna(pr).to_numpy('float32')
  for frame,keydf in zip((a,b,c),keys): frame[col+'__logfreq']=np.log1p(vk(keydf[col]).map(ct).fillna(0)).to_numpy('float32')
 for col in a.select_dtypes(exclude='number'):
  levels=pd.Index(pd.concat([vk(a[col]),vk(b[col]),vk(c[col])],ignore_index=True).unique()); dtype=pd.CategoricalDtype(levels)
  for frame in (a,b,c): frame[col]=vk(frame[col]).astype(dtype)
 return a,b,c,list(a.select_dtypes(exclude='number'))
trp,tep=locate(); tr,te=pd.read_csv(trp),pd.read_csv(tep); y=tr[TARGET].to_numpy('int8'); xb,kb=views(tr); xt,kt=views(te)
outer=StratifiedKFold(5,shuffle=True,random_state=SEED); oof=np.zeros(len(tr)); folds=np.zeros(len(tr),dtype='int8'); tests=[]; rows=[]
for fold,(fi,vi) in enumerate(outer.split(xb,y),1):
 a,b,c,cats=encode(xb.iloc[fi],kb.iloc[fi],y[fi],xb.iloc[vi],kb.iloc[vi],xt,kt,fold)
 m=lgb.LGBMClassifier(objective='binary',n_estimators=3000,learning_rate=.025,num_leaves=31,max_depth=7,min_child_samples=220,subsample=.85,subsample_freq=1,colsample_bytree=.82,reg_alpha=.25,reg_lambda=7.,random_state=SEED+fold,n_jobs=-1,verbosity=-1)
 m.fit(a,y[fi],eval_set=[(b,y[vi])],eval_metric='auc',categorical_feature=cats,callbacks=[lgb.early_stopping(180,verbose=False)])
 pv=m.predict_proba(b)[:,1]; oof[vi]=pv; tests.append(m.predict_proba(c)[:,1]); folds[vi]=fold; rows.append({'fold':fold,'auc':float(roc_auc_score(y[vi],pv)),'best_iteration':int(m.best_iteration_)}); print(rows[-1],flush=True); del a,b,c,m; gc.collect()
pred=np.mean(tests,axis=0); auc=float(roc_auc_score(y,oof)); std=float(np.std([r['auc'] for r in rows],ddof=1))
pd.DataFrame({ID:tr[ID],'fold':folds,'y':y,'pred':oof}).to_csv('/kaggle/working/oof_pair_lattice_lgb.csv',index=False)
pd.DataFrame({ID:te[ID],'pred':pred}).to_csv('/kaggle/working/test_pair_lattice_lgb.csv',index=False)
pd.DataFrame({ID:tr[ID],'fold':folds}).to_csv('/kaggle/working/folds.csv',index=False)
metrics={'provenance':'official competition data only; no public predictions','concept_credit':'Szymon Klapinski / beicicc lattice concept','folds':5,'inner_folds':4,'seed':SEED,'key_count':len(kb.columns),'pair_keys':[c for c in kb if c.startswith('pair__')],'fold_metrics':rows,'oof_auc':auc,'fold_auc_std':std}
Path('/kaggle/working/metrics_pair_lattice_lgb.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2),flush=True)
